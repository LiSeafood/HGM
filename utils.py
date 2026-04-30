import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models, transforms
import scanpy as sc
import pandas as pd
import numpy as np
import rpy2.robjects as ro
from dhg import Hypergraph
from sklearn.neighbors import NearestNeighbors
import os
import cv2
import random
import importlib
from PIL import Image
import gc

# 别忘了DLPFC后期可能还要做一个refinement的步骤，取邻域类


def fix_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    ro.r["set.seed"](seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def preprocess(path, hvg_num=3000, img_model_name="vit"):
    adata = sc.read_visium(path, count_file="filtered_feature_bc_matrix.h5")
    adata.var_names_make_unique()
    # 筛选高变基因。这里筛选的个数也有问题，会较大程度上影响性能
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=hvg_num)
    adata = adata[:, adata.var["highly_variable"]].copy()
    label = pd.read_table(os.path.join(path, "metadata.tsv"))
    adata.obs["ground_truth"] = label["layer_guess_reordered"].values
    adata.obsm["img_emb"] = encode_img(adata.obsm["spatial"], path, img_model_name)
    sc.pp.normalize_total(adata, target_sum=1e4)  # 归一化
    sc.pp.log1p(adata)  # 对数化
    # 标准化。标准化之后似乎更加极端了，好的更加好坏的更加坏了。
    # sc.pp.scale(adata, zero_center=False, max_value=10)
    print("preprocess done, X.shape:", adata.shape)
    return adata


# 如何确定k1\k2\k3的值呢？构建超图的metric也值得商榷
def KnnHyperGraph(adata, k1=8, k2=8, k3=3):
    spatial = adata.obsm["spatial"]  # (n_spots, 2)
    nn = NearestNeighbors(n_neighbors=k1 + 1, metric="euclidean").fit(spatial)
    indices = nn.kneighbors(spatial, return_distance=False)  # shape=(n_spots, k1 + 1)
    shg = Hypergraph(num_v=spatial.shape[0], e_list=indices.tolist())

    genes = np.asarray(adata.X.toarray(), dtype=np.float32, order="C")
    nn = NearestNeighbors(
        n_neighbors=k2 + 1, metric="correlation", algorithm="brute", n_jobs=-1
    ).fit(genes)
    indices = nn.kneighbors(genes, return_distance=False)  # shape=(n_spots, k2 + 1)
    fhg = Hypergraph(num_v=genes.shape[0], e_list=indices.tolist())

    img_emb = adata.obsm["img_emb"]
    nn = NearestNeighbors(n_neighbors=k3 + 1, metric="cosine").fit(img_emb)
    indices = nn.kneighbors(img_emb, return_distance=False)  # shape=(n_spots, k3 + 1)
    ihg = Hypergraph(num_v=img_emb.shape[0], e_list=indices.tolist())
    print(
        f"spatial hypergraph: |E|={shg.num_e}, k={k1}, gene hypergraph: |E|={fhg.num_e}, k={k2}, img hypergraph: |E|={ihg.num_e}, k={k3}"
    )
    return shg, fhg, ihg


def infoNCE(p1, p2, temperature=0.2):
    """跨视图对比损失。p1和p2是两个视图的表示，shape=(N, d)。"""
    p1 = F.normalize(p1, dim=1)
    p2 = F.normalize(p2, dim=1)
    logits = torch.mm(p1, p2.t()) / temperature  # 相似度矩阵(N, N)
    labels = torch.arange(p1.size(0), device=p1.device)
    # p1->p2 和 p2->p1 对称的对比学习
    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_12 + loss_21)


def compute_contrastive_loss(
    zs,
    zf,
    zi,
    mode="pairwise",
    temperature=0.2,
    lambda_morph=1.0,
    lambda_cross=1.0,
    lambda_gene_spatial=1.0,
    lambda_gene_image=1.0,
    lambda_common=1.0,
    pairwise_w_sf=1.0,
    pairwise_w_si=1.0,
    pairwise_w_fi=1.0,
    pairwise_eps=1e-12,
    pairwise_logits=None,
    prior_weights=(0.45, 0.10, 0.45),
    lambda_prior=0.1,
):
    """三分支对比损失。

    mode='pairwise': 三个分支两两对称对比（可通过pairwise权重控制）。
    mode='pairwise_global': 三个分支两两对比，权重由3个全局可学习参数经softmax得到。
    mode='pairwise_global_prior': 在pairwise_global基础上加入到先验分布的KL约束。
    mode='hierarchical': 先做(空间,图像)形态内部对齐，再与基因分支跨模态对齐。
    mode='gene_anchor': pairwise的特例，等价于设置w_si=0，仅保留(gene,spatial)与(gene,image)。
    mode='common_center': 先融合三分支得到共同表示，再让三个分支分别向共同表示对齐。
    """
    if mode in ["pairwise", "gene_anchor", "pairwise_global", "pairwise_global_prior"]:
        loss_sf = infoNCE(zs, zf, temperature=temperature)
        loss_si = infoNCE(zs, zi, temperature=temperature)
        loss_fi = infoNCE(zf, zi, temperature=temperature)
        if mode in ["pairwise_global", "pairwise_global_prior"]:
            if pairwise_logits is None:
                raise ValueError(f"{mode} mode requires pairwise_logits")
            weights = torch.softmax(pairwise_logits, dim=0)
            w_sf, w_si, w_fi = weights[0], weights[1], weights[2]
            base_loss = w_sf * loss_sf + w_si * loss_si + w_fi * loss_fi
            if mode == "pairwise_global_prior":
                prior = torch.tensor(
                    prior_weights, dtype=weights.dtype, device=weights.device
                )
                prior = torch.clamp(prior, min=pairwise_eps)
                prior = prior / prior.sum()
                kl = torch.sum(
                    weights
                    * (
                        torch.log(weights + pairwise_eps)
                        - torch.log(prior + pairwise_eps)
                    )
                )
                loss = base_loss + lambda_prior * kl
            else:
                kl = torch.tensor(0.0, dtype=weights.dtype, device=weights.device)
                loss = base_loss
        elif mode == "gene_anchor":
            w_sf = lambda_gene_spatial
            w_si = 0.0
            w_fi = lambda_gene_image
            denom = w_sf + w_si + w_fi + pairwise_eps
            loss = (w_sf * loss_sf + w_si * loss_si + w_fi * loss_fi) / denom
            kl = torch.tensor(0.0, dtype=loss_sf.dtype, device=loss_sf.device)
        else:
            w_sf = pairwise_w_sf
            w_si = pairwise_w_si
            w_fi = pairwise_w_fi
            denom = w_sf + w_si + w_fi + pairwise_eps
            loss = (w_sf * loss_sf + w_si * loss_si + w_fi * loss_fi) / denom
            kl = torch.tensor(0.0, dtype=loss_sf.dtype, device=loss_sf.device)

        w_sf_log = (
            float(w_sf.detach().item())
            if isinstance(w_sf, torch.Tensor)
            else float(w_sf)
        )
        w_si_log = (
            float(w_si.detach().item())
            if isinstance(w_si, torch.Tensor)
            else float(w_si)
        )
        w_fi_log = (
            float(w_fi.detach().item())
            if isinstance(w_fi, torch.Tensor)
            else float(w_fi)
        )
        detail = {
            "loss_sf": float(loss_sf.item()),
            "loss_si": float(loss_si.item()),
            "loss_fi": float(loss_fi.item()),
            "w_sf": w_sf_log,
            "w_si": w_si_log,
            "w_fi": w_fi_log,
            "kl_prior": float(kl.detach().item()),
        }
        return loss, detail

    if mode == "hierarchical":
        # 第一层：空间-图像内部对齐
        loss_morph = infoNCE(zs, zi, temperature=temperature)
        # 第二层：联合形态嵌入与基因特征跨模态对齐
        z_morph = 0.5 * (zs + zi)
        loss_cross = infoNCE(z_morph, zf, temperature=temperature)
        loss = lambda_morph * loss_morph + lambda_cross * loss_cross
        detail = {
            "loss_morph": float(loss_morph.item()),
            "loss_cross": float(loss_cross.item()),
        }
        return loss, detail

    if mode == "common_center":
        z_common = (zs + zf + zi) / 3.0
        loss_s_common = infoNCE(zs, z_common, temperature=temperature)
        loss_f_common = infoNCE(zf, z_common, temperature=temperature)
        loss_i_common = infoNCE(zi, z_common, temperature=temperature)
        loss = lambda_common * (loss_s_common + loss_f_common + loss_i_common) / 3.0
        detail = {
            "loss_s_common": float(loss_s_common.item()),
            "loss_f_common": float(loss_f_common.item()),
            "loss_i_common": float(loss_i_common.item()),
        }
        return loss, detail

    raise ValueError(f"Unsupported contrast mode: {mode}")


def cluster_score(adata, z_eval, n_neighbors=15, model_name="EEE"):
    """运行 KMeans / mclust / Leiden，并返回分类结果与评估指标。"""
    from rpy2.robjects.packages import importr
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        fowlkes_mallows_score,
    )

    y_true = pd.Categorical(adata.obs["ground_truth"]).codes  # 转为整数标签
    true_k = int(np.unique(y_true).size)
    print(f"有效样本数：{len(y_true)} | 真实聚类数：{true_k}")

    # 1) KMeans
    # km_labels = KMeans(n_clusters=true_k, random_state=0, n_init=20).fit_predict(z_eval)

    # 2) mclust
    importr("mclust")
    rmclust = ro.r["Mclust"]
    z_64 = np.asarray(z_eval, dtype=np.float64)
    r_cols = {f"PC{i+1}": ro.FloatVector(z_64[:, i]) for i in range(z_eval.shape[1])}
    res = rmclust(ro.DataFrame(r_cols), ro.IntVector([true_k]), model_name)
    mclust_labels = np.asarray(res.rx2("classification"), dtype=int) - 1

    # 3) Leiden（按目标簇数搜索最优 resolution）
    # adata_eval = adata.copy()
    # adata_eval.obsm["X_hgst"] = z_eval
    # sc.pp.neighbors(adata_eval, use_rep="X_hgst", n_neighbors=n_neighbors)
    # best_diff = 10**9
    # best_res = None
    # best_labels = None
    # for reso in np.linspace(0.1, 4.0, 79):
    #     sc.tl.leiden(
    #         adata_eval, resolution=float(reso), random_state=0, key_added="leiden_tmp"
    #     )
    #     cur_labels = adata_eval.obs["leiden_tmp"].to_numpy()
    #     cur_k = pd.Series(cur_labels).nunique()
    #     diff = abs(cur_k - true_k)
    #     if diff < best_diff:
    #         best_diff = diff
    #         best_res = float(reso)
    #         best_labels = cur_labels.copy()
    #     if diff == 0:
    #         break
    # leiden_labels = pd.Categorical(best_labels).codes

    # 4) 分类结果表
    cluster_df = pd.DataFrame(index=adata.obs_names)
    cluster_df["ground_truth"] = adata.obs["ground_truth"].astype(str).values
    # cluster_df["kmeans"] = km_labels
    cluster_df["mclust"] = mclust_labels
    # cluster_df["leiden"] = leiden_labels

    # 5) 评估指标
    def eval_scores(y_t, y_p):
        return {
            "ARI": adjusted_rand_score(y_t, y_p),
            "NMI": normalized_mutual_info_score(y_t, y_p),
            "FMI": fowlkes_mallows_score(y_t, y_p),
        }

    results = {
        # "KMeans": eval_scores(y_true, km_labels),
        "mclust": eval_scores(y_true, mclust_labels),
        # "Leiden": eval_scores(y_true, leiden_labels),
    }
    res_df = pd.DataFrame(results).T[["ARI", "NMI", "FMI"]]

    return cluster_df, res_df


def _hf_extract_embedding(outputs, repo_id):
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state[:, 0, :]
    raise ValueError(f"Unexpected HF model output format for repo: {repo_id}")


def _hf_vision_embed(subimages, repo_id, device, batch_size=8, use_fp16=True):
    try:
        transformers = importlib.import_module("transformers")
        AutoImageProcessor = getattr(transformers, "AutoImageProcessor")
        AutoModel = getattr(transformers, "AutoModel")
    except ImportError as exc:
        raise ImportError(
            "Using Hugging Face vision models requires 'transformers'. "
            "Please run: pip install transformers"
        ) from exc

    processor = AutoImageProcessor.from_pretrained(
        repo_id, use_fast=True, local_files_only=True
    )
    model = AutoModel.from_pretrained(repo_id, local_files_only=True).to(device).eval()
    pil_images = [Image.fromarray(img) for img in subimages]
    all_emb = []
    start = 0
    cur_bs = max(1, int(batch_size))

    while start < len(pil_images):
        end = min(start + cur_bs, len(pil_images))
        batch_imgs = pil_images[start:end]
        try:
            inputs = processor(images=batch_imgs, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                if device.type == "cuda" and use_fp16:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)
            emb = _hf_extract_embedding(outputs, repo_id)
            all_emb.append(emb.detach().cpu().float().numpy())
            del inputs, outputs, emb
            start = end
        except RuntimeError as e:
            if not (device.type == "cuda" and "out of memory" in str(e).lower()):
                raise
            torch.cuda.empty_cache()
            if cur_bs > 1:
                cur_bs = max(1, cur_bs // 2)
                print(f"[HF OOM] reduce batch_size to {cur_bs} for {repo_id}")
                continue
            print(f"[HF OOM] fallback to CPU for {repo_id}")
            model = model.to("cpu")
            device = torch.device("cpu")
            gc.collect()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return np.concatenate(all_emb, axis=0)


def encode_img(position, path, model_name="vit"):
    img = cv2.imread(os.path.join(path, "spatial", "full_image.tif"))
    if img is None:
        print("load img failed, check the path")
    else:  # OpenCV 读出来是 BGR，显示前转成 RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 以每个spot为核心，截取一个224x224的patch，送入预训练的图像模型提取特征向量。position是spot的坐标。在边界附近会被截断，取不到正方形
    beta_half = 112
    subimages = []
    max_x, max_y = img.shape[0], img.shape[1]
    for i in range(len(position)):
        patch = img[
            max(0, int(position[i][0]) - beta_half) : min(
                max_x, int(position[i][0]) + beta_half + 1
            ),
            max(0, int(position[i][1]) - beta_half) : min(
                max_y, int(position[i][1]) + beta_half + 1
            ),
        ]
        subimages.append(patch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name in ["phikon", "hf_phikon"]:
        return _hf_vision_embed(subimages, "owkin/phikon", device)
    if model_name in ["phikon_v2", "hf_phikon_v2"]:
        return _hf_vision_embed(subimages, "owkin/phikon-v2", device)
    if model_name.startswith("hf:"):
        repo_id = model_name.split("hf:", 1)[1].strip()
        if len(repo_id) == 0:
            raise ValueError("model_name='hf:<repo_id>' requires a non-empty repo_id")
        return _hf_vision_embed(subimages, repo_id, device)

    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Identity()
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Identity()
    elif model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Identity()
    elif model_name == "vit":
        model = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT)
        model.heads = nn.Identity()
    else:
        raise ValueError(
            f"Unsupported model_name: {model_name}. "
            "Try one of: vit, resnet18, resnet50, vgg16, densenet121, "
            "phikon, phikon_v2, hf:<repo_id>"
        )
    encoder = model.to(device).eval()

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    all_emb = []
    bs = 64 if model_name != "vit" else 16
    start = 0
    while start < len(subimages):
        end = min(start + bs, len(subimages))
        batch_imgs = [transform(i) for i in subimages[start:end]]
        input_tensors = torch.stack(batch_imgs, dim=0).to(device)
        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    batch_emb = encoder(input_tensors)
            else:
                batch_emb = encoder(input_tensors)
        all_emb.append(batch_emb.detach().cpu().float().numpy())
        del input_tensors, batch_emb
        start = end
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return np.concatenate(all_emb, axis=0)
