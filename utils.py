import torch
import torch.nn.functional as F
import scanpy as sc
import pandas as pd
import numpy as np
import rpy2.robjects as ro
from dhg import Hypergraph
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import os
import random
import warnings

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


def preprocess(path, hvg_num=3000):
    with warnings.catch_warnings():  # 屏蔽警告信息
        warnings.filterwarnings(
            "ignore",
            message=r"Use `squidpy\.read\.visium` instead\.",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Variable names are not unique\. To make them unique, call `\.var_names_make_unique`\.",
            category=UserWarning,
        )
        adata = sc.read_visium(path, count_file="filtered_feature_bc_matrix.h5")
    adata.var_names_make_unique()
    # adata.layers["counts"] = adata.X.copy()
    # 筛选高变基因。这里筛选的个数也有问题，会较大程度上影响性能
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=hvg_num)
    adata = adata[:, adata.var["highly_variable"]].copy()
    label = pd.read_table(path + "/metadata.tsv")
    adata.obs["ground_truth"] = label["layer_guess_reordered"].values
    sc.pp.normalize_total(adata, target_sum=1e4)  # 归一化
    sc.pp.log1p(adata)  # 对数化
    # 标准化。标准化之后似乎更加极端了，好的更加好坏的更加坏了。总体来说还是不好
    # sc.pp.scale(adata, zero_center=False, max_value=10)
    # print("preprocess done, adata.shape:", adata.shape)
    return adata


# 如何确定k1和k2的值呢？在模型还未定下来之前，初步实验暂定为4和8
def KnnHyperGraph(adata, k1=4,radius=0, k2=8, pca=0):
    spatial = adata.obsm["spatial"]  # (n_spots, 2)
    genes = np.asarray(adata.X.toarray(), dtype=np.float32, order="C")
    if pca:  # 可选降维以降低构图时的计算量和噪声
        genes = PCA(n_components=pca, random_state=0).fit_transform(genes)
        
    if radius:
        norm = np.linalg.norm(genes, axis=1, keepdims=True)
        genes_norm = genes / (norm + 1e-12)  # L2 归一化后的基因表示，用于候选邻居内的相似度排序
        nn = NearestNeighbors(n_neighbors= radius+1, metric="euclidean").fit(spatial)
        indices = nn.kneighbors(spatial, return_distance=False)  # shape=(n_spots, k1 + 1)
        e_list = []
        for i in range(spatial.shape[0]):
            candidate_idx = indices[i]
            # 2. 计算中心节点 i 与其空间候选邻居的 基因相似度
            center_gene = genes_norm[i]
            neighbor_genes = genes_norm[candidate_idx]
            sim_gene = neighbor_genes @ center_gene
            # 3. 过滤/排序：只保留基因相似度最高的前 k1+1 个邻居，剔除处于边界上的、基因差异大的空间邻居
            kept_idx = candidate_idx[np.argsort(-sim_gene)[: k1 + 1]]
            e_list.append(kept_idx.tolist())
        shg = Hypergraph(num_v=spatial.shape[0], e_list=e_list)
    else:
        nn = NearestNeighbors(n_neighbors= k1 + 1, metric="euclidean").fit(spatial)
        indices = nn.kneighbors(spatial, return_distance=False)  # shape=(n_spots, k1 + 1)
        shg = Hypergraph(num_v=spatial.shape[0], e_list=indices.tolist()) 

    nn = NearestNeighbors(n_neighbors=k2 + 1, metric="correlation").fit(genes)
    indices = nn.kneighbors(genes, return_distance=False)  # shape=(n_spots, k2 + 1)
    fhg = Hypergraph(num_v=genes.shape[0], e_list=indices.tolist())
    # print(
    #     f"spatial hypergraph: |E|={shg.num_e}, k={k1}, feature hypergraph: |E|={fhg.num_e}, k={k2}"
    # )
    return shg, fhg


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


def self_infoNCE(z, z_neg=None, temperature=0.2):
    """自对比损失：自身为正样本，负样本为打乱或外部给定。"""
    z = F.normalize(z, dim=1)
    n = z.size(0)
    if n <= 1:
        return z.new_tensor(0.0)
    if z_neg is None:
        perm = torch.randperm(n, device=z.device)
        z_neg = z[perm]
    else:
        z_neg = F.normalize(z_neg, dim=1)
    pos_sim = torch.sum(z * z, dim=1, keepdim=True) / temperature
    neg_sim = torch.mm(z, z_neg.t()) / temperature
    logits = torch.cat([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(n, dtype=torch.long, device=z.device)
    return F.cross_entropy(logits, labels)


def zinb_loss(x, pi, theta, mean, eps=1e-8):
    """Zero-Inflated Negative Binomial 负对数似然。"""
    x = x.float()
    pi = pi.clamp(min=eps, max=1.0 - eps)
    theta = theta.clamp(min=eps, max=1e6)
    mean = mean.clamp(min=eps, max=1e6)

    nb_case = (
        torch.lgamma(theta + eps)
        + torch.lgamma(x + 1.0)
        - torch.lgamma(x + theta + eps)
        + (theta + x) * torch.log(1.0 + (mean / (theta + eps)))
        + x * (torch.log(theta + eps) - torch.log(mean + eps))
    ) - torch.log(1.0 - pi + eps)

    zero_nb = torch.pow(theta / (theta + mean + eps), theta)
    zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
    result = torch.where(torch.lt(x, 1e-8), zero_case, nb_case)
    return torch.mean(result)


def cluster_score(adata, z_eval, pca=False, n_neighbors=15, model_name="EEE"):
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
    # print(f"有效样本数：{len(y_true)} | 真实聚类数：{true_k}")

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
