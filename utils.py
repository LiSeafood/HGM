import torch
import torch.nn.functional as F
import scanpy as sc
import pandas as pd
import numpy as np
import rpy2.robjects as ro
from dhg import Hypergraph
from sklearn.neighbors import NearestNeighbors
import os
import random


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
    adata = sc.read_visium(path, count_file="filtered_feature_bc_matrix.h5")
    adata.var_names_make_unique()
    # 筛选高变基因。这里筛选的个数也有问题，会较大程度上影响性能
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=hvg_num)
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)  # 归一化
    sc.pp.log1p(adata)  # 对数化
    # 标准化。要不要标准化是个问题，在之前的测试中标准化会导致性能下降，可能是因为标准化会抹平一些重要的表达差异。不过也有可能是因为标准化后的数值范围更小，导致模型训练不稳定。后续可以尝试调整学习率或者增加训练轮数来看看是否能提升性能。
    # sc.pp.scale(adata, zero_center=False, max_value=10)
    print("preprocess done, adata.shape:", adata.shape)
    return adata


# 如何确定k1和k2的值呢？
def KnnHyperGraph(adata, k1=8, k2=8):
    spatial = adata.obsm["spatial"]  # (n_spots, 2)
    nn = NearestNeighbors(n_neighbors=k1 + 1, metric="euclidean").fit(spatial)
    indices = nn.kneighbors(return_distance=False)  # shape=(n_spots, k1 + 1)
    shg = Hypergraph(num_v=spatial.shape[0], e_list=indices.tolist())
    print(f"空间超图构建完成: |V|={shg.num_v}, |E|={shg.num_e}, k={k1}")

    genes = np.asarray(adata.X.toarray(), dtype=np.float32, order="C")
    nn = NearestNeighbors(
        n_neighbors=k2 + 1, metric="correlation", algorithm="brute", n_jobs=-1
    ).fit(genes)
    indices = nn.kneighbors(return_distance=False)  # shape=(n_spots, k2 + 1)
    fhg = Hypergraph(num_v=genes.shape[0], e_list=indices.tolist())
    print(f"特征超图构建完成: |V|={fhg.num_v}, |E|={fhg.num_e}, k={k2}")

    return shg, fhg


# ========= 跨视图对比损失（InfoNCE） =========
def infoNCE(p1, p2, temperature=0.2):
    p1 = F.normalize(p1, dim=1)
    p2 = F.normalize(p2, dim=1)
    logits = torch.mm(p1, p2.t()) / temperature  # 相似度矩阵(N, N)
    labels = torch.arange(p1.size(0), device=p1.device)
    # p1->p2 和 p2->p1 对称的对比学习
    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_12 + loss_21)


def cluster_score(adata, z, pca=False, n_neighbors=15, model_name="EEE"):
    """运行 KMeans / mclust / Leiden，并返回分类结果与评估指标。"""
    from rpy2.robjects.packages import importr
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        fowlkes_mallows_score,
    )

    z = z.detach().cpu().numpy()
    y_true = pd.Categorical(adata.obs["ground_truth"]).codes  # 转为整数标签
    true_k = int(np.unique(y_true).size)
    print(f"有效样本数：{len(y_true)} | 真实聚类数：{true_k}")

    # 降维可能导致性能下降
    if pca:
        pca = PCA(n_components=20)
        z_eval = pca.fit_transform(z)
    else:
        z_eval = z

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
