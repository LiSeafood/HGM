import torch
import torch.nn.functional as F
import scanpy as sc
import pandas as pd
import numpy as np
from dhg import Hypergraph
from sklearn.neighbors import NearestNeighbors


def KnnHyperGraph(adata, k1=8, k2=8):
    spatial = adata.obsm["spatial"]  # (n_spots, 2)
    nn = NearestNeighbors(n_neighbors=k1 + 1, metric="euclidean").fit(spatial)
    indices = nn.kneighbors(spatial, return_distance=False)  # shape=(n_spots, k1 + 1)
    shg = Hypergraph(num_v=spatial.shape[0], e_list=indices.tolist())
    print(f"空间超图构建完成: |V|={shg.num_v}, |E|={shg.num_e}, k={k1}")

    genes = adata.X.toarray()  # (n_spots, n_genes)
    nn = NearestNeighbors(n_neighbors=k2 + 1, metric="correlation").fit(genes)
    indices = nn.kneighbors(genes, return_distance=False)  # shape=(n_spots, k2 + 1)
    fhg = Hypergraph(num_v=genes.shape[0], e_list=indices.tolist())
    print(f"特征超图构建完成: |V|={fhg.num_v}, |E|={fhg.num_e}, k={k2}")

    return shg, fhg


def preprocess(path):
    adata = sc.read_visium(
        path, count_file="filtered_feature_bc_matrix.h5", load_images=True
    )
    adata.var_names_make_unique()
    sc.pp.highly_variable_genes(
        adata, flavor="seurat_v3", n_top_genes=3000
    )  # 筛选3000个高变基因
    adata = adata[:, adata.var["highly_variable"]].copy()
    label = pd.read_table(path + "/metadata.tsv")
    valid = ~pd.isnull(label["layer_guess_reordered"])  # 去空值
    adata = adata[valid].copy()
    label = label[valid].copy()
    adata.obs["ground_truth"] = label["layer_guess_reordered"].values
    sc.pp.normalize_total(adata, target_sum=1e4)  # 归一化
    sc.pp.log1p(adata)  # 对数化
    # sc.pp.scale(adata, zero_center=False, max_value=10)
    print("preprocess done, adata.shape:", adata.shape)
    return adata


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


def cluster_score(adata, z, n_neighbors=15, model_name="EEE"):
    """运行 KMeans / mclust / Leiden，并返回分类结果与评估指标。"""
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        fowlkes_mallows_score,
    )

    z = z.detach().cpu().numpy()
    y_true = pd.Categorical(adata.obs["ground_truth"]).codes # 转为整数标签
    true_k = int(np.unique(y_true).size)
    print(f"有效样本数：{len(y_true)} | 真实聚类数：{true_k}")
    pca = PCA(n_components=20)
    z_eval = pca.fit_transform(z)

    # 1) KMeans
    km_labels = KMeans(n_clusters=true_k, random_state=0, n_init=20).fit_predict(z_eval)

    # 2) mclust
    np.random.seed(2020)
    ro.r["set.seed"](2020)
    importr("mclust")
    rmclust = ro.r["Mclust"]
    z_64 = np.asarray(z_eval, dtype=np.float64)
    r_cols = {f"PC{i+1}": ro.FloatVector(z_64[:, i]) for i in range(z_eval.shape[1])}
    res = rmclust(ro.DataFrame(r_cols), ro.IntVector([true_k]), model_name)
    mclust_labels = np.asarray(res.rx2("classification"), dtype=int) - 1

    # 3) Leiden（按目标簇数搜索最优 resolution）
    adata_eval = adata.copy()
    adata_eval.obsm["X_hgst"] = z_eval
    sc.pp.neighbors(adata_eval, use_rep="X_hgst", n_neighbors=n_neighbors)
    best_diff = 10**9
    best_res = None
    best_labels = None
    for reso in np.linspace(0.1, 4.0, 79):
        sc.tl.leiden(
            adata_eval, resolution=float(reso), random_state=0, key_added="leiden_tmp"
        )
        cur_labels = adata_eval.obs["leiden_tmp"].to_numpy()
        cur_k = pd.Series(cur_labels).nunique()
        diff = abs(cur_k - true_k)
        if diff < best_diff:
            best_diff = diff
            best_res = float(reso)
            best_labels = cur_labels.copy()
        if diff == 0:
            break
    leiden_labels = pd.Categorical(best_labels).codes

    # 4) 分类结果表
    cluster_df = pd.DataFrame(index=adata.obs_names)
    cluster_df["ground_truth"] = adata.obs["ground_truth"].astype(str).values
    cluster_df["kmeans"] = km_labels
    cluster_df["mclust"] = mclust_labels
    cluster_df["leiden"] = leiden_labels

    # 5) 评估指标
    def eval_scores(y_t, y_p):
        return {
            "ARI": adjusted_rand_score(y_t, y_p),
            "NMI": normalized_mutual_info_score(y_t, y_p),
            "FMI": fowlkes_mallows_score(y_t, y_p),
        }

    results = {
        "KMeans": eval_scores(y_true, km_labels),
        "mclust/GMM": eval_scores(y_true, mclust_labels),
        "Leiden": eval_scores(y_true, leiden_labels),
    }
    res_df = pd.DataFrame(results).T[["ARI", "NMI", "FMI"]]

    return cluster_df, res_df
