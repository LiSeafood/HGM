from utils import *
from model import HGM
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd


# 有好多超参数, 可能以后要写config文件来管理这些超参数。有可能根据不同数据的超参数不一样。
class HGMST:
    def __init__(
        self,
        path,
        prevalid=False,
        seed=2020,
        adata=None,
        hypergraph_k=(8, 8, 3),
        prebuilt_hg=None,
        prebuilt_feature=None,
        img_model_name="vit",
    ):
        fix_seed(seed)
        self.seed = seed
        self.path = path
        self.img_model_name = img_model_name
        self.adata = (
            preprocess(path, img_model_name=img_model_name)
            if adata is None
            else adata.copy()
        )
        self.prevalid = prevalid
        # 猜测：先去空值再去训练更极端（好的更好差的更差），先训练再去空值更平滑。
        if self.prevalid:  # 先去空值再去训练
            valid = ~pd.isnull(self.adata.obs["ground_truth"])  # 去空值
            self.adata = self.adata[valid]

        if prebuilt_hg is None:
            self.shg, self.fhg, self.ihg = KnnHyperGraph(
                self.adata, k1=hypergraph_k[0], k2=hypergraph_k[1], k3=hypergraph_k[2]
            )
        else:
            self.shg, self.fhg, self.ihg = prebuilt_hg

        if prebuilt_feature is None:
            toarray_fn = getattr(self.adata.X, "toarray", None)
            x = toarray_fn() if callable(toarray_fn) else self.adata.X
            self.feature = torch.tensor(np.asarray(x), dtype=torch.float32)
        else:
            self.feature = prebuilt_feature.clone().detach()

        self.model = HGM(in_dim=self.feature.shape[1])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.feature = self.feature.cuda()
            self.shg = self.shg.to(device=self.device)
            self.fhg = self.fhg.to(device=self.device)
            self.ihg = self.ihg.to(device=self.device)

    def train(
        self,
        epochs=100,
        alpha=1.0,
        beta=0.1,
        temperature=0.2,
        contrast_mode="pairwise",
        lambda_morph=1.0,
        lambda_cross=1.0,
        pairwise_w_sf=1.0,
        pairwise_w_si=1.0,
        pairwise_w_fi=1.0,
        lambda_gene_spatial=1.0,
        lambda_gene_image=1.0,
        lambda_common=1.0,
        init_pairwise_logits=(0.0, 0.0, 0.0),
        prior_weights=(0.45, 0.10, 0.45),
        lambda_prior=0.1,
        stage1_mode="gene_anchor",
        stage2_mode="pairwise_global_prior",
        stage1_ratio=0.3,
    ):
        fix_seed(self.seed)
        params = list(self.model.parameters())
        self.pairwise_logits = None
        if contrast_mode in [
            "pairwise_global",
            "pairwise_global_prior",
            "two_stage_global",
            "two_stage_global_prior",
        ]:
            self.pairwise_logits = torch.nn.Parameter(
                torch.tensor(
                    init_pairwise_logits, dtype=torch.float32, device=self.device
                )
            )
            params.append(self.pairwise_logits)

        optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=5e-4)

        history = []
        stage1_epochs = max(1, min(epochs - 1, int(round(epochs * stage1_ratio))))

        self.model.train()
        for epoch in tqdm(range(1, epochs + 1)):
            optimizer.zero_grad()
            _, zs, zf, zi, x_hat = self.model(
                self.feature, self.shg, self.fhg, self.ihg
            )
            loss_re = F.mse_loss(x_hat, self.feature)  # 重建损失

            if contrast_mode == "two_stage_global":
                cur_mode = stage1_mode if epoch <= stage1_epochs else "pairwise_global"
            elif contrast_mode == "two_stage_global_prior":
                cur_mode = (
                    stage1_mode if epoch <= stage1_epochs else "pairwise_global_prior"
                )
            elif contrast_mode in ["pairwise_global", "pairwise_global_prior"]:
                cur_mode = contrast_mode
            else:
                cur_mode = contrast_mode

            loss_con, con_detail = compute_contrastive_loss(
                zs,
                zf,
                zi,
                mode=cur_mode,
                temperature=temperature,
                lambda_morph=lambda_morph,
                lambda_cross=lambda_cross,
                pairwise_w_sf=pairwise_w_sf,
                pairwise_w_si=pairwise_w_si,
                pairwise_w_fi=pairwise_w_fi,
                lambda_gene_spatial=lambda_gene_spatial,
                lambda_gene_image=lambda_gene_image,
                lambda_common=lambda_common,
                pairwise_logits=self.pairwise_logits,
                prior_weights=prior_weights,
                lambda_prior=lambda_prior,
            )
            loss = alpha * loss_re + beta * loss_con
            loss.backward()
            optimizer.step()
            history.append(
                {
                    "epoch": epoch,
                    "contrast_mode": cur_mode,
                    "loss_re": float(loss_re.item()),
                    "loss_con": float(loss_con.item()),
                    "loss": float(loss.item()),
                    **con_detail,
                }
            )
            if epoch % epochs == 0:
                tqdm.write(
                    f"Epoch {epoch:3d} | recon={loss_re.item():.6f} | contrast={loss_con.item():.6f} | total={loss.item():.6f}"
                )
        return pd.DataFrame(history)

    def eval(self, show=False):
        fix_seed(self.seed)
        self.model.eval()
        with torch.no_grad():
            z, _, _, _, _ = self.model(self.feature, self.shg, self.fhg, self.ihg)
        adata = self.adata.copy()
        z = z.detach().cpu().numpy()
        pca = PCA(n_components=20)
        z = pca.fit_transform(z)  # 先降维再去空值，效果可能会更好，待验证
        if not self.prevalid:  # 先训练再去空值
            valid = ~pd.isnull(self.adata.obs["ground_truth"])  # 去空值
            adata = adata[valid]
            z = z[valid]
        cluster_df, res_df = cluster_score(adata, z)
        if show:
            print("聚类方法评估结果:")
            print(res_df.round(4))
        return cluster_df, res_df


def compare_contrast_schemes(
    path,
    epochs=100,
    repeats=3,
    seed=2020,
    prevalid=False,
    alpha=1.0,
    beta=0.1,
    temperature=0.2,
    lambda_morph=1.0,
    lambda_cross=1.0,
    pairwise_w_sf=1.0,
    pairwise_w_si=1.0,
    pairwise_w_fi=1.0,
    lambda_gene_spatial=1.0,
    lambda_gene_image=1.0,
    lambda_common=1.0,
    init_pairwise_logits=(0.0, 0.0, 0.0),
    prior_weights=(0.45, 0.10, 0.45),
    lambda_prior=0.1,
    stage1_mode="gene_anchor",
    stage1_ratio=0.3,
    hypergraph_k=(8, 8, 3),
    mode_list=None,
    img_model_name="vit",
):
    """在同一预处理数据上比较多种对比方案（含全局可学习权重版）。"""
    base_adata = preprocess(path, img_model_name=img_model_name)
    if prevalid:
        valid = ~pd.isnull(base_adata.obs["ground_truth"])
        base_adata = base_adata[valid].copy()

    # adata和图像嵌入固定时，超图结构不变；在外部一次性构建并复用
    prebuilt_hg = KnnHyperGraph(
        base_adata, k1=hypergraph_k[0], k2=hypergraph_k[1], k3=hypergraph_k[2]
    )
    toarray_fn = getattr(base_adata.X, "toarray", None)
    x = toarray_fn() if callable(toarray_fn) else base_adata.X
    prebuilt_feature = torch.tensor(np.asarray(x), dtype=torch.float32)

    rows = []
    if mode_list is None:
        mode_list = [
            "pairwise_global",
            "pairwise_global_prior",
            "two_stage_global_prior",
            "gene_anchor",
            "common_center",
        ]

    for mode in mode_list:
        for i in range(repeats):
            cur_seed = seed + i
            runner = HGMST(
                path=path,
                prevalid=prevalid,
                seed=cur_seed,
                adata=base_adata,
                hypergraph_k=hypergraph_k,
                prebuilt_hg=prebuilt_hg,
                prebuilt_feature=prebuilt_feature,
                img_model_name=img_model_name,
            )
            runner.train(
                epochs=epochs,
                alpha=alpha,
                beta=beta,
                temperature=temperature,
                contrast_mode=mode,
                lambda_morph=lambda_morph,
                lambda_cross=lambda_cross,
                pairwise_w_sf=pairwise_w_sf,
                pairwise_w_si=pairwise_w_si,
                pairwise_w_fi=pairwise_w_fi,
                lambda_gene_spatial=lambda_gene_spatial,
                lambda_gene_image=lambda_gene_image,
                lambda_common=lambda_common,
                init_pairwise_logits=init_pairwise_logits,
                prior_weights=prior_weights,
                lambda_prior=lambda_prior,
                stage1_mode=stage1_mode,
                stage1_ratio=stage1_ratio,
            )
            _, res_df = runner.eval(show=False)
            metric = res_df.loc["mclust", ["ARI", "NMI", "FMI"]].to_dict()
            row = {
                "mode": mode,
                "run": i,
                "seed": cur_seed,
                "ARI": float(metric["ARI"]),
                "NMI": float(metric["NMI"]),
                "FMI": float(metric["FMI"]),
            }
            if (
                mode
                in [
                    "pairwise_global",
                    "pairwise_global_prior",
                    "two_stage_global",
                    "two_stage_global_prior",
                ]
                and runner.pairwise_logits is not None
            ):
                w = torch.softmax(runner.pairwise_logits.detach().cpu(), dim=0)
                row.update(
                    {
                        "w_sf": float(w[0].item()),
                        "w_si": float(w[1].item()),
                        "w_fi": float(w[2].item()),
                    }
                )
            rows.append(row)

    detail_df = pd.DataFrame(rows)
    summary_df = (
        detail_df.groupby("mode")[["ARI", "NMI", "FMI"]].agg(["mean", "std"]).round(4)
    )
    return detail_df, summary_df
