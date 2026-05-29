from utils import *
from model import HGM
from tqdm.auto import tqdm
from sklearn.decomposition import PCA


# 有好多超参数, 可能以后要写config文件来管理这些超参数。有可能根据不同数据的超参数不一样。
class HGMST:
    def __init__(
        self,
        path=None,
        k1=4,
        radius=0,
        k2=8,
        pca=0,
        seed=2020,
        prevalid=False,
        use_zinb=False,
        dgi=0,
        use_fused=False,
        fuse_mode="neighbor_union",
        fusion_head="attention",
        k_fuse=None,
        fusion_iters=1,
        adata=None,
        fused_con_weight=0.0,
        hvg_num=3000,
    ):
        fix_seed(seed)
        if adata is None:
            if path is None:
                raise ValueError("Either path or adata must be provided.")
            self.adata = preprocess(path, hvg_num=hvg_num)
        else:
            self.adata = adata.copy()
        self.prevalid = prevalid
        self.use_zinb = use_zinb
        self.dgi = dgi
        self.use_fused = use_fused
        self.fuse_mode = fuse_mode
        self.fusion_head = fusion_head
        self.k_fuse = k_fuse
        self.fusion_iters = fusion_iters
        self.fused_con_weight = fused_con_weight
        # 猜测：先去空值再去训练更极端（好的更好差的更差），先训练再去空值更平滑。
        if self.prevalid:  # 先去空值再去训练
            valid_mask = ~pd.isnull(self.adata.obs["ground_truth"])  # 去空值
            self.adata = self.adata[valid_mask]
        if self.use_fused:
            self.shg, self.fhg, self.hfg = KnnHyperGraph(
                self.adata,
                k1=k1,
                radius=radius,
                k2=k2,
                pca=pca,
                fuse_mode=fuse_mode,
                k_fuse=k_fuse,
                fusion_iters=fusion_iters,
            )
        else:
            self.shg, self.fhg = KnnHyperGraph(
                self.adata, k1=k1, radius=radius, k2=k2, pca=pca
            )
            self.hfg = None
        self.feature = torch.tensor(self.adata.X.toarray(), dtype=torch.float32)
        # self.counts = torch.tensor(
        #     self.adata.layers["counts"].toarray(), dtype=torch.float32
        # )
        self.model = HGM(
            in_dim=self.feature.shape[1],
            use_zinb=use_zinb,
            use_fused=use_fused,
            fusion_head=fusion_head,
        )
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.feature = self.feature.cuda()
            # self.counts = self.counts.cuda()
            self.shg = self.shg.to(device="cuda")
            self.fhg = self.fhg.to(device="cuda")
            if self.hfg is not None:
                self.hfg = self.hfg.to(device="cuda")

    def train(self, epochs=100):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-3, weight_decay=5e-4
        )

        # 超参数暂时设置成这样，待后续调整
        alpha = 1.0
        beta = 0.1
        temperature = 0.2

        self.model.train()
        for epoch in tqdm(range(1, epochs + 1)):
            optimizer.zero_grad()
            outputs = self.model(self.feature, self.shg, self.fhg, self.hfg)
            z = outputs[0]
            if self.use_zinb:
                _, zs, zf, zg, pi, theta, mean = outputs
                loss_re = zinb_loss(self.counts, pi, theta, mean)  # 重建损失
            else:
                _, zs, zf, zg, x_hat = outputs
                loss_re = F.mse_loss(x_hat, self.feature)  # 重建损失

            # 基础的视图间对比
            loss_con = infoNCE(zs, zf, temperature=temperature)
            # 如果使用融合视图，额外把 zs,zf 拉向 zg
            if self.use_fused and (zg is not None):
                loss_con_fg = infoNCE(zs, zg, temperature=temperature) + infoNCE(zf, zg, temperature=temperature)
                loss_con = loss_con + self.fused_con_weight * loss_con_fg
            loss_dgi = self.feature.new_tensor(0.0)
            if self.dgi:
                perm = torch.randperm(self.feature.size(0), device=self.feature.device)
                x_corrupt = self.feature[perm]
                z_corrupt, _, _, _ = self.model.encode(
                    x_corrupt, self.shg, self.fhg, hfg=self.hfg, update_att=False
                )
                if zg is None:
                    raise ValueError("DGI with zg anchor requires use_fused=True")
                pos_logits, neg_logits = self.model.dgi_logits(z, z_corrupt, zg)
                logits = torch.cat([pos_logits, neg_logits], dim=0)
                labels = torch.cat(
                    [
                        torch.ones_like(pos_logits),
                        torch.zeros_like(neg_logits),
                    ],
                    dim=0,
                )
                loss_dgi = F.binary_cross_entropy_with_logits(logits, labels)
            loss = alpha * loss_re + beta * loss_con + self.dgi * loss_dgi
            loss.backward()
            optimizer.step()
            # if epoch % epochs == 0:
            #     tqdm.write(
            #         f"Epoch {epoch:3d} | recon={loss_re.item():.6f} | contrast={loss_con.item():.6f} | total={loss.item():.6f}"
            #     )

    def eval(self, show=False, refine_radius=240.0):
        self.model.eval()
        with torch.no_grad():
            z = self.model(self.feature, self.shg, self.fhg, self.hfg)[0]
        adata = self.adata.copy()
        z = z.detach().cpu().numpy()
        pca = PCA(n_components=20)
        z = pca.fit_transform(z)  # 先降维再去空值，效果可能会更好，待验证
        if not self.prevalid:  # 先训练再去空值
            valid_mask = ~pd.isnull(self.adata.obs["ground_truth"])  # 去空值
            adata = adata[valid_mask]
            z = z[valid_mask]
        cluster_df, res_df = cluster_score(
            adata,
            z,
            refine_radius=refine_radius,
        )
        if show:
            print("聚类方法评估结果:")
            print(res_df.round(4))
        return cluster_df, res_df
