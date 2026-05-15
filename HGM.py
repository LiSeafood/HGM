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
        adata=None,
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
        # 猜测：先去空值再去训练更极端（好的更好差的更差），先训练再去空值更平滑。
        if self.prevalid:  # 先去空值再去训练
            valid_mask = ~pd.isnull(self.adata.obs["ground_truth"])  # 去空值
            self.adata = self.adata[valid_mask]
        self.shg, self.fhg = KnnHyperGraph(
            self.adata, k1=k1, radius=radius, k2=k2, pca=pca
        )
        self.feature = torch.tensor(self.adata.X.toarray(), dtype=torch.float32)
        # self.counts = torch.tensor(
        #     self.adata.layers["counts"].toarray(), dtype=torch.float32
        # )
        self.model = HGM(in_dim=self.feature.shape[1], use_zinb=use_zinb)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.feature = self.feature.cuda()
            # self.counts = self.counts.cuda()
            self.shg = self.shg.to(device="cuda")
            self.fhg = self.fhg.to(device="cuda")

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
            outputs = self.model(self.feature, self.shg, self.fhg)
            if self.use_zinb:
                _, zs, zf, pi, theta, mean = outputs
                loss_re = zinb_loss(self.counts, pi, theta, mean)  # 重建损失
            else:
                _, zs, zf, x_hat = outputs
                loss_re = F.mse_loss(x_hat, self.feature)  # 重建损失
            loss_con = infoNCE(zs, zf, temperature=temperature)  # 对比损失
            loss = alpha * loss_re + beta * loss_con
            loss.backward()
            optimizer.step()
            # if epoch % epochs == 0:
            #     tqdm.write(
            #         f"Epoch {epoch:3d} | recon={loss_re.item():.6f} | contrast={loss_con.item():.6f} | total={loss.item():.6f}"
            #     )

    def eval(self, show=False, refine_radius=240.0):
        self.model.eval()
        with torch.no_grad():
            z = self.model(self.feature, self.shg, self.fhg)[0]
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
