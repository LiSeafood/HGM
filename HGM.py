from utils import *
from model import HGM
from tqdm.auto import tqdm


# 有好多超参数, 可能以后要写config文件来管理这些超参数。有可能根据不同数据的超参数不一样。
class HGMST:
    def __init__(self, path):
        self.adata = preprocess(path)
        self.shg, self.fhg = KnnHyperGraph(self.adata)

    def train(self, epochs=100):
        feature = torch.tensor(self.adata.X.toarray(), dtype=torch.float32)
        self.model = HGM(in_dim=feature.shape[1])
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-3, weight_decay=5e-4
        )
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            feature = feature.cuda()
            self.shg = self.shg.to(device="cuda")
            self.fhg = self.fhg.to(device="cuda")

        # 超参数暂时设置成这样，待后续调整
        alpha = 1.0
        beta = 0.1
        temperature = 0.2

        self.model.train()
        for epoch in tqdm(range(1, epochs + 1)):
            optimizer.zero_grad()
            _, zs, zf, x_hat = self.model(feature, self.shg, self.fhg)
            loss_re = F.mse_loss(x_hat, feature)  # 重建损失
            loss_con = infoNCE(zs, zf, temperature=temperature)  # 对比损失
            loss = alpha * loss_re + beta * loss_con
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                tqdm.write(
                    f"Epoch {epoch:3d} | recon={loss_re.item():.6f} | contrast={loss_con.item():.6f} | total={loss.item():.6f}"
                )

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            z, _, _, _ = self.model(self.feature, self.shg, self.fhg)

        cluster_df, res_df = cluster_score(self.adata, z_eval=z)

        print("\n分类结果（前5行）:")
        print(cluster_df.head(5))

        print("\n聚类方法评估结果:")
        print(res_df.round(4))
