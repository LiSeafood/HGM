import torch.nn as nn
from dhg.models import HGNNP


class HGM(nn.Module):
    def __init__(self, in_dim, hid_dim=128, out_dim=32, proj_dim=32):
        super().__init__()
        self.sencoder = HGNNP(in_dim, hid_dim, out_dim, use_bn=True)
        self.fencoder = HGNNP(in_dim, hid_dim, out_dim, use_bn=True)
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hid_dim, in_dim),
        )
        # 对比学习投影头（常见做法：encoder输出 -> projection space）
        self.proj_gene = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, proj_dim),
        )
        self.proj_spatial = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, proj_dim),
        )

    def forward(self, x, shg, fhg):
        zs = self.sencoder(x, shg)
        zf = self.fencoder(x, fhg)
        z = zs + zf
        x_hat = self.decoder(z)
        # 投影到对比学习空间
        zs = self.proj_spatial(zs)
        zf = self.proj_gene(zf)
        return z, zs, zf, x_hat
