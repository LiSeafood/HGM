import torch.nn as nn
import torch
from dhg.models import HGNNP


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class HGM(nn.Module):
    def __init__(self, in_dim, hid_dim=128, out_dim=32, proj_dim=32):
        super().__init__()
        self.sencoder = HGNNP(in_dim, hid_dim, out_dim, use_bn=True)
        self.fencoder = HGNNP(in_dim, hid_dim, out_dim, use_bn=True)
        self.attention = Attention(out_dim)
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hid_dim, in_dim),
        )
        # 对比学习投影头(常见做法：encoder输出 -> projection space)。需不需要改中间层维度呢？这两层投影头真的有必要吗？
        # self.proj_feature = nn.Sequential(
        #     nn.Linear(out_dim, out_dim),
        #     nn.ReLU(),
        #     nn.Linear(out_dim, proj_dim),
        # )
        # self.proj_spatial = nn.Sequential(
        #     nn.Linear(out_dim, out_dim),
        #     nn.ReLU(),
        #     nn.Linear(out_dim, proj_dim),
        # )

    def forward(self, x, shg, fhg):
        zs = self.sencoder(x, shg)
        zf = self.fencoder(x, fhg)
        # z = zs + zf
        z_stack = torch.stack([zs, zf], dim=1)  # 自适应加权融合
        z, att = self.attention(z_stack)
        self.att = att  # 可以把注意力权重保存下来，打印查看空间和特征哪个更重要
        x_hat = self.decoder(z)
        # 投影到对比学习空间, 用于后续对比学习
        # zs = self.proj_spatial(zs)
        # zf = self.proj_feature(zf)
        return z, zs, zf, x_hat
