import torch.nn as nn
import torch
import torch.nn.functional as F
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


class ZINBDecoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
        )
        self.pi = nn.Linear(hid_dim, out_dim)
        self.disp = nn.Linear(hid_dim, out_dim)
        self.mean = nn.Linear(hid_dim, out_dim)
        self.disp_act = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.mean_act = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb):
        hidden = self.decoder(emb)
        pi = torch.sigmoid(self.pi(hidden))
        disp = self.disp_act(self.disp(hidden))
        mean = self.mean_act(self.mean(hidden))
        return pi, disp, mean


class PlainDecoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, emb):
        return self.decoder(emb)


class HGM(nn.Module):
    def __init__(self, in_dim, hid_dim=128, out_dim=32, use_zinb=False):
        super().__init__()
        self.sencoder = HGNNP(in_dim, hid_dim, out_dim, use_bn=True)
        self.fencoder = HGNNP(in_dim, hid_dim, out_dim, use_bn=True)
        self.attention = Attention(out_dim)
        self.use_zinb = use_zinb
        self.plain_decoder = PlainDecoder(out_dim, hid_dim, in_dim)
        self.zinb = ZINBDecoder(out_dim, hid_dim, in_dim)

    def forward(self, x, shg, fhg):
        zs = self.sencoder(x, shg)
        zf = self.fencoder(x, fhg)
        # z = zs + zf
        z_stack = torch.stack([zs, zf], dim=1)  # 自适应加权融合
        z, att = self.attention(z_stack)
        self.att = att  # 可以把注意力权重保存下来，打印查看空间和特征哪个更重要
        if self.use_zinb:
            pi, disp, mean = self.zinb(z)
            return z, zs, zf, pi, disp, mean
        x_hat = self.plain_decoder(z)
        return z, zs, zf, x_hat
