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


class DGIDiscriminator(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.bilinear = nn.Bilinear(hid_dim, hid_dim, 1, bias=True)

    def forward(self, z, summary):
        if summary.dim() == 1:
            summary = summary.unsqueeze(0)
        summary = summary.expand(z.size(0), -1)
        return self.bilinear(z, summary).squeeze(-1)


class HGM(nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim=128,
        out_dim=32,
        use_zinb=False,
        use_fused=False,
        fusion_head="attention",
    ):
        super().__init__()
        self.sencoder = HGNNP(in_dim, hid_dim, out_dim, use_bn=True)
        self.fencoder = HGNNP(in_dim, hid_dim, out_dim, use_bn=True)
        self.gencoder = (
            HGNNP(in_dim, hid_dim, out_dim, use_bn=True) if use_fused else None
        )
        self.attention = Attention(out_dim)
        self.use_zinb = use_zinb
        self.use_fused = use_fused
        self.fusion_head = fusion_head
        self.n_views = 2
        if fusion_head == "learnable":
            self.fusion_logits = nn.Parameter(torch.zeros(self.n_views))
        elif fusion_head != "attention":
            raise ValueError(f"Unsupported fusion_head: {fusion_head}")
        self.plain_decoder = PlainDecoder(out_dim, hid_dim, in_dim)
        self.zinb = ZINBDecoder(out_dim, hid_dim, in_dim)
        self.dgi_discriminator = DGIDiscriminator(out_dim)

    def encode(self, x, shg, fhg, hfg=None, update_att=True):
        zs = self.sencoder(x, shg)
        zf = self.fencoder(x, fhg)
        zg = None
        if self.use_fused:
            if hfg is None:
                raise ValueError("hfg must be provided when use_fused=True")
            zg = self.gencoder(x, hfg)
        z_views = [zs, zf]
        if self.fusion_head == "attention":
            z_stack = torch.stack(z_views, dim=1)
            z, att = self.attention(z_stack)
        else:
            weights = torch.softmax(self.fusion_logits, dim=0)
            z = sum(w * v for w, v in zip(weights, z_views))
            att = weights
        if update_att:
            self.att = att
        return z, zs, zf, zg

    def summary(self, z):
        return torch.sigmoid(z.mean(dim=0))

    def dgi_logits(self, z, z_corrupt, anchor):
        pos = self.dgi_discriminator(z, anchor)
        neg = self.dgi_discriminator(z_corrupt, anchor)
        return pos, neg

    def forward(self, x, shg, fhg, hfg=None):
        z, zs, zf, zg = self.encode(x, shg, fhg, hfg=hfg, update_att=True)
        if self.use_zinb:
            pi, disp, mean = self.zinb(z)
            return z, zs, zf, zg, pi, disp, mean
        x_hat = self.plain_decoder(z)
        return z, zs, zf, zg, x_hat
