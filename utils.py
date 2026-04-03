import torch
import torch.nn.functional as F

# ========= 跨视图对比损失（InfoNCE） =========
def infoNCE(p1, p2, temperature=0.2):
    p1=F.normalize(p1, dim=1)
    p2=F.normalize(p2, dim=1)
    logits = torch.mm(p1, p2.t()) / temperature # 相似度矩阵(N, N)
    labels = torch.arange(p1.size(0), device=p1.device)
    # p1->p2 和 p2->p1 对称的对比学习
    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_12 + loss_21)
