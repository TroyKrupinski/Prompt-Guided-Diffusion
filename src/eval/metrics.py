import torch
import torch.nn.functional as F

def mse(a, b):
    return F.mse_loss(a, b)

def cosine_sim(a, b, eps=1e-8):
    # a,b: (B, D) embeddings
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a * b).sum(dim=-1).mean()
