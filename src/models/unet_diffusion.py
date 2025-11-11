import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: (B,) int64 timesteps
        returns: (B, dim)
        """
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * -(math.log(10000.0) / half)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x): return self.c(x)

class DiffusionUNet(nn.Module):
    """
    U-Net that predicts epsilon in diffusion.
    Conditioned on (timestep, text embedding).
    """
    def __init__(self, emb_dim=512, base=64, time_dim=512):
        super().__init__()
        self.emb_dim = emb_dim
        self.time_dim = time_dim

        # time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, emb_dim),
        )

        # encoder
        self.c1 = ConvBlock(1, base)
        self.p1 = nn.MaxPool2d(2)
        self.c2 = ConvBlock(base, base * 2)
        self.p2 = nn.MaxPool2d(2)

        # bottleneck
        self.cb = ConvBlock(base * 2, base * 4)

        # decoder
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.c3 = ConvBlock(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.c4 = ConvBlock(base * 2, base)

        # epsilon output
        self.out = nn.Conv2d(base, 1, 1)

        # FiLM-style conditioning (same as your UNetDenoiser, but using combined emb)
        self.gamma1 = nn.Linear(emb_dim, base)
        self.beta1  = nn.Linear(emb_dim, base)
        self.gamma2 = nn.Linear(emb_dim, base*2)
        self.beta2  = nn.Linear(emb_dim, base*2)
        self.gammaB = nn.Linear(emb_dim, base*4)
        self.betaB  = nn.Linear(emb_dim, base*4)

    def film(self, x, gamma, beta):
        g = gamma.unsqueeze(-1).unsqueeze(-1)
        b = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + g) + b

    def _align(self, y, ref):
        if y.shape[-2:] != ref.shape[-2:]:
            y = F.interpolate(y, size=ref.shape[-2:], mode="nearest")
        return y

    def forward(self, x_t, t, txt_emb):
        """
        x_t: (B,1,M,T) noisy mel
        t: (B,) timesteps
        txt_emb: (B,emb_dim) text embeddings from DistilBERTEncoder
        Returns eps_pred with same shape as x_t.
        """
        # time embedding -> same space as txt
        t_emb = self.time_embed(t)      # (B, time_dim)
        t_emb = self.time_mlp(t_emb)    # (B, emb_dim)

        cond = txt_emb + t_emb          # combine text & time

        # encoder
        x1 = self.c1(x_t)
        x1 = self.film(x1, self.gamma1(cond), self.beta1(cond))
        x2 = self.c2(self.p1(x1))
        x2 = self.film(x2, self.gamma2(cond), self.beta2(cond))

        # bottleneck
        xb = self.cb(self.p2(x2))
        xb = self.film(xb, self.gammaB(cond), self.betaB(cond))

        # decoder
        y = self.u2(xb)
        y = self._align(y, x2)
        y = torch.cat([y, x2], dim=1)
        y = self.c3(y)
        y = self.u1(y)
        y = self._align(y, x1)
        y = torch.cat([y, x1], dim=1)
        y = self.c4(y)

        # Here we predict eps (noise), not clean mel directly
        return self.out(y)
