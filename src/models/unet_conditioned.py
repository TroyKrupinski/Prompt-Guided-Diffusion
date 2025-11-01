import torch
import torch.nn as nn

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

class UNetDenoiser(nn.Module):
    # Small U-Net that takes noisy mel (B,1,M,T) and a text embedding (B,E)
    # and predicts a clean mel. Conditioning is via FiLM-like affine on features.
    def __init__(self, emb_dim=512, base=64):
        super().__init__()
        self.emb_dim = emb_dim

        # Encoder
        self.c1 = ConvBlock(1, base)
        self.p1 = nn.MaxPool2d(2)
        self.c2 = ConvBlock(base, base*2)
        self.p2 = nn.MaxPool2d(2)

        # Bottleneck
        self.cb = ConvBlock(base*2, base*4)

        # Decoder
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.c3 = ConvBlock(base*4, base*2)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.c4 = ConvBlock(base*2, base)

        self.out = nn.Conv2d(base, 1, 1)

        # FiLM conditioning layers
        self.gamma1 = nn.Linear(emb_dim, base)
        self.beta1  = nn.Linear(emb_dim, base)
        self.gamma2 = nn.Linear(emb_dim, base*2)
        self.beta2  = nn.Linear(emb_dim, base*2)
        self.gammaB = nn.Linear(emb_dim, base*4)
        self.betaB  = nn.Linear(emb_dim, base*4)

    def film(self, x, gamma, beta):
        # x: (B,C,H,W), gamma/beta: (B,C)
        g = gamma.unsqueeze(-1).unsqueeze(-1)
        b = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + g) + b

    def forward(self, x, txt_emb):
        # x: (B,1,M,T), txt_emb: (B,E)
        # Encoder
        x1 = self.c1(x)
        x1 = self.film(x1, self.gamma1(txt_emb), self.beta1(txt_emb))
        x2 = self.c2(self.p1(x1))
        x2 = self.film(x2, self.gamma2(txt_emb), self.beta2(txt_emb))

        # Bottleneck
        xb = self.cb(self.p2(x2))
        xb = self.film(xb, self.gammaB(txt_emb), self.betaB(txt_emb))

        # Decoder
        y = self.u2(xb)
        y = torch.cat([y, x2], dim=1)
        y = self.c3(y)
        y = self.u1(y)
        y = torch.cat([y, x1], dim=1)
        y = self.c4(y)
        return torch.sigmoid(self.out(y))
