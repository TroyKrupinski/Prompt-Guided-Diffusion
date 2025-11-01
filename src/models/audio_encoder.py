import torch
import torch.nn as nn

class SmallAudioEncoder(nn.Module):
    # A small CNN over mel-spectrograms producing a 512-D embedding.
    # Input: (B, 1, n_mels, T)
    def __init__(self, in_ch=1, hidden=64, emb=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(hidden, hidden*2, 3, padding=1), nn.BatchNorm2d(hidden*2), nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(hidden*2, hidden*4, 3, padding=1), nn.BatchNorm2d(hidden*4), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.proj = nn.Linear(hidden*4, emb)

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.proj(h)
