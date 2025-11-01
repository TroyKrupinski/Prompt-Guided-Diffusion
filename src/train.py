import argparse, os, numpy as np, torch, torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from models.unet_conditioned import UNetDenoiser
from models.text_encoder import DistilBERTEncoder
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
class MelDataset(Dataset):
    def __init__(self, mels_dir, segment_frames=860):
        # segment_frames ~ 10s at hop=512 & sr=22.05k; adjust as needed
        self.paths = list(Path(mels_dir).glob("**/*.npy"))
        self.segment_frames = segment_frames

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        mel = np.load(self.paths[idx])  # (M, T) in [0,1]
        # ensure at least segment_frames by padding/tiling
        M, T = mel.shape
        if T < self.segment_frames:
            rep = int(np.ceil(self.segment_frames / T))
            mel = np.tile(mel, (1, rep))[:, :self.segment_frames]
        else:
            start = np.random.randint(0, T - self.segment_frames + 1)
            mel = mel[:, start:start+self.segment_frames]
        # create "noisy" input by adding Gaussian noise (bootstrap denoiser)
        clean = torch.from_numpy(mel).unsqueeze(0)  # (1,M,T)
        noisy = (clean + 0.1 * torch.randn_like(clean)).clamp(0,1)
        return noisy, clean

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mels_dir", type=str, default="data/mels")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out_dir", type=str, default="runs/exp1")
    ap.add_argument("--prompt", type=str, default="slower tempo with mellow piano")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = MelDataset(args.mels_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = UNetDenoiser().to(device)
    txt = DistilBERTEncoder().to(device)

    opt = torch.optim.AdamW(list(model.parameters()) + list(txt.proj.parameters()), lr=args.lr)
    loss_fn = nn.L1Loss()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        total = 0.0
        for noisy, clean in tqdm(dl, desc=f"Epoch {epoch}"):
            noisy = noisy.to(device)        # (B,1,M,T)
            clean = clean.to(device)
            txt_emb = txt([args.prompt], device=device).repeat(noisy.size(0), 1)
            pred = model(noisy, txt_emb)
            loss = loss_fn(pred, clean)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * noisy.size(0)

        avg = total / len(ds)
        print(f"Epoch {epoch} | L1: {avg:.4f}")
        torch.save(
            {"model": model.state_dict(), "txt": txt.state_dict(), "epoch": epoch},
            str(Path(args.out_dir) / f"ckpt_{epoch}.pt")
        )

if __name__ == "__main__":
    main()
