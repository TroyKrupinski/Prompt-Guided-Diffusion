import argparse, os, numpy as np, torch, torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from models.unet_diffusion import DiffusionUNet
from models.text_encoder import DistilBERTEncoder
from diffusion.scheduler import DiffusionScheduler

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class MelDatasetHifiGAN(Dataset):
    def __init__(self, mels_dir, segment_frames=860):
        self.paths = list(Path(mels_dir).glob("**/*.npy"))
        self.segment_frames = segment_frames

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        mel = np.load(self.paths[idx])  # (num_mels=80, T), log-mel from HiFi-GAN
        M, T = mel.shape
        if T < self.segment_frames:
            rep = int(np.ceil(self.segment_frames / T))
            mel = np.tile(mel, (1, rep))[:, :self.segment_frames]
        else:
            start = np.random.randint(0, T - self.segment_frames + 1)
            mel = mel[:, start:start + self.segment_frames]
        x0 = torch.from_numpy(mel).unsqueeze(0)  # (1, M, T)
        return x0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mels_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out_dir", type=str, default="runs/diffusion_fma_hifigan")
    ap.add_argument("--prompt", type=str, default="energetic rock")
    ap.add_argument("--timesteps", type=int, default=1000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = MelDatasetHifiGAN(args.mels_dir)
    if len(ds) == 0:
        raise RuntimeError(f"No .npy mel files found in {args.mels_dir}")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 80-mel diffusion UNet (same class, just trained on 80 bands)
    model = DiffusionUNet().to(device)
    txt   = DistilBERTEncoder().to(device)
    sched = DiffusionScheduler(timesteps=args.timesteps, device=device)

    opt = torch.optim.AdamW(list(model.parameters()) + list(txt.proj.parameters()), lr=args.lr)
    loss_fn = nn.MSELoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x0 in tqdm(dl, desc=f"Diff-HiFi Epoch {epoch}"):
            x0 = x0.to(device)  # (B,1,M,T)
            B = x0.size(0)

            # sample timesteps
            t = sched.sample_timesteps(B)

            # forward diffusion: q(x_t | x0)
            x_t, noise = sched.q_sample(x0, t)

            # text conditioning
            txt_emb = txt([args.prompt], device=device).repeat(B, 1)

            # predict eps
            eps_pred = model(x_t, t, txt_emb)

            loss = loss_fn(eps_pred, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * B

        avg = total / len(ds)
        print(f"Epoch {epoch} | eps MSE: {avg:.6f}")

        torch.save(
            {"model": model.state_dict(), "txt": txt.state_dict(), "epoch": epoch},
            str(out_dir / f"ckpt_diff_hifigan_{epoch}.pt"),
        )

if __name__ == "__main__":
    main()
