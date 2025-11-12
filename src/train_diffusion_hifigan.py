import argparse, os, numpy as np, torch, torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from utils_logging import append_csv, plot_curve_from_csv

from models.unet_diffusion import DiffusionUNet
from models.text_encoder import DistilBERTEncoder
from diffusion.scheduler import DiffusionScheduler

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------------------
# Dataset: 80-bin HiFi-GAN mels
# ---------------------------
class MelDatasetHifiGAN(Dataset):
    def __init__(self, mels_dir, segment_frames=860):
        self.paths = list(Path(mels_dir).glob("**/*.npy"))
        self.segment_frames = segment_frames

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        mel = np.load(self.paths[idx])  # (80, T) log-mel
        M, T = mel.shape
        if T < self.segment_frames:
            rep = int(np.ceil(self.segment_frames / T))
            mel = np.tile(mel, (1, rep))[:, :self.segment_frames]
        else:
            start = np.random.randint(0, T - self.segment_frames + 1)
            mel = mel[:, start:start + self.segment_frames]
        x0 = torch.from_numpy(mel).float().unsqueeze(0)  # (1, 80, T)
        # normalize to [0,1] roughly if needed (assume already ~log mel [-12,4]):
        x0 = (x0 - (-12.0)) / (4.0 - (-12.0))
        x0 = x0.clamp(0.0, 1.0)
        return x0  # (1,80,T)

# ---------------------------
# Prompt → mel transform utils (80-mel)
# ---------------------------
def _time_resample_t(mel_bt, scale):
    """mel_bt: (B,M,T). scale>1 compress (faster), <1 stretch (slower). Keep T via crop/pad."""
    B, M, T = mel_bt.shape
    new_T = int(np.round(T/scale))
    x = torch.linspace(0, T-1, steps=new_T, device=mel_bt.device)
    x0 = x.long()
    x1 = (x0 + 1).clamp(max=T-1)
    w = x - x0
    mel0 = mel_bt.index_select(2, x0)
    mel1 = mel_bt.index_select(2, x1)
    mel_new = (1 - w) * mel0 + w * mel1
    if new_T >= T:
        st = (new_T - T) // 2
        mel_new = mel_new[:, :, st:st+T]
    else:
        pad_left = (T - new_T)//2
        pad_right = T - new_T - pad_left
        mel_new = torch.nn.functional.pad(mel_new, (pad_left, pad_right))
    return mel_new

def _ema_smooth_time(mel_bt, alpha=0.25):
    """Simple EMA along time: adds 'reverb-like' smear."""
    out = mel_bt.clone()
    for t in range(1, mel_bt.shape[2]):
        out[:, :, t] = alpha*out[:, :, t] + (1-alpha)*out[:, :, t-1]
    return out

def _distort(mel_bt, strength=0.6):
    """Mild nonlinearity in [0,1] domain."""
    x = mel_bt * 2.0 - 1.0
    y = torch.tanh(x * (1.0 + 2.0*strength))
    return ((y + 1.0) * 0.5).clamp(0.0, 1.0)

def _eq_bass(mel_bt, boost_db=6.0, n_low_bins=16):
    g = 10.0 ** (boost_db / 20.0)
    idx = min(n_low_bins, mel_bt.shape[1])
    mask = torch.ones_like(mel_bt)
    mask[:, :idx, :] = g
    return (mel_bt * mask).clamp(0.0, 1.0)

def _gain(mel_bt, gain_db=6.0):
    g = 10.0 ** (gain_db / 20.0)
    return (mel_bt * g).clamp(0.0, 1.0)

def apply_prompt_transform_80(mel_bmt, prompt):
    """
    mel_bmt: (B,1,80,T) in [0,1]. Returns target mel after a prompt-specific transform.
    Descriptive prompts -> identity. Control prompts -> consistent transform.
    """
    B, _, M, T = mel_bmt.shape
    mel_bt = mel_bmt.squeeze(1)  # (B,80,T)
    p = prompt.lower().strip()

    if "increase bpm" in p or "up tempo" in p or "faster" in p:
        tgt = _time_resample_t(mel_bt, scale=1.15)
    elif "decrease bpm" in p or "slow down" in p or "slower" in p:
        tgt = _time_resample_t(mel_bt, scale=0.85)
    elif "add reverb" in p:
        tgt = _ema_smooth_time(mel_bt, alpha=0.25)
    elif "remove reverb" in p:
        sm = _ema_smooth_time(mel_bt, alpha=0.25)
        tgt = (mel_bt - 0.15*sm).clamp(0.0, 1.0)
    elif "add distortion" in p:
        tgt = _distort(mel_bt, strength=0.6)
    elif "remove distortion" in p:
        tgt = torch.pow(mel_bt, 1.2).clamp(0.0, 1.0)
    elif "more bass" in p:
        tgt = _eq_bass(mel_bt, boost_db=6.0, n_low_bins=16)
    elif "less bass" in p:
        tgt = _eq_bass(mel_bt, boost_db=-6.0, n_low_bins=16)
    elif "increase volume" in p:
        tgt = _gain(mel_bt, gain_db=6.0)
    elif "decrease volume" in p or "lower volume" in p:
        tgt = _gain(mel_bt, gain_db=-6.0)
    elif "add percussion" in p:
        hi = slice(int(M*0.7), M)
        mask = torch.ones_like(mel_bt); mask[:, hi, :] = 10.0 ** (6.0/20.0)
        tgt = (mel_bt * mask).clamp(0.0, 1.0)
    elif "mute percussion" in p:
        hi = slice(int(M*0.7), M)
        mask = torch.ones_like(mel_bt); mask[:, hi, :] = 10.0 ** (-9.0/20.0)
        tgt = (mel_bt * mask).clamp(0.0, 1.0)
    elif "add vocals" in p:
        mid_lo, mid_hi = int(M*0.25), int(M*0.6)
        mask = torch.ones_like(mel_bt); mask[:, mid_lo:mid_hi, :] = 10.0 ** (4.0/20.0)
        tgt = (mel_bt * mask).clamp(0.0, 1.0)
    elif "remove vocals" in p:
        mid_lo, mid_hi = int(M*0.25), int(M*0.6)
        mask = torch.ones_like(mel_bt); mask[:, mid_lo:mid_hi, :] = 10.0 ** (-6.0/20.0)
        tgt = (mel_bt * mask).clamp(0.0, 1.0)
    else:
        tgt = mel_bt  # identity for descriptive prompts

    return tgt.unsqueeze(1)  # (B,1,80,T)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mels_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out_dir", type=str, default="runs/diffusion_fma_hifigan")
    ap.add_argument("--prompt", type=str, default="energetic rock")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--timesteps", type=int, default=1000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = MelDatasetHifiGAN(args.mels_dir)
    if len(ds) == 0:
        raise RuntimeError(f"No .npy mel files found in {args.mels_dir}")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = DiffusionUNet().to(device)
    txt   = DistilBERTEncoder().to(device)
    sched = DiffusionScheduler(timesteps=args.timesteps, device=device)

    opt = torch.optim.AdamW(list(model.parameters()) + list(txt.proj.parameters()), lr=args.lr)
    loss_fn = nn.MSELoss()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    PROMPT_POOL = [
        # Styles / Genres
        "energetic rock with strong drums",
        "soft piano with mellow tempo",
        "jazz with saxophone solo",
        "ambient electronic pads",
        "lofi chill beat",
        "classical orchestral strings",
        "acoustic folk guitar",
        "upbeat pop melody",
        # Control prompts
        "increase BPM", "decrease BPM", "up tempo", "slow down",
        "add reverb", "remove reverb",
        "add distortion", "remove distortion",
        "more bass", "less bass",
        "increase volume", "decrease volume",
        "add percussion", "mute percussion",
        "add vocals", "remove vocals"
    ]

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x0 in tqdm(dl, desc=f"Diff-HiFi Epoch {epoch}"):
            x0 = x0.to(device)                            # (B,1,80,T) in [0,1]
            B = x0.size(0)

            # Random prompt per batch + text emb
            random_prompt = np.random.choice(PROMPT_POOL)
            txt_emb = txt([random_prompt], device=device).repeat(B, 1)

            # Define *target* mel from prompt transform
            x0_tgt = apply_prompt_transform_80(x0, random_prompt)  # (B,1,80,T)

            # Sample timesteps and diffuse the *target* (teach model to denoise to target)
            t = sched.sample_timesteps(B)
            x_t, noise = sched.q_sample(x0_tgt, t)

            # Predict noise
            eps_pred = model(x_t, t, txt_emb)
            loss = loss_fn(eps_pred, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * B

        avg = total / len(ds)
        print(f"Epoch {epoch} | eps MSE: {avg:.6f}")
        # 1) numeric-only file for plotting
        append_csv(out_dir / "eps_mse.csv",
                {"epoch": epoch, "eps_mse": avg},
                header_order=["epoch","eps_mse"])

        # 2) prompts in their own file (no plotting)
        append_csv(out_dir / "prompts.csv",
                {"epoch": epoch, "prompt": random_prompt},
                header_order=["epoch","prompt"])


        torch.save(
            {"model": model.state_dict(), "txt": txt.state_dict(), "epoch": epoch},
            str(out_dir / f"ckpt_diff_hifigan_{epoch}.pt"),
        )

    if args.plot:
        plot_curve_from_csv(out_dir / "eps_mse.csv", "epoch", "eps_mse",
                            out_dir / "training_curve.png",
                            title="Diffusion(HiFiGAN) eps-MSE vs Epoch")

if __name__ == "__main__":
    main()
