import argparse, os, numpy as np, torch, torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from utils_logging import append_csv, plot_curve_from_csv
from models.unet_conditioned import UNetDenoiser
from models.text_encoder import DistilBERTEncoder

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------------------
# Mel dataset (unchanged IO)
# ---------------------------
class MelDataset(Dataset):
    def __init__(self, mels_dir, segment_frames=860):
        self.paths = list(Path(mels_dir).glob("**/*.npy"))
        self.segment_frames = segment_frames

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        mel = np.load(self.paths[idx])  # (M, T) in [0,1]
        M, T = mel.shape
        if T < self.segment_frames:
            rep = int(np.ceil(self.segment_frames / T))
            mel = np.tile(mel, (1, rep))[:, :self.segment_frames]
        else:
            start = np.random.randint(0, T - self.segment_frames + 1)
            mel = mel[:, start:start+self.segment_frames]
        mel = torch.from_numpy(mel).float()                # (M,T) in [0,1]
        return mel  # we'll build (src -> tgt) in the training loop


# ---------------------------
# Prompt → mel transform utils
# ---------------------------
def _time_resample_t(mel_bt, scale):
    """
    mel_bt: (B, M, T) float
    scale > 1.0  => compress time (up tempo / increase BPM)
    scale < 1.0  => stretch time  (down tempo / decrease BPM)
    returns (B, M, T) keeping same T via interpolation + center crop/pad
    """
    B, M, T = mel_bt.shape
    # build new time grid
    new_T = int(np.round(T/scale))
    x = torch.linspace(0, T-1, steps=new_T, device=mel_bt.device)
    x = x.clamp(0, T-1)
    # gather via linear interp along time
    # prepare indices
    x0 = x.long()
    x1 = (x0 + 1).clamp(max=T-1)
    w = x - x0
    # (B,M,new_T)
    mel0 = mel_bt.index_select(2, x0)
    mel1 = mel_bt.index_select(2, x1)
    mel_new = (1-w)*mel0 + w*mel1
    # now pad or crop back to T
    if new_T >= T:
        start = (new_T - T)//2
        mel_new = mel_new[:, :, start:start+T]
    else:
        pad = ( (T-new_T)//2, T - new_T - (T-new_T)//2 )
        mel_new = torch.nn.functional.pad(mel_new, (pad[0], pad[1]))
    return mel_new

def _ema_smooth_time(mel_bt, alpha=0.2):
    """Simple EMA smoothing along time (adds 'reverb-like' tail)."""
    B, M, T = mel_bt.shape
    out = mel_bt.clone()
    for t in range(1, T):
        out[:, :, t] = alpha*out[:, :, t] + (1-alpha)*out[:, :, t-1]
    return out

def _distort(mel_bt, strength=0.5):
    """
    Mild nonlinearity; assume input ~[0,1]. Move to [-1,1], apply tanh, map back.
    """
    x = (mel_bt * 2.0) - 1.0
    y = torch.tanh(x * (1.0 + strength*2.0))
    y = (y + 1.0) * 0.5
    return y.clamp(0.0, 1.0)

def _eq_bass(mel_bt, boost_db=6.0, n_low_bins=16):
    """Boost/cut low freq bins by simple gain on first n_low_bins."""
    B, M, T = mel_bt.shape
    g = 10.0 ** (boost_db/20.0)
    mask = torch.ones_like(mel_bt)
    idx = min(n_low_bins, M)
    mask[:, :idx, :] = g
    out = (mel_bt * mask).clamp(0.0, 1.0)
    return out

def _gain(mel_bt, gain_db=6.0):
    g = 10.0 ** (gain_db/20.0)
    return (mel_bt * g).clamp(0.0, 1.0)

def apply_prompt_transform(mel_bmt, prompt):
    """
    mel_bmt: (B,1,M,T) in [0,1]
    Returns target mel after applying a prompt-specific transform.
    Descriptive prompts -> identity.
    Control prompts -> consistent transform.
    """
    B, C, M, T = mel_bmt.shape
    mel_bt = mel_bmt.squeeze(1)  # (B,M,T)

    p = prompt.lower().strip()

    # --- tempo / BPM ---
    if "increase bpm" in p or "up tempo" in p or "faster" in p:
        tgt = _time_resample_t(mel_bt, scale=1.15)  # compress ~15%
    elif "decrease bpm" in p or "slow down" in p or "slower" in p:
        tgt = _time_resample_t(mel_bt, scale=0.85)  # stretch ~15%

    # --- reverb (temporal smoothing) ---
    elif "add reverb" in p:
        tgt = _ema_smooth_time(mel_bt, alpha=0.25)
    elif "remove reverb" in p:
        # simple high-pass along time: original - smoothed (noisy de-reverb approximation)
        sm = _ema_smooth_time(mel_bt, alpha=0.25)
        tgt = (mel_bt - 0.15*sm).clamp(0.0, 1.0)

    # --- distortion ---
    elif "add distortion" in p:
        tgt = _distort(mel_bt, strength=0.6)
    elif "remove distortion" in p:
        # very soft expansion toward linear
        tgt = torch.pow(mel_bt, 1.2).clamp(0.0, 1.0)

    # --- bass EQ ---
    elif "more bass" in p:
        tgt = _eq_bass(mel_bt, boost_db=6.0, n_low_bins=16)
    elif "less bass" in p:
        tgt = _eq_bass(mel_bt, boost_db=-6.0, n_low_bins=16)

    # --- volume ---
    elif "increase volume" in p:
        tgt = _gain(mel_bt, gain_db=6.0)
    elif "decrease volume" in p or "lower volume" in p:
        tgt = _gain(mel_bt, gain_db=-6.0)

    # --- percussion/vocals placeholders (approximate via HF/MF EQ tweaks) ---
    elif "add percussion" in p:
        # boost highs roughly
        B_, M_, T_ = mel_bt.shape
        mask = torch.ones_like(mel_bt)
        hi = slice(int(M_*0.7), M_)
        mask[:, hi, :] = 10.0 ** (6.0/20.0)
        tgt = (mel_bt * mask).clamp(0.0, 1.0)
    elif "mute percussion" in p:
        B_, M_, T_ = mel_bt.shape
        mask = torch.ones_like(mel_bt)
        hi = slice(int(M_*0.7), M_)
        mask[:, hi, :] = 10.0 ** (-9.0/20.0)
        tgt = (mel_bt * mask).clamp(0.0, 1.0)

    elif "add vocals" in p:
        # mild mid-band boost
        mid_lo, mid_hi = int(M*0.25), int(M*0.6)
        mask = torch.ones_like(mel_bt)
        mask[:, mid_lo:mid_hi, :] = 10.0 ** (4.0/20.0)
        tgt = (mel_bt * mask).clamp(0.0, 1.0)
    elif "remove vocals" in p:
        mid_lo, mid_hi = int(M*0.25), int(M*0.6)
        mask = torch.ones_like(mel_bt)
        mask[:, mid_lo:mid_hi, :] = 10.0 ** (-6.0/20.0)
        tgt = (mel_bt * mask).clamp(0.0, 1.0)

    else:
        # descriptive genres / styles -> identity
        tgt = mel_bt

    return tgt.unsqueeze(1)  # (B,1,M,T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="Save training curve PNG at end")
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

    for epoch in range(1, args.epochs+1):
        model.train()
        total = 0.0
        for mel in tqdm(dl, desc=f"Epoch {epoch}"):
            mel = mel.to(device)               # (B,M,T) in [0,1]
            B, M, T = mel.shape

            # Build source (slightly noised) and text embedding
            src = mel.unsqueeze(1)                                  # (B,1,M,T)
            src_noisy = (src + 0.05 * torch.randn_like(src)).clamp(0,1)

            # Random prompt per batch
            random_prompt = np.random.choice(PROMPT_POOL)
            txt_emb = txt([random_prompt], device=device).repeat(B, 1)

            # Target is prompt-transformed mel
            tgt = apply_prompt_transform(src, random_prompt)        # (B,1,M,T)

            # Predict transform
            pred = model(src_noisy, txt_emb)                        # (B,1,M,T)
            loss = loss_fn(pred, tgt)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * B

        avg = total / len(ds)
#        keep loss.csv numeric-only
        append_csv(Path(args.out_dir) / "loss.csv",
                {"epoch": epoch, "L1": avg},
                header_order=["epoch","L1"])

        # log prompt separately to avoid commas breaking CSV parsing
        append_csv(Path(args.out_dir) / "prompts.csv",
                {"epoch": epoch, "prompt": random_prompt},
                header_order=["epoch","prompt"])


        torch.save(
            {"model": model.state_dict(), "txt": txt.state_dict(), "epoch": epoch},
            str(Path(args.out_dir) / f"ckpt_{epoch}.pt")
        )

    if args.plot:
        plot_curve_from_csv(Path(args.out_dir) / "loss.csv", "epoch", "L1",
                            Path(args.out_dir) / "training_curve.png",
                            title="Baseline L1 vs Epoch")

if __name__ == "__main__":
    main()
