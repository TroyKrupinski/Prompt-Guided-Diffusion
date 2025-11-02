import argparse, os, numpy as np, torch, soundfile as sf, librosa
import torch.nn.functional as F
from pathlib import Path
from scipy.ndimage import median_filter  # pip install scipy if missing
from models.unet_conditioned import UNetDenoiser
from models.text_encoder import DistilBERTEncoder
from audio.utils import mel_from_wav

# Keep these identical to preprocessing
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP = 512
WIN = 2048
FMIN = 20
FMAX = SR // 2  # 11025

def rms(x, eps=1e-8):
    return float(np.sqrt(np.mean(np.square(x)) + eps))

def cosine_xfade(a_tail, b_head):
    n = len(a_tail)
    t = np.linspace(0, np.pi, n, endpoint=False)
    wA = 0.5*(1+np.cos(t))
    wB = 1 - wA
    return a_tail*wA + b_head*wB

def pad_to_multiple_frames(mel, mult=4):
    M, T = mel.shape
    padT = (mult - (T % mult)) % mult
    if padT:
        mel = np.pad(mel, ((0,0),(0,padT)), mode="edge")
    return mel, T


# ... top of file (imports unchanged) ...
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=str, required=True)
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--out", type=str, default="output.wav")
    ap.add_argument("--ckpt", type=str, default="", help="Path to a specific checkpoint .pt")  # <-- NEW
    ap.add_argument("--xfade_seconds", type=float, default=1.5)
    ap.add_argument("--warmup_cut", type=float, default=0.5)
    ap.add_argument("--gl_iter", type=int, default=96)
    args = ap.parse_args()
    # ...
    # 4) Load model
    ckpts = []
    if args.ckpt:
        from pathlib import Path
        ck = Path(args.ckpt)
        if ck.is_file():
            ckpts = [ck]
    if not ckpts:
        ckpts = sorted(Path("runs/exp1").glob("ckpt_*.pt")) + sorted(Path("runs/fma_exp").glob("ckpt_*.pt"))
    model = UNetDenoiser().to(device).eval()
    txt   = DistilBERTEncoder().to(device).eval()
    if ckpts:
        state = torch.load(str(ckpts[-1]), map_location=device)
        model.load_state_dict(state["model"])
        txt.load_state_dict(state["txt"])
        print(f"[infer] Loaded checkpoint: {ckpts[-1]}")
    else:
        print("[infer] No checkpoints found; using random weights (expect very rough audio).")
