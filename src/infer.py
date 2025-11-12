import argparse, os, numpy as np, torch, soundfile as sf, librosa
import torch.nn.functional as F
from pathlib import Path
from scipy.ndimage import median_filter  # pip install scipy if missing

from models.unet_conditioned import UNetDenoiser
from models.text_encoder import DistilBERTEncoder
from audio.utils import mel_from_wav
from utils_logging import ensure_dir, save_mel_png_from_wav

# Keep these identical to preprocessing (baseline pipeline)
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP = 512
WIN = 2048
FMIN = 20
FMAX = SR // 2  # 11025

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pad_to_multiple_frames(mel, mult=4):
    M, T = mel.shape
    padT = (mult - (T % mult)) % mult
    if padT:
        mel = np.pad(mel, ((0,0),(0,padT)), mode="edge")
    return mel, T

def rms(x, eps=1e-8):
    return float(np.sqrt(np.mean(np.square(x)) + eps))

def cosine_xfade(a_tail, b_head):
    n = len(a_tail)
    t = np.linspace(0, np.pi, n, endpoint=False)
    wA = 0.5*(1+np.cos(t))
    wB = 1 - wA
    return a_tail*wA + b_head*wB

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", type=str, default="results/vis_infer",
                    help="Folder to store spectrogram PNGs")
    ap.add_argument("--wav", type=str, required=True,
                    help="Source WAV to continue")
    ap.add_argument("--prompt", type=str, required=True,
                    help="Text prompt for conditioning")
    ap.add_argument("--out", type=str, default="output.wav",
                    help="Output WAV path")
    ap.add_argument("--ckpt", type=str, default="",
                    help="Path to checkpoint .pt (if empty, auto-picks latest)")
    ap.add_argument("--xfade_seconds", type=float, default=1.5,
                    help="Length of crossfade at the seam")
    ap.add_argument("--warmup_cut", type=float, default=0.5,
                    help="Seconds trimmed from Griffin-Lim warmup")
    ap.add_argument("--gl_iter", type=int, default=96,
                    help="Griffin-Lim iterations")
    args = ap.parse_args()

    print("[infer] CWD:", os.getcwd())
    print("[infer] WAV:", args.wav)

    # 1) Load source audio
    y_src, _sr = librosa.load(args.wav, sr=SR, mono=True)
    print(f"[infer] Source dur: {len(y_src)/SR:.2f}s")

    # 2) Compute source mel (normalized) + true dB bounds
    mel_norm = mel_from_wav(
        y_src, sr=SR, n_mels=N_MELS, hop_length=HOP, n_fft=N_FFT, fmin=FMIN, fmax=FMAX
    )  # (M,T) in [0,1] or normalized as per audio.utils
    # We’ll estimate dB range from power mel of source for realistic mapping back
    S_pow_src = librosa.feature.melspectrogram(
        y=y_src, sr=SR, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX
    )
    S_db_src = librosa.power_to_db(S_pow_src, ref=np.max)
    db_min, db_max = float(S_db_src.min()), float(S_db_src.max())
    print(f"[infer] Source mel dB range: [{db_min:.1f}, {db_max:.1f}]")

    mel_pad, orig_T = pad_to_multiple_frames(mel_norm, mult=4)
    mel_t = torch.from_numpy(mel_pad).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,M,T)

    # 3) Load model/checkpoint
    ckpts = []
    if args.ckpt:
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

    # 4) Predict denoised mel with text conditioning
    with torch.no_grad():
        txt_emb = txt([args.prompt], device=device)  # (1,512)
        pred = model(mel_t, txt_emb)                # (1,1,M,T)

    # Crop back to original T and remove batch/channel
    pred = pred[..., :N_MELS, :orig_T].squeeze().cpu().numpy()  # (M,T)

    # Optional light temporal smoothing to reduce GL artifacts
    pred = median_filter(pred, size=(1, 3))

    # 5) Map normalized mel → dB and invert to waveform
    S_db = pred * (db_max - db_min) + db_min
    S_pow = librosa.db_to_power(S_db)
    y_gen = librosa.feature.inverse.mel_to_audio(
        S_pow, sr=SR, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
        fmin=FMIN, fmax=FMAX, n_iter=args.gl_iter
    )

    # Trim GL warmup if requested
    warm = int(max(args.warmup_cut, 0.0) * SR)
    if warm > 0 and len(y_gen) > warm:
        y_gen = y_gen[warm:]

    # 6) Crossfade seam and concatenate
    xfade_n = int(max(args.xfade_seconds, 0.0) * SR)
    xfade_n = min(xfade_n, len(y_src), len(y_gen))

    if xfade_n > 0:
        # RMS match on seam head
        src_tail = y_src[-xfade_n:]
        gen_head = y_gen[:xfade_n]
        g_src = rms(src_tail)
        g_gen = rms(gen_head)
        gain = g_src / max(g_gen, 1e-6)
        y_gen = y_gen * gain

        seam = cosine_xfade(y_src[-xfade_n:], y_gen[:xfade_n])
        y_out = np.concatenate([y_src[:-xfade_n], seam, y_gen[xfade_n:]], axis=0)
    else:
        y_out = np.concatenate([y_src, y_gen], axis=0)

    # Simple peak norm
    peak = np.max(np.abs(y_out)) + 1e-9
    if peak > 1.0:
        y_out = y_out / peak

    # 7) Write audio + visuals
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), y_out, SR)
    print(f"[infer] WROTE {out_path} | dur={len(y_out)/SR:.2f}s")

    vis_dir = Path(args.log_dir)
    ensure_dir(vis_dir)
    save_mel_png_from_wav(args.wav, str(vis_dir / "orig.png"), n_mels=128)
    save_mel_png_from_wav(str(out_path), str(vis_dir / "baseline_infer.png"), n_mels=128)
    print(f"[infer] Wrote mel PNGs to: {vis_dir}")

if __name__ == "__main__":
    main()
