import argparse, os, numpy as np, torch, soundfile as sf, librosa
from pathlib import Path
from scipy.ndimage import median_filter
from utils_logging import ensure_dir, save_mel_png_from_wav

from models.unet_conditioned import UNetDenoiser
from models.text_encoder import DistilBERTEncoder
from audio.utils import mel_from_wav

SR = 22050
N_MELS = 128
N_FFT = 2048
HOP = 512
WIN = 2048
FMIN = 20
FMAX = SR // 2  # 11025

def pad_to_multiple(mel, mult=4):
    M, T = mel.shape
    padT = (mult - (T % mult)) % mult
    if padT:
        mel = np.pad(mel, ((0, 0), (0, padT)), mode="edge")
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
    ap.add_argument("--log_dir", type=str, default="results/vis", help="Folder for mel PNGs")
    ap.add_argument("--wav", type=str, required=True)
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_prefix", type=str, default="results/sample")
    ap.add_argument("--xfade_seconds", type=float, default=1.0)
    ap.add_argument("--warmup_cut", type=float, default=0.5)
    ap.add_argument("--gl_iter", type=int, default=96)
    args = ap.parse_args()

    print("[compare] CWD:", os.getcwd())
    print("[compare] WAV:", args.wav)
    print("[compare] CKPT:", args.ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[compare] Device:", device)

    # 1) Load source audio
    y, sr = librosa.load(args.wav, sr=SR, mono=True)
    print(f"[compare] Source dur: {len(y)/SR:.2f}s")

    # 2) Source mel (normalized) + true dB mel range
    mel_norm = mel_from_wav(y, sr=SR, n_mels=N_MELS, hop_length=HOP, n_fft=N_FFT, fmin=FMIN, fmax=FMAX)
    S_pow_src = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX
    )
    S_db_src = librosa.power_to_db(S_pow_src, ref=np.max)
    db_min, db_max = float(S_db_src.min()), float(S_db_src.max())
    print(f"[compare] Source mel dB range: [{db_min:.1f}, {db_max:.1f}]")

    mel_padded, orig_T = pad_to_multiple(mel_norm, mult=4)
    mel_t = torch.from_numpy(mel_padded).unsqueeze(0).unsqueeze(0).to(device)

    # 3) Load model + text encoder
    unet = UNetDenoiser().to(device).eval()
    txt  = DistilBERTEncoder().to(device).eval()
    state = torch.load(args.ckpt, map_location=device)
    unet.load_state_dict(state["model"])
    txt.load_state_dict(state["txt"])
    print("[compare] Loaded checkpoint:", args.ckpt)

    with torch.no_grad():
        # conditioned: real prompt
        txt_emb_cond = txt([args.prompt], device=device)
        # "unconditioned": neutral text prompt used for all clips
        txt_emb_uncond = txt(["music"], device=device)

        pred_uncond = unet(mel_t, txt_emb_uncond)
        pred_cond   = unet(mel_t, txt_emb_cond)

    pred_uncond = pred_uncond[..., :N_MELS, :orig_T].squeeze().cpu().numpy()
    pred_cond   = pred_cond[...,   :N_MELS, :orig_T].squeeze().cpu().numpy()

    # 4) Light temporal smoothing in mel space
    pred_uncond = median_filter(pred_uncond, size=(1, 3))
    pred_cond   = median_filter(pred_cond,   size=(1, 3))

    # 5) Map [0,1] → source dB range for both
    def mel_to_audio_from_db(mel_norm_clip):
        S_db = mel_norm_clip * (db_max - db_min) + db_min
        S_pow = librosa.db_to_power(S_db)
        y_hat = librosa.feature.inverse.mel_to_audio(
            S_pow, sr=SR, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
            fmin=FMIN, fmax=FMAX, n_iter=args.gl_iter
        )
        return y_hat

    print("[compare] Inverting unconditioned mel to audio...")
    y_uncond = mel_to_audio_from_db(pred_uncond)

    print("[compare] Inverting conditioned mel to audio...")
    y_cond = mel_to_audio_from_db(pred_cond)

    # 6) Trim Griffin-Lim warmup
    warm = int(max(args.warmup_cut, 0.0) * SR)
    if warm > 0:
        if len(y_uncond) > warm: y_uncond = y_uncond[warm:]
        if len(y_cond) > warm:   y_cond   = y_cond[warm:]

    # 7) RMS match and crossfade
    tail_n = int(max(args.xfade_seconds, 0.5) * SR)
    if len(y) < tail_n: tail_n = len(y)

    def join_with_crossfade(y_src, y_gen):
        if len(y_gen) < tail_n:
            y_gen = np.pad(y_gen, (0, tail_n - len(y_gen)), mode="constant")

        src_tail = y_src[-tail_n:]
        gen_head = y_gen[:tail_n]

        g_src = rms(src_tail)
        g_gen = rms(gen_head)
        gain = g_src / max(g_gen, 1e-6)
        y_gen *= gain

        xfade_n = int(args.xfade_seconds * SR)
        xfade_n = min(xfade_n, len(y_src), len(y_gen))
        if xfade_n > 0:
            seam = cosine_xfade(y_src[-xfade_n:], y_gen[:xfade_n])
            y_out = np.concatenate([y_src[:-xfade_n], seam, y_gen[xfade_n:]], axis=0)
        else:
            y_out = np.concatenate([y_src, y_gen], axis=0)

        peak = float(np.max(np.abs(y_out)) + 1e-9)
        if peak > 1.0:
            y_out = y_out / peak
        return y_out

    y_uncond_out = join_with_crossfade(y, y_uncond)
    y_cond_out   = join_with_crossfade(y, y_cond)

    # 8) Write files
    out_uncond = Path(args.out_prefix + "_uncond.wav").resolve()
    out_cond   = Path(args.out_prefix + "_cond.wav").resolve()
    out_uncond.parent.mkdir(parents=True, exist_ok=True)
    out_cond.parent.mkdir(parents=True, exist_ok=True)

    sf.write(str(out_uncond), y_uncond_out, SR)
    sf.write(str(out_cond),   y_cond_out,   SR)

    print(f"[compare] Wrote unconditioned: {out_uncond}")
    print(f"[compare] Wrote conditioned:   {out_cond}")
    vis_dir = Path(args.log_dir)
    ensure_dir(vis_dir)
    save_mel_png_from_wav(args.wav, vis_dir / "orig.png", n_mels=128)
    save_mel_png_from_wav(str(out_uncond), vis_dir / "baseline_uncond.png", n_mels=128)
    save_mel_png_from_wav(str(out_cond),   vis_dir / "baseline_cond.png",   n_mels=128)
    print(f"[compare] Wrote mel PNGs to: {vis_dir}")

if __name__ == "__main__":
    main()
