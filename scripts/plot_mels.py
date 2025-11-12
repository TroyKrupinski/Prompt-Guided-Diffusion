import argparse, numpy as np, librosa, librosa.display, matplotlib.pyplot as plt, soundfile as sf, torch
from pathlib import Path

def load_mel_from_wav(wav_path, sr=22050, n_mels=128):
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=n_mels, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

def load_hifigan_mel_npy(npy_path):  # (80, T) log-mel already
    mel = np.load(npy_path)
    return mel

def show(ax, mel, title):
    img = librosa.display.specshow(mel, x_axis='time', y_axis='mel', sr=22050, hop_length=256, ax=ax)
    ax.set_title(title)
    plt.colorbar(img, ax=ax, format="%+2.0f dB")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_wav", required=True)
    ap.add_argument("--baseline_wav", required=False, help="Griffin–Lim output")
    ap.add_argument("--diffusion_wav", required=False, help="HiFi-GAN output")
    ap.add_argument("--out", default="results/mel_comparison.png")
    ap.add_argument("--hifigan", action="store_true", help="Use 80-mel for all plots (rough comparison)")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_mels = 80 if args.hifigan else 128

    fig, axs = plt.subplots(1, 3 if (args.baseline_wav and args.diffusion_wav) else 1, figsize=(15,4))

    if not isinstance(axs, np.ndarray): axs = np.array([axs])

    m0 = load_mel_from_wav(args.orig_wav, n_mels=n_mels)
    show(axs[0], m0, "Original (mel)")

    i = 1
    if args.baseline_wav:
        m1 = load_mel_from_wav(args.baseline_wav, n_mels=n_mels)
        show(axs[i], m1, "Baseline (UNet + Griffin–Lim)")
        i += 1
    if args.diffusion_wav:
        m2 = load_mel_from_wav(args.diffusion_wav, n_mels=n_mels)
        show(axs[i], m2, "Diffusion + HiFi-GAN")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print("[plot_mels] Wrote", args.out)

if __name__ == "__main__":
    main()
