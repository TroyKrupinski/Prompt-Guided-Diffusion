import argparse, os, numpy as np, librosa
from pathlib import Path

def to_mel(wav_path, sr=22050, n_mels=128, hop_length=512, n_fft=2048, fmin=20, fmax=None):
    y, _sr = librosa.load(wav_path, sr=sr, mono=True)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax
    )
    S = librosa.power_to_db(S, ref=np.max)
    # min-max normalize to [0,1]
    Smin, Smax = S.min(), S.max()
    Sm = (S - Smin) / (Smax - Smin + 1e-8)
    return Sm.astype("float32")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True, help="Folder with input .wav files")
    ap.add_argument("--out_dir", type=str, required=True, help="Where to write .npy mel files")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--n_mels", type=int, default=128)
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wavs = [p for p in in_dir.glob("**/*.wav")]
    if not wavs:
        print(f"No .wav files found in {in_dir}")
        return

    for p in wavs:
        mel = to_mel(str(p), sr=args.sr, n_mels=args.n_mels)
        rel = p.relative_to(in_dir).with_suffix(".npy")
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, mel)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
