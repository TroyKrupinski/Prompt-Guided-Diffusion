
import os, csv, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import librosa, librosa.display

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def append_csv(csv_path, row_dict, header_order=None):
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)
    write_header = (not csv_path.exists())
    if header_order is None: header_order = list(row_dict.keys())
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header_order)
        if write_header: w.writeheader()
        w.writerow(row_dict)

def plot_curve_from_csv(csv_path, x_key, y_key, out_png, title="Training Curve"):
    import pandas as pd
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(6,4))
    plt.plot(df[x_key], df[y_key])
    plt.xlabel(x_key); plt.ylabel(y_key); plt.title(title)
    ensure_dir(Path(out_png).parent)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def save_mel_png_from_wav(wav_path, out_png, sr=22050, n_mels=128, n_fft=1024, hop=256, fmin=20, fmax=None, log_power=True):
    y, _sr = librosa.load(wav_path, sr=sr, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, fmin=fmin, fmax=fmax or sr//2, power=2.0)
    if log_power:
        S = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(8,3))
    librosa.display.specshow(S, x_axis='time', y_axis='mel', sr=sr, hop_length=hop)
    plt.colorbar(format="%+2.0f dB"); plt.title(Path(wav_path).name)
    ensure_dir(Path(out_png).parent)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def save_mel_png_from_npy(mel_npy_path, out_png, sr=22050, hop=256, is_log=True, title=None):
    M = np.load(mel_npy_path)  # (n_mels, T)
    plt.figure(figsize=(8,3))
    if is_log:
        librosa.display.specshow(M, x_axis='time', y_axis='mel', sr=sr, hop_length=hop)
        plt.colorbar(format="%+2.0f dB")
    else:
        plt.imshow(M, aspect='auto', origin='lower')
        plt.colorbar()
    plt.title(title or Path(mel_npy_path).name)
    ensure_dir(Path(out_png).parent)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
