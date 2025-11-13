import argparse, json, librosa, librosa.display, matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pathlib import Path

def plot_spectrograms(before_path, after_path, title, out_path):
    y1, sr1 = librosa.load(before_path, sr=22050)
    y2, sr2 = librosa.load(after_path, sr=22050)
    S1 = librosa.power_to_db(np.abs(librosa.feature.melspectrogram(y=y1, sr=sr1, n_mels=128)), ref=np.max)
    S2 = librosa.power_to_db(np.abs(librosa.feature.melspectrogram(y=y2, sr=sr2, n_mels=128)), ref=np.max)
    fig, axs = plt.subplots(2, 1, figsize=(8,6))
    librosa.display.specshow(S1, sr=sr1, x_axis='time', y_axis='mel', ax=axs[0])
    axs[0].set_title("Original Input")
    librosa.display.specshow(S2, sr=sr2, x_axis='time', y_axis='mel', ax=axs[1])
    axs[1].set_title(title)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_clap_bars(result_dir, out_path):
    clap_files = list(Path(result_dir).glob("clap_*.txt"))
    names, values = [], []
    for f in clap_files:
        try:
            with open(f) as fh:
                for line in fh:
                    if "CLAP cosine similarity" in line:
                        val = float(line.strip().split(":")[-1])
                        names.append(f.stem.replace("clap_", ""))
                        values.append(val)
        except: pass
    if len(values) == 0: return
    df = pd.DataFrame({"Prompt": names, "CLAP Similarity": values})
    df.sort_values("CLAP Similarity", inplace=True)
    plt.figure(figsize=(8,4))
    plt.barh(df["Prompt"], df["CLAP Similarity"], color="skyblue")
    plt.xlabel("Cosine Similarity")
    plt.title("CLAP Alignment Across Prompts")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # CLAP bar chart
    plot_clap_bars(args.results_dir, out / "clap_alignment.png")

    # Spectrogram comparisons
    examples = [
        ("data/fma/wavs_medium/000/000002.wav", "results/baseline_slowdown_000002_cond.wav", "Baseline: Slow Down"),
        ("data/fma/wavs_medium/000/000002.wav", "results/baseline_increaseBPM_000002_cond.wav", "Baseline: Increase BPM"),
        ("data/fma/wavs_medium/000/000002.wav", "results/diffusion_slowdown_000002.wav", "Diffusion: Slow Down"),
        ("data/fma/wavs_medium/000/000002.wav", "results/diffusion_increaseBPM_000002.wav", "Diffusion: Increase BPM")
    ]
    for before, after, title in examples:
        if Path(after).exists():
            out_path = out / f"{title.replace(': ','_').replace(' ','_')}.png"
            plot_spectrograms(before, after, title, out_path)

if __name__ == "__main__":
    main()
