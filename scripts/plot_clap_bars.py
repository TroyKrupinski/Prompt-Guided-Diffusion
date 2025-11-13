import re
from pathlib import Path
import matplotlib.pyplot as plt

# Project root = parent of "scripts" directory
ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT / "results"

def parse_clap_txt(path: Path):
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return None

    # Read entire file and search for any floats
    text = path.read_text(errors="ignore")
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)

    if not matches:
        print(f"[WARN] Could not parse score in: {path}")
        return None

    # Take the *last* numeric value in the file
    score = float(matches[-1])
    return score

def main():
    print(f"[INFO] Script path: {Path(__file__).resolve()}")
    print(f"[INFO] ROOT:      {ROOT}")
    print(f"[INFO] BASE_DIR:  {BASE_DIR}")

    prompts = ["slowdown", "increaseBPM", "reverb", "bass", "distortion"]

    # Adjust these patterns if your filenames differ
    baseline_files = {
        p: BASE_DIR / f"clap_baseline_{p}_000002_cond.txt"
        for p in prompts
    }
    diffusion_files = {
        p: BASE_DIR / f"clap_diffusion_{p}_000002.txt"
        for p in prompts
    }

    baseline_scores = []
    diffusion_scores = []

    print("\nPrompt\tBaseline\tDiffusion")
    for p in prompts:
        b_path = baseline_files[p]
        d_path = diffusion_files[p]

        bscore = parse_clap_txt(b_path)
        dscore = parse_clap_txt(d_path)

        if bscore is None:
            bscore = 0.0
        if dscore is None:
            dscore = 0.0

        baseline_scores.append(bscore)
        diffusion_scores.append(dscore)
        print(f"{p}\t{bscore:.4f}\t{dscore:.4f}")

    # Bar plot
    x = range(len(prompts))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar([i - width/2 for i in x], baseline_scores, width, label="Baseline UNet")
    plt.bar([i + width/2 for i in x], diffusion_scores, width, label="Diffusion + HiFi-GAN")

    plt.xticks(list(x), prompts, rotation=30)
    plt.ylabel("CLAP cosine similarity")
    plt.title("Prompt–Audio Alignment (CLAP) per Prompt and Model")
    plt.legend()
    plt.tight_layout()

    out_path = ROOT / "clap_alignment_barplot.png"
    plt.savefig(out_path, dpi=200)
    print(f"\n[OK] Saved bar chart to {out_path.resolve()}")

if __name__ == "__main__":
    main()
