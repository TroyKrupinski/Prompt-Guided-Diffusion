import argparse
import os
from pathlib import Path
import sys
import json

import librosa
import numpy as np
import torch
import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HIFIGAN_ROOT = PROJECT_ROOT / "hifi-gan"

# Make hifi-gan visible for internal imports like in their code
sys.path.insert(0, str(HIFIGAN_ROOT))

# Load env.AttrDict
env_path = HIFIGAN_ROOT / "env.py"
env_spec = importlib.util.spec_from_file_location("hifigan_env", env_path)
hifigan_env = importlib.util.module_from_spec(env_spec)
env_spec.loader.exec_module(hifigan_env)
AttrDict = hifigan_env.AttrDict

# Load meldataset.mel_spectrogram
meld_path = HIFIGAN_ROOT / "meldataset.py"
meld_spec = importlib.util.spec_from_file_location("hifigan_meldataset", meld_path)
hifigan_meld = importlib.util.module_from_spec(meld_spec)
meld_spec.loader.exec_module(hifigan_meld)
mel_spectrogram = hifigan_meld.mel_spectrogram


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True, help="Folder with .wav files")
    ap.add_argument("--out_dir", type=str, required=True, help="Where to write .npy mels")
    ap.add_argument(
        "--config",
        type=str,
        required=True,
        help="HiFi-GAN config.json (for UNIVERSAL_V1)",
    )
    args = ap.parse_args()

    with open(args.config, "r") as f:
        h = AttrDict(json.load(f))

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_paths = sorted(in_dir.rglob("*.wav"))
    print(f"[preprocess_hifigan_mels] Found {len(wav_paths)} wav files under {in_dir}")

    for wav_path in wav_paths:
        rel = wav_path.relative_to(in_dir)
        out_path = out_dir / rel.with_suffix(".npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Load audio as HiFi-GAN expects
        y, sr = librosa.load(str(wav_path), sr=h.sampling_rate, mono=True)

        y_tensor = torch.from_numpy(y).float().unsqueeze(0)  # (1,T)

        with torch.no_grad():
            mel = mel_spectrogram(
                y_tensor,
                h.n_fft,
                h.num_mels,
                h.sampling_rate,
                h.hop_size,
                h.win_size,
                h.fmin,
                h.fmax,
                center=False,
            )  # (1, num_mels, T)

        mel_np = mel.squeeze(0).cpu().numpy()  # (num_mels, T)
        np.save(out_path, mel_np)
        print(f"[preprocess_hifigan_mels] Wrote {out_path}")

if __name__ == "__main__":
    main()
