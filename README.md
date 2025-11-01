# Prompt‑Guided Diffusion for Music Continuations (Bootstrap)

This repository is a **runnable starter** for your CSCE 5218 project.
It gives you an **end‑to‑end path** TODAY (Windows‑friendly):
- Create a Conda env
- Preprocess WAVs → Mel‑spectrograms
- Train a **text‑conditioned spectrogram autoencoder** (bootstrap)
- Run inference to produce a **continuation stub** (concatenate and denoise)

Later, swap the autoencoder core for a **diffusion U‑Net + scheduler** without changing data & logging.

## Quickstart (Windows / Conda)

```powershell
# 1) Clone or unzip
# If you downloaded the zip, extract it, then:
cd prompt_guided_music_diffusion

# 2) Create env (CUDA optional — CPU works too)
conda env create -f environment.yml
conda activate pgmd

# 3) (Optional) If PyTorch didn't install with CUDA, install your build from:
# https://pytorch.org/get-started/locally/

# 4) Install package in editable mode (local imports)
pip install -e .

# 5) Put a few short WAV files (22.05 kHz mono) in data/wavs/
# You can test with any music snippets you have locally.

# 6) Preprocess → Mel‑spectrograms
python scripts/preprocess.py --in_dir data/wavs --out_dir data/mels

# 7) Train bootstrap model
python src/train.py --mels_dir data/mels --epochs 5 --batch_size 4 --out_dir runs/exp1

# 8) Inference: generate a continuation stub for an input WAV + text prompt
python src/infer.py --wav path/to/song.wav --prompt "slower tempo with mellow piano" --out out.wav
```

## Repo Layout

```
prompt_guided_music_diffusion/
├─ README.md
├─ requirements.txt
├─ environment.yml
├─ pyproject.toml
├─ setup.cfg
├─ .gitignore
├─ LICENSE
├─ configs/
│  └─ default.yaml
├─ data/
│  ├─ wavs/           # put your .wav files here (22.05 kHz mono)
│  └─ mels/           # preprocessed .npy mel tensors
├─ scripts/
│  └─ preprocess.py   # wav → mel .npy
└─ src/
   ├─ __init__.py
   ├─ train.py
   ├─ infer.py
   ├─ audio/
   │  ├─ __init__.py
   │  └─ utils.py
   ├─ models/
   │  ├─ __init__.py
   │  ├─ audio_encoder.py     # small CNN over mels
   │  ├─ text_encoder.py      # DistilBERT wrapper
   │  └─ unet_conditioned.py  # small U‑Net "denoiser" with text conditioning
   └─ eval/
      ├─ __init__.py
      └─ metrics.py           # MSE, cosine similarity stubs
```

## What’s Included vs. What’s Next

**Included (runnable today):**
- Mel pipeline with `librosa`
- Lightweight **U‑Net denoiser** that learns to reconstruct clean mels from noisy mels
- **Text conditioning** via DistilBERT pooled embeddings (late fusion)
- Training loop w/ checkpoints and sample exports
- Inference that appends a predicted continuation segment

**Next (after bootstrap works):**
- Replace denoiser with **diffusion U‑Net** + noise schedule
- Add **cross‑attention** conditioning (audio+text)
- Add **HiFi‑GAN** vocoder for WAV synthesis from mels
- Add **CLAP** alignment score, FAD, and ablations


## Dataset Preparation (MAESTRO / FMA)

**MAESTRO → WAV 22.05kHz mono**
```bash
python scripts/prepare_datasets.py maestro --src /path/to/maestro-v3.0/ --dst data/maestro/wavs
python scripts/preprocess.py --in_dir data/maestro/wavs --out_dir data/maestro/mels
```

**FMA → WAV 22.05kHz mono (small/medium/large)**
```bash
python scripts/prepare_datasets.py fma --src /path/to/fma_small/ --dst data/fma/wavs
python scripts/preprocess.py --in_dir data/fma/wavs --out_dir data/fma/mels
```

**FMA prompts from metadata**
```bash
python scripts/prepare_datasets.py fma_prompts --tracks_csv /path/to/tracks.csv --genres_csv /path/to/genres.csv --out data/fma/prompts.jsonl
```

> Tip: You can train separate runs on MAESTRO and FMA to compare prompt adherence and continuity across domains.
