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
# 1) Clone or unzip dataset of your choice
cd prompt_guided_music_diffusion

# 2) Create env 
conda env create -f environment.yml
conda activate pgmd

# 4) Install package in editable mode (local imports)
pip install -e .

# 5) Put a few short WAV files (22.05 kHz mono) in data/wavs/

# 6) Preprocess → Mel‑spectrograms
python scripts/preprocess.py --in_dir data/wavs --out_dir data/mels

# 7) Train bootstrap model
python src/train.py --mels_dir data/mels --epochs 5 --batch_size 4 --out_dir runs/exp1

# 8) Inference: generate a continuation stub for an input WAV + text prompt
python src/infer.py --wav path/to/song.wav --prompt "slower tempo with mellow piano" --out out.wav
```


## What’s Included vs. What’s Next

**Included (runnable today):**
- Mel pipeline with `librosa`
- Lightweight **U‑Net denoiser** that learns to reconstruct clean mels from noisy mels
- **Text conditioning** via DistilBERT pooled embeddings (late fusion)
- Training loop w/ checkpoints and sample exports
- Inference that appends a predicted continuation segment




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
