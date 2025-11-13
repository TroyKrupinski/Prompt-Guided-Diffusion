Prompt-Guided Diffusion for Music Continuation

This repository contains a full text-conditioned music continuation system built from scratch.
It includes two pipelines trained on the FMA-Medium (22GB) dataset:

Pipeline A — Baseline

Conditional 2D UNet denoiser

128-bin log-mels

Griffin-Lim mel→audio

Learns to map noisy → clean mels conditioned on text prompts

Pipeline B — Diffusion + HiFi-GAN (True Diffusion)

Conditional Diffusion U-Net

80-bin HiFi-GAN log-mels

UNIVERSAL_V1 HiFi-GAN vocoder

Learns to follow text prompts via mel-space transformations

Produces higher fidelity continuations


1. Clone this repository
git clone https://github.com/TroyKrupinski/Prompt-Guided-Diffusion
cd Prompt-Guided-Diffusion

2. Clone HiFi-GAN (UNIVERSAL V1 vocoder)
git clone https://github.com/jik876/hifi-gan.git
# Make sure UNIVERSAL_V1 exists:
# hifi-gan/hifigan__universal_v1/config.json
# hifi-gan/hifigan__universal_v1/g_02500000

3. Create and activate environment
conda env create -f environment.yml
conda activate pgmd

4. Install project locally
pip install -e .



Dataset prep:
Convert MP3 → WAV (22.05kHz mono)
python scripts/prepare_datasets.py fma --src data/fma_medium --dst data/fma/wavs_medium --sr 22050
Baseline Mel-Spectrograms (128 bins)
python scripts/preprocess.py --in_dir data/fma/wavs_medium --out_dir data/fma/mels_medium
Diffusion Mel-Spectrograms (80-bin HiFi-GAN log-mels)
python scripts/preprocess_hifigan_mels.py --in_dir data/fma/wavs_medium --out_dir data/fma/mels_medium_hifigan --config hifi-gan/hifigan__universal_v1/config.json

Pipeline A:
python src/train.py --mels_dir data/fma/mels_medium --epochs 15 --batch_size 4 --lr 2e-4 --out_dir runs/fma_medium_baseline --plot
Inference — conditioned + unconditioned continuations
python src/compare_cond_uncond.py --wav data/fma/wavs_medium/000/000002.wav --prompt "increase BPM with energetic drums" --ckpt runs/fma_medium_baseline/ckpt_15.pt --out_prefix results/baseline_000002 --log_dir results/vis_baseline_000002

This produces:

baseline_000002_cond.wav

baseline_000002_uncond.wav

Spectrogram visualizations

Pipeline B — Diffusion + HiFi-GAN Vocoder
python src/train_diffusion_hifigan.py --mels_dir data/fma/mels_medium_hifigan --epochs 15 --batch_size 4 --lr 2e-4 --out_dir runs/fma_medium_diffusion_hifigan --timesteps 1000 --plot
Inference — text-conditioned HiFi-GAN continuation
python src/infer_diffusion.py --wav data/fma/wavs_medium/000/000002.wav --prompt "add reverb and ambient space" --ckpt runs/fma_medium_diffusion_hifigan/ckpt_diff_hifigan_15.pt --hifigan_config hifi-gan/hifigan__universal_v1/config.json --hifigan_ckpt hifi-gan/hifigan__universal_v1/g_02500000 --out results/diffusion_reverb_000002.wav --steps 100

Evaluation & Metrics

MSE + COSINE
python src/eval/eval_reconstruction.py --wav data/fma/wavs_medium/000/000002.wav --ckpt runs/fma_medium_baseline/ckpt_15.pt
2. CLAP Audio-Text Alignment

Baseline:
python src/eval/eval_clap.py --wav results/baseline_000002_cond.wav --prompt "increase BPM with energetic drums" | Tee-Object results/clap_baseline_increaseBPM_000002_cond.txt
Diffusion:
python src/eval/eval_clap.py --wav results/diffusion_increaseBPM_000002.wav --prompt "increase BPM with energetic drums" | Tee-Object results/clap_diffusion_increaseBPM_000002.txt
3. Prompt-Effect Consistency (baseline or diffusion)

python src/eval/eval_prompt_effect.py --mels_dir data/fma/mels_medium --ckpt runs/fma_medium_baseline/ckpt_15.pt --prompt_a "increase BPM" --prompt_b "decrease BPM" --limit 20
4. CLAP Bar Chart (visual comparison)
python scripts/make_clap_barplot.py --results_dir results --out_path results/clap_alignment_barplot.png

5. Training Curves
Baseline
runs/fma_medium_baseline/training_curve.png
Diffusion:
runs/fma_medium_diffusion_hifigan/training_curve.png
Prompt-Guided-Diffusion/
│
├── data/
│   └── fma/
│       ├── wavs_medium/
│       ├── mels_medium/
│       └── mels_medium_hifigan/
│
├── hifi-gan/
│   └── hifigan__universal_v1/
│       ├── config.json
│       └── g_02500000
│
├── runs/
│   ├── fma_medium_baseline/
│   └── fma_medium_diffusion_hifigan/
│
├── results/
│   ├── baseline_*.wav
│   ├── diffusion_*.wav
│   └── clap_*.txt
│
├── scripts/
│   ├── preprocess.py
│   ├── preprocess_hifigan_mels.py
│   ├── prepare_datasets.py
│   └── make_clap_barplot.py
│
└── src/
    ├── train.py
    ├── train_diffusion_hifigan.py
    ├── compare_cond_uncond.py
    ├── infer_diffusion.py
    ├── eval/
    └── models/


