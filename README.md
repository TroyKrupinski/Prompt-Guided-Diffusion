1. Preprocess Audio (MP3 → WAV @ 22.05kHz Mono)
python scripts/prepare_datasets.py fma --src data/fma_medium --dst data/fma/wavs_medium --sr 22050

2A. Baseline Mel Spectrograms (128-bin mel)
python scripts/preprocess.py --in_dir data/fma/wavs_medium --out_dir data/fma/mels_medium
2B. Diffusion Mel Spectrograms (80-bin HiFi-GAN log-mel)
python scripts/preprocess_hifigan_mels.py --in_dir data/fma/wavs_medium --out_dir data/fma/mels_medium_hifigan

3A. Train Baseline Pipeline (UNet + Griffin-Lim)
python src/train.py --mels_dir data/fma/mels_medium --epochs 15 --batch_size 4 --lr 2e-4 --out_dir runs/fma_medium_baseline --plot
3B. Train Diffusion Pipeline (Diffusion UNet + HiFi-GAN)
python src/train_diffusion_hifigan.py --mels_dir data/fma/mels_medium_hifigan --epochs 15 --batch_size 4 --lr 2e-4 --out_dir runs/fma_medium_diffusion_hifigan --timesteps 1000 --plot


4A. Baseline Inference (Conditioned & Unconditioned)
python src/compare_cond_uncond.py --wav data/fma/wavs_medium/000/000002.wav --prompt "increase BPM with energetic drums" --ckpt runs/fma_medium_baseline/ckpt_15.pt --out_prefix results/baseline_000002 --log_dir results/vis_baseline_000002
4B. Diffusion Inference (HiFi-GAN continuation)
python src/infer_diffusion.py --wav data/fma/wavs_medium/000/000002.wav --prompt "increase BPM with energetic drums" --ckpt runs/fma_medium_diffusion_hifigan/ckpt_diff_hifigan_15.pt --hifigan_config hifi-gan/hifigan__universal_v1/config.json --hifigan_ckpt hifi-gan/hifigan__universal_v1/g_02500000 --out results/diffusion_increaseBPM_000002.wav --steps 100

5. Reconstruction Metrics (Baseline Only)
python src/eval/eval_reconstruction.py --wav data/fma/wavs_medium/000/000002.wav --ckpt runs/fma_medium_baseline/ckpt_15.pt


6. CLAP Alignment Evaluation (Both Pipelines)
Baseline
python src/eval/eval_clap.py --wav results/baseline_000002_cond.wav --prompt "increase BPM with energetic drums" | Tee-Object results/clap_baseline_increaseBPM_000002_cond.txt
Diffusion
python src/eval/eval_clap.py --wav results/diffusion_increaseBPM_000002.wav --prompt "increase BPM with energetic drums" | Tee-Object results/clap_diffusion_increaseBPM_000002.txt

7. Prompt Effect Consistency Check (Baseline or Diffusion)
python src/eval/eval_prompt_effect.py --mels_dir data/fma/mels_medium --ckpt runs/fma_medium_baseline/ckpt_15.pt --prompt_a "increase BPM" --prompt_b "decrease BPM" --limit 20 --batch_size 8

8. Generate Training Curves (Manual)
Baseline
python src/utils_logging.py --plot loss.csv
Diffusion
python src/utils_logging.py --plot eps_mse.csv

CLAP Bar chart
python scripts/make_clap_barplot.py --results_dir results --out_path results/clap_alignment_barplot.png
