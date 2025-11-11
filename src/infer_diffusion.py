import argparse
import numpy as np
import torch
import librosa
import soundfile as sf
from pathlib import Path

from models.unet_diffusion import DiffusionUNet
from models.text_encoder import DistilBERTEncoder
from diffusion.scheduler import DiffusionScheduler
from vocoder.hifigan import HiFiGANVocoder

SR = 22050           # HiFi-GAN sampling rate (UNIVERSAL_V1)
N_MELS_DIFF = 80     # diffusion model trained on 80-mel HiFi-GAN log mels
SEG_FRAMES = 860     # same segment length used in train_diffusion_hifigan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=str, required=True,
                    help="Source WAV file to continue")
    ap.add_argument("--prompt", type=str, required=True,
                    help="Text prompt to condition diffusion on")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Diffusion checkpoint (ckpt_diff_hifigan_*.pt)")
    ap.add_argument("--out", type=str, default="diffusion_hifigan.wav",
                    help="Output WAV path")
    ap.add_argument("--timesteps", type=int, default=1000,
                    help="Total diffusion timesteps T used in training")
    ap.add_argument("--steps", type=int, default=100,
                    help="Number of reverse steps to use at inference")
    ap.add_argument("--hifigan_config", type=str, required=True,
                    help="HiFi-GAN config.json (UNIVERSAL_V1)")
    ap.add_argument("--hifigan_ckpt", type=str, required=True,
                    help="HiFi-GAN generator checkpoint (g_02500000)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 1) Load source audio (for concatenation / continuation)
    # ------------------------------------------------------------------
    print("[infer_diff] Loading source audio:", args.wav)
    y_src, sr = librosa.load(args.wav, sr=SR, mono=True)
    print(f"[infer_diff] Source duration: {len(y_src) / SR:.2f}s")

    # ------------------------------------------------------------------
    # 2) Load diffusion UNet + text encoder from checkpoint
    # ------------------------------------------------------------------
    print("[infer_diff] Loading diffusion model checkpoint:", args.ckpt)
    unet = DiffusionUNet().to(device).eval()
    txt = DistilBERTEncoder().to(device).eval()

    state = torch.load(args.ckpt, map_location=device)
    unet.load_state_dict(state["model"])
    txt.load_state_dict(state["txt"])

    # ------------------------------------------------------------------
    # 3) Build scheduler + text embedding
    # ------------------------------------------------------------------
    sched = DiffusionScheduler(timesteps=args.timesteps, device=device)

    with torch.no_grad():
        txt_emb = txt([args.prompt], device=device)  # (1, 512)

    # Shape of the generated mel (B, C=1, M=80, T=SEG_FRAMES)
    shape = (1, 1, N_MELS_DIFF, SEG_FRAMES)

    # ------------------------------------------------------------------
    # 4) Sample mel from diffusion (multi-step)
    # ------------------------------------------------------------------
    print(f"[infer_diff] Sampling diffusion with {args.steps} steps...")
    mel_sample = sched.p_sample_loop(
        unet,
        shape,
        txt_emb,
        device=device,
        steps=args.steps,
    )  # (1, 1, 80, T)

    # Drop the channel dimension -> (1, 80, T)
    mel_for_vocoder = mel_sample.squeeze(1)  # (1, 80, T)

    # Since diffusion was trained directly on HiFi-GAN log mels, we do NOT
    # rescale or interpolate. Optionally clamp to avoid extreme values.
    mel_for_vocoder = torch.clamp(mel_for_vocoder, min=-12.0, max=4.0)

    # ------------------------------------------------------------------
    # 5) Run HiFi-GAN vocoder
    # ------------------------------------------------------------------
    print("[infer_diff] Initializing HiFi-GAN...")
    vocoder = HiFiGANVocoder(
        config_path=args.hifigan_config,
        ckpt_path=args.hifigan_ckpt,
        device=device,
    )

    print("[infer_diff] Running HiFi-GAN vocoder...")
    audio_gen = vocoder.mel_to_audio(mel_for_vocoder)
    audio_gen = np.asarray(audio_gen).squeeze()  # ensure 1D (T,)

    # ------------------------------------------------------------------
    # 6) Concatenate original + continuation and write to disk
    # ------------------------------------------------------------------
    y_out = np.concatenate([y_src, audio_gen], axis=0)

    # Simple peak normalization to avoid clipping
    peak = np.max(np.abs(y_out)) + 1e-9
    if peak > 1.0:
        y_out = y_out / peak

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), y_out, SR)

    print("[infer_diff] Wrote", out_path, "| duration:", len(y_out) / SR)


if __name__ == "__main__":
    main()
