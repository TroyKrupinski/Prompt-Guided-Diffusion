import argparse, numpy as np, torch, librosa, soundfile as sf
from pathlib import Path

from models.unet_diffusion import DiffusionUNet
from models.text_encoder import DistilBERTEncoder
from diffusion.scheduler import DiffusionScheduler
from vocoder.hifigan import HiFiGANVocoder

SR = 22050  # HiFi-GAN UNIVERSAL_V1 uses 22050 Hz :contentReference[oaicite:3]{index=3}
N_MELS_DIFF = 128  # your diffusion model's mel bins (training)
SEG_FRAMES = 860   # or whatever you used in train_diffusion

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=str, required=True)
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, default="diffusion_hifigan.wav")
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--hifigan_config", type=str, required=True)
    ap.add_argument("--hifigan_ckpt", type=str, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[infer_diff] Loading source audio:", args.wav)
    y_src, sr = librosa.load(args.wav, sr=SR, mono=True)
    print(f"[infer_diff] Source duration: {len(y_src)/SR:.2f}s")

    print("[infer_diff] Loading diffusion model checkpoint:", args.ckpt)
    unet = DiffusionUNet().to(device).eval()
    txt  = DistilBERTEncoder().to(device).eval()
    state = torch.load(args.ckpt, map_location=device)
    unet.load_state_dict(state["model"])
    txt.load_state_dict(state["txt"])

    sched = DiffusionScheduler(timesteps=args.timesteps, device=device)

    with torch.no_grad():
        txt_emb = txt([args.prompt], device=device)  # (1,512)

    # Shape for generated mel
    shape = (1, 1, N_MELS_DIFF, SEG_FRAMES)

    
    print(f"[infer_diff] Sampling diffusion with {args.steps} steps...")
    # Sample diffusion mel: (B, 1, N_MELS_DIFF, T)
    mel_noisy = sched.p_sample_loop(unet, shape, txt_emb, device=device, steps=args.steps)
    # mel_noisy: (1, 1, N_MELS_DIFF, T)

    # ---- Prepare mel for HiFi-GAN ----
    # Drop the channel dimension -> (1, N_MELS_DIFF, T)
    mel_for_vocoder = mel_noisy.squeeze(1)  # (1, 128, T) if N_MELS_DIFF=128

    # Interpolate 128 -> 80 mel bins if needed
    if mel_for_vocoder.shape[1] != 80:
        print("[infer_diff] Interpolating diffusion mel from",
              mel_for_vocoder.shape[1], "to 80 mel bins for HiFi-GAN...")
        mel_for_vocoder = mel_for_vocoder.unsqueeze(1)  # (1,1,M,T)
        mel_for_vocoder = torch.nn.functional.interpolate(
            mel_for_vocoder,
            size=(80, mel_for_vocoder.shape[-1]),  # (H=80, W=T)
            mode="bilinear",
            align_corners=False,
        )
        mel_for_vocoder = mel_for_vocoder.squeeze(1)  # (1,80,T)

    # Clamp to [0,1] to avoid crazy values
    mel_for_vocoder = torch.clamp(mel_for_vocoder, 0.0, 1.0)

    # Normalize and map to a log-mel-ish scale
    mean = mel_for_vocoder.mean()
    std = mel_for_vocoder.std() + 1e-6
    mel_for_vocoder = (mel_for_vocoder - mean) / std

    # Spread it a bit, roughly target ~[-12, 4]
    mel_for_vocoder = mel_for_vocoder * 4.0 - 4.0  # (1,80,T)



    print("[infer_diff] Initializing HiFi-GAN...")
    vocoder = HiFiGANVocoder(
        config_path=args.hifigan_config,
        ckpt_path=args.hifigan_ckpt,
        device=device
    )

    print("[infer_diff] Running HiFi-GAN vocoder...")
    audio_gen = vocoder.mel_to_audio(mel_for_vocoder)  # numpy, mono

    # Concatenate original + continuation
    y_out = np.concatenate([y_src, audio_gen], axis=0)
    peak = np.max(np.abs(y_out)) + 1e-9
    if peak > 1.0:
        y_out = y_out / peak

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), y_out, SR)
    print("[infer_diff] Wrote", out_path, "| duration:", len(y_out)/SR)

if __name__ == "__main__":
    main()
