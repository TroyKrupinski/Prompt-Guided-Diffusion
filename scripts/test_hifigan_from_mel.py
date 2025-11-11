import argparse
import numpy as np
import torch
import soundfile as sf
from pathlib import Path

from vocoder.hifigan import HiFiGANVocoder

SR = 22050

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mel", type=str, required=True,
                    help="Path to a .npy mel from data/fma/mels_hifigan")
    ap.add_argument("--config", type=str, required=True,
                    help="HiFi-GAN config.json")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="HiFi-GAN generator checkpoint (g_02500000)")
    ap.add_argument("--out", type=str, default="hifigan_test.wav")
    args = ap.parse_args()

    mel = np.load(args.mel)  # (80, T)
    mel = torch.from_numpy(mel).unsqueeze(0)  # (1,80,T)

    vocoder = HiFiGANVocoder(
        config_path=args.config,
        ckpt_path=args.ckpt,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    audio = vocoder.mel_to_audio(mel)
    audio = np.asarray(audio).squeeze()
    peak = np.max(np.abs(audio)) + 1e-9
    if peak > 1.0:
        audio = audio / peak

    out_path = Path(args.out).resolve()
    sf.write(str(out_path), audio, SR)
    print("[test_hifi] Wrote", out_path)

if __name__ == "__main__":
    main()
