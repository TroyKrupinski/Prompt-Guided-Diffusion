import argparse, numpy as np, torch, soundfile as sf, librosa
from pathlib import Path
from models.unet_conditioned import UNetDenoiser
from models.text_encoder import DistilBERTEncoder
from audio.utils import mel_from_wav
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=str, required=True)
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--out", type=str, default="out.wav")
    ap.add_argument("--sr", type=int, default=22050)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y, sr = librosa.load(args.wav, sr=args.sr, mono=True)

    mel = mel_from_wav(y, sr=sr)  # (M,T) in [0,1]
    mel_t = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,M,T)

    # load latest checkpoint if exists; otherwise random weights for demo
    ckpts = sorted(Path("runs/exp1").glob("ckpt_*.pt"))
    model = UNetDenoiser().to(device).eval()
    txt   = DistilBERTEncoder().to(device).eval()
    if ckpts:
        state = torch.load(str(ckpts[-1]), map_location=device)
        model.load_state_dict(state["model"])
        txt.load_state_dict(state["txt"])

    # naive "continuation": denoise the last chunk & loop it once (stub)
    # (You will replace with diffusion-based next-chunk prediction.)
    txt_emb = txt([args.prompt], device=device)
    with torch.no_grad():
        pred = model(mel_t, txt_emb)  # (1,1,M,T)
    pred = pred.squeeze().cpu().numpy()

    # Convert mel back to a simple waveform via Griffin-Lim (placeholder)
    S_db = pred * 80.0 - 80.0  # approx invert [0,1] → dB [-80,0]
    S = librosa.db_to_power(S_db)
    y_hat = librosa.feature.inverse.mel_to_audio(S, sr=sr, n_iter=32)
    # Concatenate original + "continuation"
    y_out = np.concatenate([y, y_hat], axis=0)
    sf.write(args.out, y_out, sr)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
