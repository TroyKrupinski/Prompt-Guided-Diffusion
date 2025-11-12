# src/eval/eval_clap.py
import argparse, sys
from pathlib import Path

import numpy as np
import torch
import librosa
from laion_clap import CLAP_Module  # pip install laion-clap

TARGET_SR = 48_000   # CLAP models expect 48 kHz
CLIP_SEC  = 10.0     # evaluate on a ~10s window by default


def load_for_clap(wav_path: str, target_sr: int = TARGET_SR,
                  clip_seconds: float = CLIP_SEC, center_crop: bool = True) -> torch.Tensor:
    """Load mono audio, resample to 48k, center-crop or pad to clip_seconds, peak-normalize to [-1, 1]."""
    y, sr = librosa.load(wav_path, sr=None, mono=True)     # load native
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    need = int(clip_seconds * sr)
    if len(y) >= need:
        if center_crop:
            start = (len(y) - need) // 2
            y = y[start:start + need]
        else:
            y = y[:need]
    else:
        y = np.pad(y, (0, need - len(y)))

    peak = float(np.max(np.abs(y)) + 1e-9)
    y = (y / peak).astype(np.float32)                      # [-1, 1]
    return torch.from_numpy(y).unsqueeze(0)                # (1, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav",     type=str, required=True, help="Path to audio file (wav)")
    ap.add_argument("--prompt",  type=str, required=True, help="Text prompt")
    ap.add_argument("--seconds", type=float, default=CLIP_SEC, help="Seconds of audio to evaluate (default 10)")
    ap.add_argument("--edge",    action="store_true", help="Use head crop instead of center crop")
    ap.add_argument("--device",  type=str, default="auto", choices=["auto","cpu","cuda"], help="Inference device")
    args = ap.parse_args()

    device = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        ("cpu"  if args.device == "auto" else args.device)
    )

    # Load CLAP
    model = CLAP_Module(enable_fusion=False).to(device)
    model.eval()

    # Prep audio
    audio = load_for_clap(args.wav, target_sr=TARGET_SR, clip_seconds=args.seconds, center_crop=not args.edge).to(device)

    with torch.no_grad():
        # Get embeddings as tensors on same device
        audio_emb = model.get_audio_embedding_from_data(audio, use_tensor=True).to(device)          # (1, D)
        text_emb  = model.get_text_embedding([args.prompt], use_tensor=True).to(device)             # (1, D)

        # L2-normalize improves cosine stability
        audio_emb = torch.nn.functional.normalize(audio_emb, dim=1)
        text_emb  = torch.nn.functional.normalize(text_emb,  dim=1)

        cos = torch.nn.functional.cosine_similarity(audio_emb, text_emb, dim=1)                     # (1,)
        score = float(cos.item())

    print(f"CLAP cosine similarity: {score:.4f}")
    # Optional: print a JSON line for easy parsing downstream
    # import json; print(json.dumps({"wav": args.wav, "prompt": args.prompt, "score": score}))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[eval_clap] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
