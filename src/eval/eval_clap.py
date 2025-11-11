import argparse, torch, librosa, numpy as np
from pathlib import Path
from laion_clap import CLAP_Module  # pre-trained CLAP

SR = 48000  # CLAP default

def load_audio(path, target_sr=SR, max_len_seconds=30.0):
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    max_len = int(max_len_seconds * target_sr)
    if len(y) > max_len:
        y = y[:max_len]
    return y, target_sr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=str, required=True)
    ap.add_argument("--prompt", type=str, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLAP_Module(enable_fusion=False).to(device)
    model.eval()

    audio, sr = load_audio(args.wav)
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(device)  # (1,T)

    with torch.no_grad():
        audio_emb = model.get_audio_embedding_from_data(
            audio_tensor, use_tensor=True
        )  # (1,D)
        text_emb = model.get_text_embedding([args.prompt])  # (1,D)

        cos = torch.nn.functional.cosine_similarity(audio_emb, text_emb, dim=1)
        print(f"CLAP cosine similarity: {cos.item():.4f}")

if __name__ == "__main__":
    main()
