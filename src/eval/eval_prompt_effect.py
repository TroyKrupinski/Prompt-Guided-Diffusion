import argparse, numpy as np, torch
from pathlib import Path

from models.unet_conditioned import UNetDenoiser
from models.text_encoder import DistilBERTEncoder
from models.audio_encoder import SmallAudioEncoder

class MelDatasetOnce(torch.utils.data.Dataset):
    def __init__(self, mels_dir, segment_frames=860, limit=100):
        self.paths = list(Path(mels_dir).glob("**/*.npy"))[:limit]
        self.segment_frames = segment_frames
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        mel = np.load(self.paths[idx])
        M, T = mel.shape
        if T < self.segment_frames:
            rep = int(np.ceil(self.segment_frames / T))
            mel = np.tile(mel, (1, rep))[:, :self.segment_frames]
        else:
            start = np.random.randint(0, T - self.segment_frames + 1)
            mel = mel[:, start:start+self.segment_frames]
        mel = torch.from_numpy(mel).unsqueeze(0)  # (1,M,T)
        return mel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mels_dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--prompt_a", type=str, default="energetic rock with strong drums")
    ap.add_argument("--prompt_b", type=str, default="slow ambient pads")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = MelDatasetOnce(args.mels_dir, limit=args.limit)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    unet = UNetDenoiser().to(device).eval()
    txt  = DistilBERTEncoder().to(device).eval()
    audio_enc = SmallAudioEncoder().to(device).eval()

    state = torch.load(args.ckpt, map_location=device)
    unet.load_state_dict(state["model"])
    txt.load_state_dict(state["txt"])

    diffs = []

    with torch.no_grad():
        for clean in dl:
            clean = clean.to(device)
            noisy = (clean + 0.1 * torch.randn_like(clean)).clamp(0, 1)
            B = clean.size(0)

            emb_a = txt([args.prompt_a], device=device).repeat(B, 1)
            emb_b = txt([args.prompt_b], device=device).repeat(B, 1)

            pred_a = unet(noisy, emb_a)
            pred_b = unet(noisy, emb_b)

            ea = audio_enc(pred_a).view(B, -1)
            eb = audio_enc(pred_b).view(B, -1)

            # cosine_sim(ea, eb) → we want "difference"
            # 1 - cosine similarity is a simple dissimilarity.
            cos_sim = torch.nn.functional.cosine_similarity(ea, eb, dim=1)
            diffs.extend((1 - cos_sim).cpu().numpy().tolist())

    print(f"Mean prompt-induced embedding distance (1 - cos): {np.mean(diffs):.4f}")

if __name__ == "__main__":
    main()
