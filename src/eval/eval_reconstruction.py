import argparse, numpy as np, torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from models.unet_conditioned import UNetDenoiser
from models.text_encoder import DistilBERTEncoder
from models.audio_encoder import SmallAudioEncoder
from eval.metrics import mse, cosine_sim

class MelDataset(Dataset):
    def __init__(self, mels_dir, segment_frames=860):
        self.paths = list(Path(mels_dir).glob("**/*.npy"))
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

def run_eval(mels_dir, ckpt, prompt, use_text=True, max_batches=200, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = MelDataset(mels_dir)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    unet = UNetDenoiser().to(device).eval()
    txt  = DistilBERTEncoder().to(device).eval()
    audio_enc = SmallAudioEncoder().to(device).eval()

    state = torch.load(ckpt, map_location=device)
    unet.load_state_dict(state["model"])
    # unconditional checkpoint won't have txt; guard for that
    try:
        txt.load_state_dict(state["txt"])
    except Exception:
        pass

    all_mse, all_cos = [], []
    with torch.no_grad():
        for i, clean in enumerate(dl):
            if i >= max_batches:
                break
            clean = clean.to(device)
            noisy = (clean + 0.1 * torch.randn_like(clean)).clamp(0,1)

            B = clean.size(0)
            if use_text:
                emb = txt([prompt], device=device).repeat(B, 1)
            else:
                emb = torch.zeros(B, 512, device=device)

            pred = unet(noisy, emb)

            all_mse.append(mse(pred, clean).item())

            ce = audio_enc(clean).view(B, -1)
            pe = audio_enc(pred).view(B, -1)
            all_cos.append(cosine_sim(ce, pe).item())

    return float(np.mean(all_mse)), float(np.mean(all_cos))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mels_dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--prompt", type=str, default="energetic rock")
    ap.add_argument("--use_text", action="store_true")
    ap.add_argument("--out", type=str, default=None, help="Optional file to save output metrics")

    args = ap.parse_args()
    
    mse_val, cos_val = run_eval(args.mels_dir, args.ckpt, args.prompt, use_text=args.use_text)
    print(f"MSE: {mse_val:.6f}")
    print(f"Cosine similarity: {cos_val:.4f}")
    if args.out:
        try:
            with open(args.out, "w") as f:
                # Write available metrics safely
                if 'mse' in locals():
                    f.write(f"MSE: {mse:.6f}\n")
                if 'cos' in locals():
                    f.write(f"Cosine similarity: {cos:.6f}\n")
                f.write(f"Prompt: {args.prompt}\n")
            print(f"\nResults saved to {args.out}")
        except Exception as e:
            print(f"Failed to write output file: {e}")