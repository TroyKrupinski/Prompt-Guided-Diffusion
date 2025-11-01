#!/usr/bin/env python
import argparse, os, json, csv, sys
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import librosa
import soundfile as sf

SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}

def resample_to_wav(in_path: Path, out_path: Path, sr: int = 22050, mono: bool = True):
    """Load arbitrary audio file, resample to sr, convert to mono, write 16-bit PCM WAV."""
    y, _sr = librosa.load(str(in_path), sr=sr, mono=mono)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), y, sr, subtype="PCM_16")

def crawl_audio_files(root: Path) -> List[Path]:
    files = []
    for p in root.rglob("*"):
        if p.suffix.lower() in SUPPORTED_AUDIO_EXTS:
            files.append(p)
    return files

def prepare_maestro(maestro_root: Path, out_wavs_dir: Path, sr: int = 22050):
    """Process MAESTRO audio to 22.05kHz mono WAVs preserving split folder structure if present."""
    audio_files = crawl_audio_files(maestro_root)
    if not audio_files:
        print(f"[MAESTRO] No audio files found in {maestro_root}. Did you extract the dataset?")
        return
    print(f"[MAESTRO] Found {len(audio_files)} audio files.")
    for src in audio_files:
        # preserve relative path starting from maestro_root
        rel = src.relative_to(maestro_root)
        dst = out_wavs_dir / rel.with_suffix(".wav")
        try:
            resample_to_wav(src, dst, sr=sr, mono=True)
            print(f"[MAESTRO] Wrote {dst}")
        except Exception as e:
            print(f"[MAESTRO] Failed {src}: {e}")

def prepare_fma(fma_root: Path, out_wavs_dir: Path, sr: int = 22050):
    """Process FMA (small/medium/large) audio to 22.05kHz mono WAVs; preserves folder layout (e.g., 000/000123.mp3)."""
    audio_files = crawl_audio_files(fma_root)
    if not audio_files:
        print(f"[FMA] No audio files found in {fma_root}. Ensure you've extracted the FMA archive(s).")
        return
    print(f"[FMA] Found {len(audio_files)} audio files.")
    for src in audio_files:
        rel = src.relative_to(fma_root)
        dst = out_wavs_dir / rel.with_suffix(".wav")
        try:
            resample_to_wav(src, dst, sr=sr, mono=True)
            print(f"[FMA] Wrote {dst}")
        except Exception as e:
            print(f"[FMA] Failed {src}: {e}")

def generate_fma_prompts(tracks_csv: Path, genres_csv: Optional[Path], out_jsonl: Path):
    """Create prompt strings from FMA metadata.
    Output JSONL with fields: {"track_id": int, "prompt": str}
    If genres.csv is provided, we decode top-level/leaf names; otherwise use 'genre_id' columns.
    """
    # Load genre id -> name if available
    genre_names: Dict[int, str] = {}
    if genres_csv and genres_csv.exists():
        with open(genres_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    gid = int(row.get("genre_id") or row.get("id"))
                    name = row.get("title") or row.get("name") or ""
                    if gid is not None and name:
                        genre_names[gid] = name
                except Exception:
                    continue

    # FMA tracks.csv has a complex multi-index; many exports flatten columns.
    # We support a flexible read: try DictReader and look for reasonable fields.
    rows = []
    with open(tracks_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    def first_available(row, keys: List[str], default=""):
        for k in keys:
            if k in row and row[k]:
                return row[k]
        return default

    # Common keys that may exist depending on the FMA CSV variant
    title_keys  = ["track_title", "title", "track.name", "name"]
    artist_keys = ["artist_name", "artist", "artist.name"]
    genre_keys  = ["genre_top", "genre", "track.genre_top", "genres"]

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    c = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            tid_str = first_available(row, ["track_id", "id", "track.id"], default="")
            if not tid_str:
                continue
            try:
                tid = int(tid_str)
            except Exception:
                continue

            title  = first_available(row, title_keys, default="unknown track")
            artist = first_available(row, artist_keys, default="unknown artist")
            genre  = first_available(row, genre_keys, default="")

            # If 'genres' is a list of ids, decode via genres.csv if available
            if genre and genre.isdigit() and genre_names:
                gname = genre_names.get(int(genre), "")
                if gname:
                    genre = gname

            # Prompt template
            parts = []
            if genre:  parts.append(f"genre: {genre}")
            if artist: parts.append(f"artist: {artist}")
            parts.append("describe continuation: keep structure and melody while following the prompt")
            prompt = "; ".join(parts)

            f.write(json.dumps({"track_id": tid, "prompt": prompt}, ensure_ascii=False) + "\n")
            c += 1
    print(f"[FMA] Wrote {c} prompt lines → {out_jsonl}")

def main():
    ap = argparse.ArgumentParser(description="Prepare MAESTRO and/or FMA datasets for the project.")
    sub = ap.add_subparsers(dest="cmd")

    # MAESTRO
    ap_mae = sub.add_parser("maestro", help="Process MAESTRO audio to 22.05kHz mono WAVs")
    ap_mae.add_argument("--src", type=str, required=True, help="Path to extracted MAESTRO root")
    ap_mae.add_argument("--dst", type=str, default="data/maestro/wavs", help="Output WAV folder")
    ap_mae.add_argument("--sr", type=int, default=22050)

    # FMA audio
    ap_fma = sub.add_parser("fma", help="Process FMA audio to 22.05kHz mono WAVs")
    ap_fma.add_argument("--src", type=str, required=True, help="Path to extracted FMA audio root (e.g., fma_small/)")
    ap_fma.add_argument("--dst", type=str, default="data/fma/wavs", help="Output WAV folder")
    ap_fma.add_argument("--sr", type=int, default=22050)

    # FMA prompts
    ap_fmap = sub.add_parser("fma_prompts", help="Generate prompts JSONL from FMA metadata CSVs")
    ap_fmap.add_argument("--tracks_csv", type=str, required=True, help="Path to FMA tracks.csv (flattened)")
    ap_fmap.add_argument("--genres_csv", type=str, default="", help="Path to FMA genres.csv (optional)")
    ap_fmap.add_argument("--out", type=str, default="data/fma/prompts.jsonl")

    args = ap.parse_args()

    if args.cmd == "maestro":
        prepare_maestro(Path(args.src), Path(args.dst), sr=args.sr)
    elif args.cmd == "fma":
        prepare_fma(Path(args.src), Path(args.dst), sr=args.sr)
    elif args.cmd == "fma_prompts":
        genres_csv = Path(args.genres_csv) if args.genres_csv else None
        generate_fma_prompts(Path(args.tracks_csv), genres_csv, Path(args.out))
    else:
        print("No command provided. Use one of: maestro | fma | fma_prompts")
        sys.exit(1)

if __name__ == "__main__":
    main()
