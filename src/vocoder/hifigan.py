import json
import torch
import numpy as np
from pathlib import Path
import importlib.util
import sys

# Locate the cloned hifi-gan repo (sibling of src/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
HIFIGAN_ROOT = PROJECT_ROOT / "hifi-gan"

# Make sure hifi-gan is on sys.path so its internal imports (e.g., "from utils import ...") work
sys.path.insert(0, str(HIFIGAN_ROOT))

# --- load env.py from hifi-gan as its own module ---
env_path = HIFIGAN_ROOT / "env.py"
env_spec = importlib.util.spec_from_file_location("hifigan_env", env_path)
hifigan_env = importlib.util.module_from_spec(env_spec)
env_spec.loader.exec_module(hifigan_env)
AttrDict = hifigan_env.AttrDict

# --- load models.py from hifi-gan as its own module ---
models_path = HIFIGAN_ROOT / "models.py"
models_spec = importlib.util.spec_from_file_location("hifigan_models", models_path)
hifigan_models = importlib.util.module_from_spec(models_spec)
models_spec.loader.exec_module(hifigan_models)
Generator = hifigan_models.Generator


class HiFiGANVocoder:
    """
    Wrapper around UNIVERSAL_V1 HiFi-GAN (g_02500000 + config.json).
    Expects mel of shape (1, num_mels, T) roughly matching the config.
    """
    def __init__(self, config_path: str, ckpt_path: str, device: str = "cpu"):
        self.device = torch.device(device)

        # Load config (JSON -> AttrDict) exactly like hifi-gan does
        with open(config_path, "r") as f:
            data = f.read()
        h = AttrDict(json.loads(data))

        self.h = h
        self.generator = Generator(h).to(self.device)

        # Load checkpoint
        state = torch.load(ckpt_path, map_location=self.device)
        # official checkpoints store weights under "generator"
        if "generator" in state:
            self.generator.load_state_dict(state["generator"])
        else:
            self.generator.load_state_dict(state)

        self.generator.eval()
        self.generator.remove_weight_norm()  # same as inference.py in hifi-gan

        for p in self.generator.parameters():
            p.requires_grad = False

        print("[HiFiGAN] Loaded config from", config_path)
        print("[HiFiGAN] Loaded weights from", ckpt_path)
        print("[HiFiGAN] num_mels:", h.num_mels, "sr:", h.sampling_rate)

    @torch.no_grad()
    def mel_to_audio(self, mel: torch.Tensor) -> np.ndarray:
        """
        mel: torch.Tensor (1, num_mels, T)
        Must roughly match HiFi-GAN's expected mel: 80 bins, log-mel, etc.
        """
        mel = mel.to(self.device)
        audio = self.generator(mel) #(1,1,T)
        audio = audio.squeeze() #> 1D (T,)
        return audio.cpu().numpy()
