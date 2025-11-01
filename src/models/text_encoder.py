import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class DistilBERTEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", out_dim=512):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.proj = nn.Linear(self.model.config.hidden_size, out_dim)

    @torch.no_grad()
    def encode_text(self, texts, device=None):
        toks = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=64
        )
        if device:
            toks = {k: v.to(device) for k, v in toks.items()}
        out = self.model(**toks).last_hidden_state[:,0]  # first token representation
        return self.proj(out)

    def forward(self, texts, device=None):
        return self.encode_text(texts, device=device)
