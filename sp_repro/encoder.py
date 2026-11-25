import hashlib
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from .data import MAX_RESIDUES, Protein, model_input


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class FrozenEncoder:
    def __init__(self, path: Path, device: torch.device):
        if not (path / "model.safetensors").is_file():
            raise ValueError(f"encoder directory lacks model.safetensors: {path}")
        self.device = device
        self.fingerprint = {
            name: sha256(path / name)
            for name in ("model.safetensors", "config.json", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json")
        }
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            path, local_files_only=True, use_safetensors=True, add_pooling_layer=False
        ).eval().requires_grad_(False).to(device)
        self.width = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, proteins: list[Protein]) -> tuple[torch.Tensor, torch.Tensor]:
        sequences = [model_input(protein.sequence) for protein in proteins]
        tokens = self.tokenizer(sequences, padding=True, return_tensors="pt")
        lengths = torch.tensor([len(sequence) for sequence in sequences])
        if not torch.equal(tokens["attention_mask"].sum(1), lengths + 2):
            raise ValueError("expected one encoder token per residue and two special tokens")
        encoded = self.model(**tokens.to(self.device)).last_hidden_state[:, 1:-1]
        padded = torch.zeros((len(proteins), MAX_RESIDUES, self.width), dtype=torch.float16)
        padded[:, :encoded.shape[1]] = encoded.cpu()
        return padded, lengths


def device_for(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)
