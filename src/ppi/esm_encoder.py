import torch
from typing import List, Tuple, Optional

import esm


class ESMEncoder:
    """
    Wrapper for loading ESM pretrained models and extracting per-token features.

    Uses fair-esm API (`esm.pretrained`) and the alphabet batch converter.
    """

    def __init__(self, model_name: str = "esm2_t33_650M_UR50D", device: Optional[torch.device] = None, max_len: int = 1024):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.alphabet = getattr(esm.pretrained, model_name)()
        self.model.eval()
        self.model.to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.padding_idx = self.alphabet.padding_idx
        self.eos_idx = self.alphabet.eos_idx

        # Hidden size depends on model; for esm2_t33_650M_UR50D it is 1280
        # Access via model.embed_dim for ESM2
        self.hidden_dim = getattr(self.model, "embed_dim", None)
        self.repr_layer = getattr(self.model, "num_layers", 33)
        self.max_len = max_len

    @torch.no_grad()
    def encode_sequences(self, seqs: List[str], repr_layer: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert sequences to per-token features and an attention mask.

        Returns:
            features: [batch, seq_len, hidden_dim]
            attention_mask: [batch, seq_len] with 1 for real AA tokens
        """
        # Truncate sequences to model maximum length
        trunc = [s[: self.max_len] for s in seqs]
        data = [(f"seq{i}", s) for i, s in enumerate(trunc)]
        labels, strs, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)

        layer = self.repr_layer if repr_layer is None else repr_layer
        out = self.model(tokens, repr_layers=[layer], return_contacts=False)
        feats = out["representations"][layer]

        # Build mask: non-padding and not BOS/EOS
        mask = (tokens != self.padding_idx).to(torch.bool)
        mask[:, 0] = False
        mask = mask & (tokens != self.eos_idx)

        return feats, mask

    @torch.no_grad()
    def encode_pair_batch(
        self,
        seqs_a: List[str],
        seqs_b: List[str],
        repr_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode two lists of sequences into per-token features and masks.
        """
        feats_a, mask_a = self.encode_sequences(seqs_a, repr_layer)
        feats_b, mask_b = self.encode_sequences(seqs_b, repr_layer)
        return feats_a, mask_a, feats_b, mask_b
