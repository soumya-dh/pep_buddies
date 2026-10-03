"""
encoders.py - Uniform wrappers around frozen protein language models.

Every encoder exposes the same contract::

    enc = get_encoder("esm2-35m")
    emb = enc.embed(["VTTEVAFGL", "FVRQCFNPM"])   # -> (2, 9, dim) float32

The returned tensor is strictly per-residue: special tokens (BOS/EOS/pad) are
stripped, so ``emb.shape[1] == len(sequence)``. All sequences in one call must be
the same length, which holds for this dataset (peptides are all 9-mers; HLA
G-domains are all 182-mers).

Models are used **frozen** -- no fine-tuning. On a CPU-only container that is the
only tractable option, and it keeps the comparison against one-hot/BLOSUM
baselines about representation quality rather than training budget.

``random_init=True`` builds the correct architecture with random weights and
downloads nothing but the config. It exists to validate the cache/head pipeline
end-to-end without pretrained weights; **numbers from a random-init encoder are
meaningless as results** and are tagged as such in the cache metadata.
"""

import os
import logging
from typing import Dict, List, Optional, Sequence

# Default to offline if hub cache is populated to avoid timeout retries in sandboxed environments
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

logger = logging.getLogger(__name__)

#: Short key -> Hugging Face repo id.
MODEL_REGISTRY: Dict[str, Dict[str, object]] = {
    "esm2-8m":   {"hf": "facebook/esm2_t6_8M_UR50D",    "family": "esm2", "dim": 320},
    "esm2-35m":  {"hf": "facebook/esm2_t12_35M_UR50D",  "family": "esm2", "dim": 480},
    "esm2-150m": {"hf": "facebook/esm2_t30_150M_UR50D", "family": "esm2", "dim": 640},
    "esm2-650m": {"hf": "facebook/esm2_t33_650M_UR50D", "family": "esm2", "dim": 1280},
    # Second family. Ankh-base is ~750 MB versus ProtT5-XL's ~11 GB, which
    # matters on a CPU-only container with limited disk.
    "ankh-base": {"hf": "ElnaggarLab/ankh-base",        "family": "t5",   "dim": 768},
}


class SequenceEncoder:
    """Base class: frozen model -> per-residue embeddings."""

    family = "base"

    def __init__(self, model_key: str, random_init: bool = False, device: str = "cpu"):
        if model_key not in MODEL_REGISTRY:
            raise KeyError(
                f"unknown model '{model_key}'; known: {sorted(MODEL_REGISTRY)}"
            )
        self.model_key = model_key
        self.hf_name = str(MODEL_REGISTRY[model_key]["hf"])
        self.random_init = random_init
        self.device = device
        self._tok = None
        self._model = None

    # -- to be provided by subclasses ---------------------------------- #

    def _load(self) -> None:
        raise NotImplementedError

    def _encode_batch(self, seqs: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    # ------------------------------------------------------------------ #

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        return int(self._model.config.hidden_size)

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._load()
            self._model.eval()
            n = sum(p.numel() for p in self._model.parameters())
            logger.info(
                "Loaded %s (%s%s): %.1fM params, hidden=%d",
                self.model_key, self.hf_name,
                ", RANDOM WEIGHTS" if self.random_init else "",
                n / 1e6, self._model.config.hidden_size,
            )

    def embed(
        self,
        sequences: Sequence[str],
        batch_size: int = 32,
        log_every: int = 20,
    ) -> np.ndarray:
        """
        Per-residue embeddings for ``sequences``, shape (n, length, dim).

        All sequences must share a length, so the output is a dense tensor with
        no padding to track.
        """
        if not sequences:
            raise ValueError("no sequences given")
        lengths = {len(s) for s in sequences}
        if len(lengths) != 1:
            raise ValueError(
                f"all sequences must be the same length; got {sorted(lengths)[:5]}"
            )

        self._ensure_loaded()
        out: List[np.ndarray] = []
        n_batches = (len(sequences) + batch_size - 1) // batch_size

        with torch.no_grad():
            for bi, start in enumerate(range(0, len(sequences), batch_size)):
                batch = list(sequences[start:start + batch_size])
                emb = self._encode_batch(batch)
                expected = (len(batch), len(batch[0]))
                if emb.shape[:2] != expected:
                    raise RuntimeError(
                        f"{self.model_key} returned {emb.shape[:2]}, expected "
                        f"{expected} -- special-token stripping is wrong"
                    )
                out.append(emb)
                if log_every and (bi % log_every == 0 or bi == n_batches - 1):
                    logger.info("  %s: batch %d/%d", self.model_key, bi + 1, n_batches)

        return np.concatenate(out, axis=0)


class Esm2Encoder(SequenceEncoder):
    """
    ESM-2, via Hugging Face ``AutoModel``.

    The ESM-2 tokenizer wraps each sequence in ``<cls>`` and ``<eos>``, so the
    raw output is length+2; both are dropped here.
    """

    family = "esm2"

    def _load(self) -> None:
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        local_only = (os.environ.get("TRANSFORMERS_OFFLINE") == "1") or (os.environ.get("HF_HUB_OFFLINE") == "1")
        self._tok = AutoTokenizer.from_pretrained(self.hf_name, local_files_only=local_only)

        # add_pooling_layer=False: we read last_hidden_state, never pooler_output.
        if self.random_init:
            cfg = AutoConfig.from_pretrained(self.hf_name, local_files_only=local_only)
            self._model = AutoModel.from_config(cfg, add_pooling_layer=False)
        else:
            self._model = AutoModel.from_pretrained(self.hf_name, add_pooling_layer=False, local_files_only=local_only)
        self._model.to(self.device)

    def _encode_batch(self, seqs: Sequence[str]) -> np.ndarray:
        enc = self._tok(list(seqs), return_tensors="pt", add_special_tokens=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        hidden = self._model(**enc).last_hidden_state  # (B, L+2, D)
        return hidden[:, 1:-1, :].cpu().numpy().astype(np.float32)


class T5Encoder(SequenceEncoder):
    """
    Encoder-only use of a T5-family protein model (Ankh, ProtT5).

    These tokenizers expect residues and append a single ``</s>``, which is
    dropped here. ProtT5 additionally wants space-separated residues; Ankh does
    not. ``spaced`` covers both.
    """

    family = "t5"

    def __init__(self, *args, spaced: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.spaced = spaced

    def _load(self) -> None:
        from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

        self._tok = AutoTokenizer.from_pretrained(self.hf_name)
        if self.random_init:
            self._model = T5EncoderModel(AutoConfig.from_pretrained(self.hf_name))
        else:
            self._model = T5EncoderModel.from_pretrained(self.hf_name)
        self._model.to(self.device)

    def _encode_batch(self, seqs: Sequence[str]) -> np.ndarray:
        prepared = [" ".join(s) for s in seqs] if self.spaced else list(seqs)
        enc = self._tok(prepared, return_tensors="pt", add_special_tokens=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        hidden = self._model(**enc).last_hidden_state  # (B, L+1, D)
        return hidden[:, :len(seqs[0]), :].cpu().numpy().astype(np.float32)


def get_encoder(
    model_key: str,
    random_init: bool = False,
    device: str = "cpu",
) -> SequenceEncoder:
    """Build the encoder registered under ``model_key``."""
    family = MODEL_REGISTRY[model_key]["family"] if model_key in MODEL_REGISTRY else None
    if family == "esm2":
        return Esm2Encoder(model_key, random_init=random_init, device=device)
    if family == "t5":
        spaced = "prot_t5" in str(MODEL_REGISTRY[model_key]["hf"]).lower()
        return T5Encoder(model_key, random_init=random_init, device=device, spaced=spaced)
    raise KeyError(f"unknown model '{model_key}'; known: {sorted(MODEL_REGISTRY)}")
