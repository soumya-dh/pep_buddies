"""
cache.py - On-disk cache of per-position foundation-model embeddings.

Design notes
------------
* **Keyed by unique sequence, not by dataset row.** The dataset has 28,166 rows
  but only 5,633 unique peptides and 75 unique HLA G-domain sequences, so
  embedding rows would waste ~5x the compute and ~5x the disk.
* **Per-position, not pooled.** Requirement 5 of Phase 2. Mean-pooling a 9-mer
  destroys exactly the anchor-residue signal (P2/P9) that drives stability, and
  the HLA pocket positions can only be sliced out of a per-position tensor.
* **float16.** Halves disk for no measurable loss on downstream ridge/MLP heads.
  ESM-2 650M peptides: 5,633 x 9 x 1280 x 2B ~= 130 MB; HLA: 75 x 182 x 1280 x 2B
  ~= 35 MB.
* Each (model, kind) pair gets its own fixed-length store, because peptides are
  length 9 and HLA G-domains are length 182.

Layout, under ``data/embeddings/{model_slug}/``::

    {kind}.npy          float16 memmap, shape (n_sequences, length, dim)
    {kind}.index.json   {sequence: row} plus metadata
"""

import json
import logging
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ROOT = "data/embeddings"


def model_slug(model_key: str) -> str:
    """Filesystem-safe directory name for a model key."""
    return model_key.replace("/", "__").replace(":", "_")


class EmbeddingCache:
    """
    Fixed-length per-position embedding store for one (model, kind) pair.

    ``kind`` is a free-form label such as ``"peptides"`` or ``"hla"``; it only
    needs to be stable, since it names the files.
    """

    def __init__(
        self,
        model_key: str,
        kind: str,
        root: str = DEFAULT_ROOT,
    ):
        self.model_key = model_key
        self.kind = kind
        self.dir = os.path.join(root, model_slug(model_key))
        self.array_path = os.path.join(self.dir, f"{kind}.npy")
        self.index_path = os.path.join(self.dir, f"{kind}.index.json")
        self._index: Optional[Dict[str, int]] = None
        self._meta: Dict[str, object] = {}
        self._array: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #

    def exists(self) -> bool:
        return os.path.exists(self.array_path) and os.path.exists(self.index_path)

    @property
    def index(self) -> Dict[str, int]:
        if self._index is None:
            if not self.exists():
                raise FileNotFoundError(
                    f"no cache at {self.array_path}; build it with "
                    f"`python -m src.embeddings.build --model {self.model_key} "
                    f"--kind {self.kind}`"
                )
            with open(self.index_path) as f:
                payload = json.load(f)
            self._index = payload["sequences"]
            self._meta = payload.get("meta", {})
        return self._index

    @property
    def meta(self) -> Dict[str, object]:
        self.index  # force load
        return self._meta

    @property
    def array(self) -> np.ndarray:
        """Read-only memmap of shape (n_sequences, length, dim)."""
        if self._array is None:
            shape = tuple(self.meta["shape"])  # type: ignore[index]
            self._array = np.memmap(
                self.array_path, dtype=np.float16, mode="r", shape=shape
            )
        return self._array

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #

    def write(
        self,
        sequences: Sequence[str],
        embeddings: np.ndarray,
        extra_meta: Optional[Dict[str, object]] = None,
    ) -> None:
        """
        Persist embeddings for ``sequences``.

        ``embeddings`` must be (n_sequences, length, dim) and aligned with
        ``sequences``. Sequences must be unique.
        """
        if len(sequences) != len(embeddings):
            raise ValueError(
                f"{len(sequences)} sequences vs {len(embeddings)} embedding rows"
            )
        if len(set(sequences)) != len(sequences):
            raise ValueError("sequences must be unique")
        if embeddings.ndim != 3:
            raise ValueError(f"expected (n, length, dim), got {embeddings.shape}")

        os.makedirs(self.dir, exist_ok=True)
        arr = np.asarray(embeddings, dtype=np.float16)
        mm = np.memmap(self.array_path, dtype=np.float16, mode="w+", shape=arr.shape)
        mm[:] = arr
        mm.flush()
        del mm

        meta: Dict[str, object] = {
            "model_key": self.model_key,
            "kind": self.kind,
            "shape": list(arr.shape),
            "dtype": "float16",
            "n_sequences": int(arr.shape[0]),
            "length": int(arr.shape[1]),
            "dim": int(arr.shape[2]),
        }
        if extra_meta:
            meta.update(extra_meta)

        with open(self.index_path, "w") as f:
            json.dump(
                {"meta": meta, "sequences": {s: i for i, s in enumerate(sequences)}},
                f,
            )

        self._index = None
        self._array = None
        logger.info(
            "Wrote %s embeddings: %s (%.1f MB)",
            self.kind, arr.shape, arr.nbytes / 1e6,
        )

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #

    def get(self, sequences: Sequence[str]) -> np.ndarray:
        """
        Per-position embeddings for ``sequences``, as float32 (n, length, dim).

        Sequences may repeat; this is how a 28k-row dataset is served from a
        5.6k-row store.
        """
        idx = self.index
        missing = [s for s in dict.fromkeys(sequences) if s not in idx]
        if missing:
            raise KeyError(
                f"{len(missing)} sequence(s) not in the {self.kind} cache for "
                f"{self.model_key}, e.g. {missing[:3]}. Rebuild the cache."
            )
        rows = np.fromiter((idx[s] for s in sequences), dtype=np.int64, count=len(sequences))
        return np.asarray(self.array[rows], dtype=np.float32)

    def get_mean(self, sequences: Sequence[str]) -> np.ndarray:
        """Mean-pooled embeddings, (n, dim). The weakest featurisation; a baseline."""
        return self.get(sequences).mean(axis=1)

    def get_flat(self, sequences: Sequence[str]) -> np.ndarray:
        """Flattened per-position embeddings, (n, length * dim)."""
        emb = self.get(sequences)
        return emb.reshape(emb.shape[0], -1)

    def get_positions(
        self,
        sequences: Sequence[str],
        positions: Sequence[int],
    ) -> np.ndarray:
        """
        Embeddings at selected 0-based positions, (n, len(positions), dim).

        Used to slice the 34 pocket contact residues out of the full 182-aa
        G-domain embedding -- in-distribution input to the model, pocket-specific
        features out.
        """
        emb = self.get(sequences)
        pos = np.asarray(positions, dtype=np.int64)
        if pos.min() < 0 or pos.max() >= emb.shape[1]:
            raise IndexError(
                f"positions must be within [0, {emb.shape[1]}); "
                f"got [{pos.min()}, {pos.max()}]"
            )
        return emb[:, pos, :]
