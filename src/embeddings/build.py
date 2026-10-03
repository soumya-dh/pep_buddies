"""
build.py - Populate the embedding cache for one model and one sequence kind.

Two kinds of sequence are embedded:

``peptides``
    The unique 9-mer peptides (5,633 of them across the whole dataset).

``hla``
    The unique **full 182-aa mature G-domain** sequences, one per allele (75).
    Deliberately *not* the 34-mer pseudosequence: that string is a
    non-contiguous concatenation of contact residues and is out-of-distribution
    nonsense to a protein language model trained on real chains. Embedding the
    real G-domain and then slicing the 34 contact positions out of the
    per-position output gives in-distribution input *and* pocket-specific
    features. ``EmbeddingCache.get_positions`` does the slicing; the 0-based
    indices are stored in the cache metadata as ``contact_positions_0based``.

Examples::

    python -m src.embeddings.build --model esm2-35m --kind peptides
    python -m src.embeddings.build --model esm2-35m --kind hla
    python -m src.embeddings.build --model esm2-35m --kind all --random-init
"""

import argparse
import logging
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.embeddings.cache import DEFAULT_ROOT, EmbeddingCache
from src.embeddings.encoders import MODEL_REGISTRY, get_encoder
from src.hla_database import MHC_I_PSEUDO_POSITIONS_1BASED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA = "data/processed/cleaned_stability_data.csv"

#: The 34 Nielsen contact positions as 0-based indices into the G-domain.
CONTACT_POSITIONS_0BASED: List[int] = [p - 1 for p in MHC_I_PSEUDO_POSITIONS_1BASED]


def unique_peptides(df: pd.DataFrame) -> List[str]:
    """Sorted unique peptides. Sorting makes the cache row order reproducible."""
    return sorted(df["peptide"].dropna().astype(str).unique().tolist())


def unique_hla_sequences(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Sorted unique G-domain sequences, with one representative allele each.

    Two alleles can share a sequence (``HLA-B*14:01(C67S)`` and
    ``HLA-B*14:02(C67S)`` share a pseudosequence here), so the mapping is
    sequence-keyed and the allele list is only for metadata.
    """
    pairs = (
        df[["allele", "hla_seq"]].dropna().astype(str)
        .drop_duplicates(subset=["hla_seq"])
        .sort_values("hla_seq")
    )
    return pairs["hla_seq"].tolist(), pairs["allele"].tolist()


def build(
    model_key: str,
    kind: str,
    data_csv: str = DEFAULT_DATA,
    root: str = DEFAULT_ROOT,
    batch_size: int = 32,
    random_init: bool = False,
    overwrite: bool = False,
) -> EmbeddingCache:
    """Embed one kind of sequence with one model and write it to the cache."""
    cache = EmbeddingCache(model_key, kind, root=root)
    if cache.exists() and not overwrite:
        logger.info(
            "%s/%s already cached (%s); pass --overwrite to rebuild",
            model_key, kind, cache.meta.get("shape"),
        )
        return cache

    df = pd.read_csv(data_csv)

    extra = {"random_init": random_init, "source_csv": os.path.basename(data_csv)}
    if kind == "peptides":
        seqs = unique_peptides(df)
    elif kind == "hla":
        seqs, alleles = unique_hla_sequences(df)
        extra["alleles"] = alleles
        extra["contact_positions_0based"] = CONTACT_POSITIONS_0BASED
    else:
        raise ValueError(f"kind must be 'peptides' or 'hla', got {kind!r}")

    logger.info(
        "Embedding %d unique %s sequences (length %d) with %s%s",
        len(seqs), kind, len(seqs[0]), model_key,
        " [RANDOM WEIGHTS -- pipeline validation only]" if random_init else "",
    )

    enc = get_encoder(model_key, random_init=random_init)
    emb = enc.embed(seqs, batch_size=batch_size)

    if kind == "hla" and max(CONTACT_POSITIONS_0BASED) >= emb.shape[1]:
        raise ValueError(
            f"contact position {max(CONTACT_POSITIONS_0BASED)} exceeds G-domain "
            f"embedding length {emb.shape[1]}"
        )

    cache.write(seqs, emb, extra_meta=extra)
    return cache


def parse_args():
    p = argparse.ArgumentParser(description="Build the per-position embedding cache")
    p.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    p.add_argument("--kind", default="all", choices=["peptides", "hla", "all"])
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--random-init", action="store_true",
        help="Random weights instead of pretrained. Validates the pipeline "
             "offline; results are NOT meaningful."
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    kinds = ["peptides", "hla"] if args.kind == "all" else [args.kind]
    for kind in kinds:
        cache = build(
            args.model, kind,
            data_csv=args.data, root=args.root,
            batch_size=args.batch_size,
            random_init=args.random_init,
            overwrite=args.overwrite,
        )
        logger.info("%s/%s -> %s", args.model, kind, cache.meta.get("shape"))


if __name__ == "__main__":
    main()
