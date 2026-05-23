"""
Shared utilities — deduplicated helpers used across modules.

Previously these were copy-pasted into model.py, model_v17.py,
model_v18.py, and train_v18.py.  Now there is a single source of truth.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from .vocabulary.pos import POS2IDX, N_POS


# ===========================================================================
# POS TAG PRIORITY (used by model, generator, training)
# ===========================================================================

TAG_PRIORITY: Dict[int, int] = {
    POS2IDX["PUNCT"]: 0, POS2IDX["DET"]: 1, POS2IDX["PRON"]: 2,
    POS2IDX["AUX"]: 3, POS2IDX["CONJ"]: 4, POS2IDX["PART"]: 5,
    POS2IDX["PREP"]: 6, POS2IDX["NUM"]: 7, POS2IDX["ADV"]: 8,
    POS2IDX["ADJ"]: 9, POS2IDX["NOUN"]: 10, POS2IDX["VERB"]: 11,
    POS2IDX["X"]: 12,
}


def get_primary_pos(allowed_types: set[int]) -> int:
    """Return the primary (most-specific) POS tag from an allowed set."""
    if not allowed_types:
        return POS2IDX["X"]
    return min(allowed_types, key=lambda t: TAG_PRIORITY.get(t, 99))


# ===========================================================================
# RSS MEMORY MONITORING
# ===========================================================================

def get_rss_mb() -> int:
    """Get current process RSS in MB (0 if unavailable)."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024  # KB -> MB
    except (OSError, ValueError):
        pass
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    except (ImportError, OSError):
        return 0


# ===========================================================================
# CORPUS LOADING
# ===========================================================================

def load_fineweb_edu(
    n_samples: int = 50000,
    split: str = "train",
    subset: str = "sample-10BT",
    min_length: int = 20,
    max_length: int = 2000,
) -> List[str]:
    """Load text samples from the fineweb-edu dataset on HuggingFace."""
    from datasets import load_dataset
    from .errors import CorpusError

    print(f"Loading fineweb-edu ({subset}, split={split})...")

    dataset = None
    for name in ["HuggingFaceFW/fineweb-edu", "HuggingFW/fineweb-edu"]:
        try:
            dataset = load_dataset(name, name=subset, split=split, streaming=True)
            print(f"  Loaded from '{name}' with subset '{subset}'")
            break
        except Exception:
            continue

    if dataset is None:
        for name in ["HuggingFaceFW/fineweb-edu", "HuggingFW/fineweb-edu"]:
            try:
                dataset = load_dataset(name, split=split, streaming=True)
                print(f"  Loaded from '{name}' without subset")
                break
            except Exception:
                continue

    if dataset is None:
        raise CorpusError(
            "Could not load fineweb-edu. Check internet and HuggingFace access."
        )

    texts: List[str] = []
    scanned = 0
    for example in dataset:
        scanned += 1
        if len(texts) >= n_samples:
            break
        text = example.get("text", "").strip()
        if min_length <= len(text) <= max_length:
            texts.append(text)
        if scanned % 10000 == 0:
            print(f"  Scanned {scanned} examples, collected {len(texts)} texts...")
        if scanned > n_samples * 5:
            break

    print(f"Loaded {len(texts)} texts from fineweb-edu (scanned {scanned}).")
    return texts


# ===========================================================================
# TEXT / SEQUENCE UTILITIES
# ===========================================================================

def tokenize_texts(texts: List[str], vocab) -> List[List[int]]:
    """Tokenize a list of texts using the vocabulary."""
    sequences: List[List[int]] = []
    for text in texts:
        tokens = vocab.encode(text)
        if tokens:
            sequences.append(tokens)
    return sequences


def truncate_sequences(
    sequences: List[List[int]], max_len: int = 30, min_len: int = 2,
) -> List[List[int]]:
    """Truncate sequences to max_len and filter short ones."""
    return [seq[:max_len] for seq in sequences if len(seq) > min_len]
