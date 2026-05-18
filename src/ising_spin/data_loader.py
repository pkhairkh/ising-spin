"""
Data loading utilities for the Ising Spin Language Model.

Downloads and processes fineweb-edu from HuggingFace.
All counting is integer-only.
"""

from typing import List, Optional
from datasets import load_dataset


def load_fineweb_edu(
    n_samples: int = 50000,
    split: str = "train",
    subset: str = "sample-10BT",
    min_length: int = 20,
    max_length: int = 2000,
    seed: int = 42,
) -> List[str]:
    """
    Load text samples from the fineweb-edu dataset on HuggingFace.

    Args:
        n_samples: Number of text samples to load.
        split: Dataset split to use.
        subset: Dataset subset/configuration.
        min_length: Minimum character length for a sample.
        max_length: Maximum character length for a sample.
        seed: Random seed for shuffling.

    Returns:
        List of text strings.
    """
    print(f"Loading fineweb-edu ({subset}, split={split})...")

    dataset_names = [
        "HuggingFaceFW/fineweb-edu",
        "HuggingFW/fineweb-edu",
    ]

    dataset = None
    for name in dataset_names:
        try:
            dataset = load_dataset(
                name,
                name=subset,
                split=split,
                streaming=True,
            )
            print(f"  Loaded from '{name}' with subset '{subset}'")
            break
        except Exception as e:
            print(f"  Could not load '{name}' subset '{subset}': {e}")
            continue

    if dataset is None:
        for name in dataset_names:
            try:
                dataset = load_dataset(
                    name,
                    split=split,
                    streaming=True,
                )
                print(f"  Loaded from '{name}' without subset")
                break
            except Exception as e:
                print(f"  Could not load '{name}': {e}")
                continue

    if dataset is None:
        raise RuntimeError(
            "Could not load fineweb-edu. Please check your internet connection "
            "and HuggingFace access."
        )

    texts = []
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

        # Safety: don't scan forever
        if scanned > n_samples * 5:
            print(f"  Stopping after scanning {scanned} examples")
            break

    print(f"Loaded {len(texts)} texts from fineweb-edu (scanned {scanned}).")
    return texts


def tokenize_texts(texts: List[str], vocab) -> List[List[int]]:
    """
    Tokenize a list of texts using the vocabulary.
    Pure integer encoding - no FP.
    """
    sequences = []
    for text in texts:
        tokens = vocab.encode(text)
        if len(tokens) > 0:
            sequences.append(tokens)
    return sequences


def truncate_sequences(
    sequences: List[List[int]],
    max_len: int = 50,
) -> List[List[int]]:
    """
    Truncate sequences to max_len and filter empty ones.
    Pure integer operation.
    """
    result = []
    for seq in sequences:
        if len(seq) > 3:  # skip very short sequences
            result.append(seq[:max_len])
    return result
