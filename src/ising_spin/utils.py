"""
Shared utilities for the Ising Spin Glass Language Model.

Single source of truth for constants and helpers that were previously
duplicated across multiple modules:
  - TAG_PRIORITY: word→POS disambiguation priority (was copy-pasted 5 times)
  - get_rss_mb(): process memory measurement (was copy-pasted 4 times)
  - validate_array(): common input validation for numpy arrays
  - validate_nonempty(): common validation for sequences
"""

from __future__ import annotations

import os
from typing import List, Sequence

import numpy as np

from .vocabulary.pos import POS2IDX


# ── TAG PRIORITY ─────────────────────────────────────────────────────────
# Lower value = higher priority when a word has multiple possible POS tags.
# Closed-class tags (DET, PRON, AUX) are prioritized because they are
# unambiguous: if a word CAN be a determiner, it IS a determiner.

TAG_PRIORITY: dict[int, int] = {
    POS2IDX["PUNCT"]: 0,
    POS2IDX["DET"]: 1,
    POS2IDX["PRON"]: 2,
    POS2IDX["AUX"]: 3,
    POS2IDX["CONJ"]: 4,
    POS2IDX["PART"]: 5,
    POS2IDX["PREP"]: 6,
    POS2IDX["NUM"]: 7,
    POS2IDX["ADV"]: 8,
    POS2IDX["ADJ"]: 9,
    POS2IDX["NOUN"]: 10,
    POS2IDX["VERB"]: 11,
    POS2IDX["X"]: 12,
}


def primary_pos_tag(allowed_tags: set[int] | frozenset[int]) -> int:
    """
    Select the primary (highest-priority) POS tag from a set of allowed tags.

    Uses TAG_PRIORITY to disambiguate: closed-class tags win over open-class.
    Returns POS2IDX["X"] for empty sets.
    """
    if not allowed_tags:
        return POS2IDX["X"]
    return min(allowed_tags, key=lambda t: TAG_PRIORITY.get(t, 99))


# ── MEMORY MEASUREMENT ───────────────────────────────────────────────────

def get_rss_mb() -> int:
    """
    Get current process RSS in MB. Returns 0 if unavailable.

    Tries /proc first (Linux), falls back to resource module (macOS/BSD).
    """
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024  # KB → MB
    except (OSError, ValueError):
        pass
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    except (ImportError, AttributeError):
        return 0


# ── INPUT VALIDATION ─────────────────────────────────────────────────────

def validate_array(
    arr: np.ndarray,
    name: str = "array",
    *,
    dtype: type | None = None,
    ndim: int | None = None,
    min_len: int = 0,
) -> None:
    """
    Validate a numpy array meets requirements.

    Raises:
        TypeError: if arr is not an ndarray or dtype/ndim mismatch.
        ValueError: if array is too short.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy ndarray, got {type(arr).__name__}")
    if dtype is not None and arr.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {arr.dtype}")
    if ndim is not None and arr.ndim != ndim:
        raise TypeError(f"{name} must have {ndim} dimensions, got {arr.ndim}")
    if len(arr) < min_len:
        raise ValueError(f"{name} must have length >= {min_len}, got {len(arr)}")


def validate_nonempty(
    seq: Sequence | List,
    name: str = "sequence",
) -> None:
    """Raise ValueError if sequence is empty."""
    if not seq:
        raise ValueError(f"{name} must not be empty")


def validate_positive(value: int, name: str = "value") -> None:
    """Raise ValueError if value is not positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_non_negative(value: int, name: str = "value") -> None:
    """Raise ValueError if value is negative."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
