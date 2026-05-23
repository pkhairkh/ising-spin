"""
Abstract base class for all recall indexes.

Each recall index maps contexts (at some abstraction level) to continuation
word distributions. At inference time, the index computes an energy for each
candidate word: LOWER energy = more likely continuation.

All computation uses integer-only arithmetic. Log2 is computed via
int_log2_fine() from the Boltzmann sampler module.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np


class AbstractRecallIndex(ABC):
    """
    Base class for all recall indexes.

    Each index maps contexts (word IDs, POS tags, topic IDs, etc.) to
    continuation word distributions. The key method is compute_energy(),
    which returns a per-candidate energy array where LOWER = more likely.

    Subclasses must implement:
      - build():        construct the index from training sequences
      - compute_energy(): compute recall energy for candidate words
      - lookup():       look up continuations for a given context
    """

    @abstractmethod
    def build(self, sequences: List[List[int]], **kwargs) -> None:
        """
        Build the index from training sequences.

        Args:
            sequences: List of tokenized sequences (list of word ID lists).
            **kwargs:  Subclass-specific parameters.
        """
        pass

    @abstractmethod
    def compute_energy(
        self,
        context_ids: List[int],
        candidate_words: np.ndarray,
        recall_scale: int = 100,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute recall energy for candidate words. LOWER energy = more likely.

        In the Boltzmann model P(w) ~ exp(-beta * E(w)), recall energy encodes
        -log2(P_ngram(w)) * scale. Words that match the index get low energy;
        unmatched words get a backoff energy (moderate) or max energy (high).

        Args:
            context_ids:    Context word IDs (may be converted internally to
                            POS tags or topic IDs by subclass).
            candidate_words: Array of candidate word IDs to score.
            recall_scale:   Energy scale factor (multiplier for log2 ratio).

        Returns:
            np.ndarray of int64 energies, shape (len(candidate_words),).
        """
        pass

    @abstractmethod
    def lookup(self, context_ids: List[int]) -> Dict:
        """
        Look up continuations for a given context.

        Args:
            context_ids: Context identifiers (word IDs, POS tags, etc.)

        Returns:
            Dict mapping n-gram level k to list of (word, count, total) tuples.
        """
        pass
