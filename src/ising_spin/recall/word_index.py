"""
Word-level n-gram recall index.

Maps exact word context tuples to continuation word distributions.
This is the PRIMARY generation mechanism — when it hits, it produces
coherent text. When it misses, POS and topic indexes take over.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import NgramIndexBase
from ..exceptions import ValidationError


class WordNgramIndex(NgramIndexBase):
    """
    Multi-level word n-gram index for exact token recall.

    Uses NgramIndexBase for all shared build/prune/energy logic.
    Only provides word-specific context key formation (the context
    IS the word IDs directly).
    """

    def __init__(self, max_n: int = 5, min_count: int = 1):
        super().__init__(
            max_n=max_n,
            min_count=min_count,
            higher_order_threshold=4,  # 4-gram and above use stricter pruning
        )

    @property
    def _label(self) -> str:
        return "WORD"

    def _context_to_key(self, context_words: List[int], k: int) -> tuple | None:
        """
        Word index: context key IS the word ID tuple directly.

        Returns None if any context word is a special token (< 4).
        """
        context = tuple(context_words[-k:]) if len(context_words) >= k else None
        if context is None:
            return None
        # Skip contexts containing special tokens
        if any(w < 4 for w in context):
            return None
        return context

    def _should_skip_continuation(self, word_id: int) -> bool:
        """Skip special tokens (word_id < 4) as continuations."""
        return word_id < 4

    # ── Word-specific convenience methods ─────────────────────────────────

    def get_best_copy_candidate(
        self,
        context_words: List[int],
        min_context_length: int = 3,
        min_confidence: float = 0.3,
    ) -> Optional[Tuple[int, int, int]]:
        """
        Find best word for direct copying (highest-confidence n-gram match).

        The copy mechanism looks for n-gram matches with context length >=
        min_context_length and confidence >= min_confidence. If found,
        the word is used directly instead of Boltzmann sampling.
        """
        matches = self.lookup(context_words)
        for k in sorted(matches.keys(), reverse=True):
            if k < min_context_length:
                break
            continuations = matches[k]
            if not continuations:
                continue
            best_word, best_count, total = continuations[0]
            if best_count * 10 >= total * int(min_confidence * 10):
                return (best_word, best_count, total)
        return None
