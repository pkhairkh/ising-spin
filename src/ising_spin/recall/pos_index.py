"""
POS-level n-gram recall index — the KEY INNOVATION of v17.

Instead of matching exact word sequences, this index matches POS TAG sequences.
This gives much longer effective context because POS n-grams are far more
regular than word n-grams.

Example:
  Word 5-gram ["the", "big", "brown", "dog", "chased"] is probably unique.
  POS  5-gram [DET, ADJ, ADJ, NOUN, VERB]   has been seen thousands of times.
  POS 15-gram (full clause pattern)           still has meaningful counts.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import NgramIndexBase
from ..vocabulary.pos import POSTypeSystem, N_POS, POS2IDX
from ..utils import TAG_PRIORITY, primary_pos_tag
from ..exceptions import ValidationError, IndexBuildError


class PosNgramIndex(NgramIndexBase):
    """
    POS-level n-gram index for abstract recall.

    Context keys are POS tag tuples, not word ID tuples.
    Continuations are still word IDs (we want to predict WORDS, not POS tags).

    Uses NgramIndexBase for all shared build/prune/energy logic.
    Only provides POS-specific context key formation.
    """

    def __init__(
        self,
        max_n: int = 15,
        min_count: int = 2,
        pos_system: Optional[POSTypeSystem] = None,
    ):
        super().__init__(
            max_n=max_n,
            min_count=min_count,
            higher_order_threshold=8,  # POS 8-gram+ uses stricter pruning
        )
        self.pos_system = pos_system

        # word_pos_tags: dict mapping word_id -> primary POS tag (int)
        self.word_pos_tags: Dict[int, int] = {}

    @property
    def _label(self) -> str:
        return "POS"

    def _context_to_key(self, context_words: List[int], k: int) -> tuple | None:
        """
        POS index: context key is the POS tag tuple derived from word IDs.

        Returns None if too many X (unknown) tags in context.
        """
        pos_context = tuple(
            self.word_pos_tags.get(w, POS2IDX["X"]) for w in context_words[-k:]
        )

        # Skip if too many unknown tags
        x_count = sum(1 for p in pos_context if p == POS2IDX["X"])
        if x_count > k // 2:
            return None

        return pos_context

    def _should_skip_continuation(self, word_id: int) -> bool:
        """Skip special tokens as continuations."""
        return word_id < 4

    # ── POS-specific build override ───────────────────────────────────────

    def build(
        self,
        sequences: List[List[int]],
        word_pos_tags: Optional[Dict[int, int]] = None,
        **kwargs,
    ) -> None:
        """
        Build POS n-gram index from training sequences.

        Args:
            sequences: Tokenized sequences.
            word_pos_tags: dict mapping word_id -> primary POS tag.
                          If not provided, derived from pos_system.

        Raises:
            IndexBuildError: if neither word_pos_tags nor pos_system is available.
        """
        if word_pos_tags is not None:
            self.word_pos_tags = word_pos_tags

        if not self.word_pos_tags:
            if self.pos_system is None:
                raise IndexBuildError(
                    "PosNgramIndex requires either word_pos_tags or pos_system"
                )
            for w, allowed in self.pos_system.allowed_types.items():
                if allowed:
                    self.word_pos_tags[w] = primary_pos_tag(allowed)

        super().build(sequences, **kwargs)

    def build_batched(
        self,
        sequences: List[List[int]],
        word_pos_tags: Optional[Dict[int, int]] = None,
        **kwargs,
    ) -> None:
        """Batched build with POS-specific setup."""
        if word_pos_tags is not None:
            self.word_pos_tags = word_pos_tags

        if not self.word_pos_tags:
            if self.pos_system is None:
                raise IndexBuildError(
                    "PosNgramIndex requires either word_pos_tags or pos_system"
                )
            for w, allowed in self.pos_system.allowed_types.items():
                if allowed:
                    self.word_pos_tags[w] = primary_pos_tag(allowed)

        super().build_batched(sequences, **kwargs)
