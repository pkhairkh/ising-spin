"""
Topic-level n-gram recall index for discourse-level coherence.

Context = sequence of topic assignments for context words.
With only ~16 topics, even 10-grams are well-populated, enabling
discourse-level recall that word/POS n-grams miss entirely.

Example:
  [SCIENCE, SCIENCE, SCIENCE, ...] (10 consecutive topic IDs) -> technical vocabulary
  This captures discourse-level coherence: words from the same topic cluster
  together, and the topic context strongly predicts which words follow.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import NgramIndexBase
from ..exceptions import IndexBuildError, ValidationError


class TopicNgramIndex(NgramIndexBase):
    """
    Topic-level n-gram index for discourse-level recall.

    Context keys are topic ID tuples, not word ID tuples.
    Continuations are still word IDs.

    Uses NgramIndexBase for all shared build/prune/energy logic.
    Only provides topic-specific context key formation.
    """

    def __init__(
        self,
        max_n: int = 10,
        min_count: int = 3,
        n_topics: int = 16,
        word_topics: Optional[np.ndarray] = None,
    ):
        """
        Args:
            max_n: Maximum topic n-gram length (default 10).
            min_count: Minimum count for a continuation to be kept.
            n_topics: Number of topics (default 16).
            word_topics: (vocab_size,) int8 array from TopicAssigner.
        """
        super().__init__(
            max_n=max_n,
            min_count=min_count,
            higher_order_threshold=6,  # Topic 6-gram+ uses stricter pruning
        )
        self.n_topics = n_topics
        self.word_topics = word_topics

    @property
    def _label(self) -> str:
        return "TOPIC"

    def _context_to_key(self, context_words: List[int], k: int) -> tuple | None:
        """
        Topic index: context key is the topic ID tuple derived from word IDs.
        """
        return tuple(
            self._word_to_topic(w) for w in context_words[-k:]
        )

    def _should_skip_continuation(self, word_id: int) -> bool:
        """Skip special tokens as continuations."""
        return word_id < 4

    def _word_to_topic(self, word_id: int) -> int:
        """Convert a word ID to its topic ID. Unknown words get topic 0."""
        if self.word_topics is not None and word_id < len(self.word_topics):
            return int(self.word_topics[word_id])
        return 0

    # ── Topic-specific build override ─────────────────────────────────────

    def build(
        self,
        sequences: List[List[int]],
        word_topics: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """
        Build topic n-gram index from training sequences.

        Args:
            sequences: Tokenized sequences.
            word_topics: (vocab_size,) int8 array from TopicAssigner.

        Raises:
            IndexBuildError: if word_topics is not provided.
        """
        if word_topics is not None:
            self.word_topics = word_topics

        if self.word_topics is None:
            raise IndexBuildError(
                "TopicNgramIndex requires word_topics (from TopicAssigner). "
                "Provide via __init__ or build()."
            )

        super().build(sequences, **kwargs)

    def build_batched(
        self,
        sequences: List[List[int]],
        word_topics: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """Batched build with topic-specific validation."""
        if word_topics is not None:
            self.word_topics = word_topics

        if self.word_topics is None:
            raise IndexBuildError(
                "TopicNgramIndex requires word_topics (from TopicAssigner). "
                "Provide via __init__ or build_batched()."
            )

        super().build_batched(sequences, **kwargs)
