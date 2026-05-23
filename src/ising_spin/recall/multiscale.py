"""
Multi-scale recall: combines word, POS, and topic indexes via product-of-experts.

This is the v17 equivalent of what recall alone did in v1-v16.
Each scale independently computes energies, then they're combined
as a product of experts (take the BEST/lowest energy per word).

The three scales cover different context ranges:
  - Word:  1-5 tokens of exact lexical context (precise but sparse)
  - POS:   1-15 tokens of syntactic context (less sparse, captures grammar)
  - Topic: 1-10 tokens of discourse context (very dense, captures coherence)

Product of experts means: for each candidate word, we take the MINIMUM energy
across all scales. This lets each scale "veto" the others' mistakes. A word
that gets low energy from ANY scale is considered likely.

Energy scales (default):
  - Word:  1600  (strongest — exact word matches are most informative)
  - POS:    800  (half — syntactic patterns are useful but less specific)
  - Topic:  400  (quarter — discourse coherence is weakest per-token signal)

The relative scales ensure that when a word n-gram hits, it dominates
(the word scale is 4x the topic scale). But when word n-grams miss
(long context), the POS and topic scales provide crucial fallback.
"""

from typing import Dict, List, Optional

import numpy as np

from .base import AbstractRecallIndex
from .word_index import WordNgramIndex
from .pos_index import PosNgramIndex
from .topic_index import TopicNgramIndex


class MultiScaleRecall:
    """
    Combines recall from all scales: word, POS, and topic.

    This is the v17 equivalent of what recall alone did in v1-v16.
    Each scale independently computes energies, then they're combined
    as a product of experts (take the BEST/lowest energy per word).

    Product of Experts: take the minimum energy per word across all scales.
    This lets each scale "veto" the others' mistakes. A word that looks
    unlikely under word n-grams but likely under POS n-grams gets the
    POS energy -- it's a reasonable continuation given the syntax even
    if the exact word sequence hasn't been seen.
    """

    def __init__(
        self,
        word_index: Optional[WordNgramIndex] = None,
        pos_index: Optional[PosNgramIndex] = None,
        topic_index: Optional[TopicNgramIndex] = None,
        word_scale: int = 1600,
        pos_scale: int = 800,
        topic_scale: int = 400,
    ):
        """
        Args:
            word_index:  WordNgramIndex for exact word n-gram recall.
            pos_index:   PosNgramIndex for POS-tag n-gram recall.
            topic_index: TopicNgramIndex for topic n-gram recall.
            word_scale:  Energy scale for word-level recall (default 1600).
            pos_scale:   Energy scale for POS-level recall (default 800).
            topic_scale: Energy scale for topic-level recall (default 400).
        """
        self.word_index = word_index
        self.pos_index = pos_index
        self.topic_index = topic_index
        self.word_scale = word_scale
        self.pos_scale = pos_scale
        self.topic_scale = topic_scale

    def compute_energy(
        self,
        context_words: List[int],
        candidate_words: np.ndarray,
        longest_only: bool = True,
        interpolated: bool = False,
        kn_backoff: bool = False,
        context_weight_factor: int = 2,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute combined energy from all scales using product of experts.

        For each candidate word, takes the MINIMUM energy across all
        available scales. This means any scale can "veto" a high energy
        from another scale by providing a lower one.

        Args:
            context_words:     Context word IDs (passed to all indexes).
            candidate_words:   Array of candidate word IDs to score.
            longest_only:      If True, use only the longest n-gram match per scale.
            interpolated:      If True, use interpolated smoothing per scale.
            kn_backoff:        If True, use Kneser-Ney backoff per scale.
            context_weight_factor: Weight exponent for context length.

        Returns:
            np.ndarray of int64 energies, shape (len(candidate_words),).
            LOWER energy = more likely.
        """
        n_candidates = len(candidate_words)
        # Start with maximum possible energy (all scales missed)
        # The max across scales is the maximum any single scale would produce
        max_scale = max(self.word_scale, self.pos_scale, self.topic_scale)
        combined = np.full(n_candidates, 20 * max_scale, dtype=np.int64)

        # Word-level recall (strongest signal, most specific)
        if self.word_index is not None and self.word_index._built:
            word_energy = self.word_index.compute_energy(
                context_ids=context_words,
                candidate_words=candidate_words,
                recall_scale=self.word_scale,
                context_weight_factor=context_weight_factor,
                longest_only=longest_only,
                interpolated=interpolated,
                kn_backoff=kn_backoff,
            )
            combined = np.minimum(combined, word_energy)

        # POS-level recall (syntactic patterns, less sparse than word)
        if self.pos_index is not None and self.pos_index._built:
            pos_energy = self.pos_index.compute_energy(
                context_ids=context_words,
                candidate_words=candidate_words,
                recall_scale=self.pos_scale,
                context_weight_factor=context_weight_factor,
                longest_only=longest_only,
                interpolated=interpolated,
                kn_backoff=kn_backoff,
            )
            combined = np.minimum(combined, pos_energy)

        # Topic-level recall (discourse coherence, very dense)
        if self.topic_index is not None and self.topic_index._built:
            topic_energy = self.topic_index.compute_energy(
                context_ids=context_words,
                candidate_words=candidate_words,
                recall_scale=self.topic_scale,
                context_weight_factor=context_weight_factor,
                longest_only=longest_only,
                interpolated=interpolated,
                kn_backoff=kn_backoff,
            )
            combined = np.minimum(combined, topic_energy)

        return combined

    def lookup_all(self, context_words: List[int]) -> Dict[str, Dict]:
        """
        Look up continuations from all scales for debugging/analysis.

        Returns:
            Dict with keys 'word', 'pos', 'topic', each mapping to the
            lookup result from that scale.
        """
        results = {}
        if self.word_index is not None:
            results["word"] = self.word_index.lookup(context_words)
        if self.pos_index is not None:
            results["pos"] = self.pos_index.lookup(context_words)
        if self.topic_index is not None:
            results["topic"] = self.topic_index.lookup(context_words)
        return results

    def summary(self) -> str:
        """Return a human-readable summary of all indexes."""
        lines = ["MultiScaleRecall:"]
        if self.word_index is not None:
            built = "built" if self.word_index._built else "NOT built"
            n_ctx = sum(
                len(self.word_index.index[k])
                for k in range(1, self.word_index.max_n + 1)
            )
            lines.append(
                f"  Word:  max_n={self.word_index.max_n}, "
                f"{n_ctx:,} contexts, scale={self.word_scale} [{built}]"
            )
        else:
            lines.append("  Word:  None")

        if self.pos_index is not None:
            built = "built" if self.pos_index._built else "NOT built"
            n_ctx = sum(
                len(self.pos_index.index[k])
                for k in range(1, self.pos_index.max_n + 1)
            )
            lines.append(
                f"  POS:   max_n={self.pos_index.max_n}, "
                f"{n_ctx:,} contexts, scale={self.pos_scale} [{built}]"
            )
        else:
            lines.append("  POS:   None")

        if self.topic_index is not None:
            built = "built" if self.topic_index._built else "NOT built"
            n_ctx = sum(
                len(self.topic_index.index[k])
                for k in range(1, self.topic_index.max_n + 1)
            )
            lines.append(
                f"  Topic: max_n={self.topic_index.max_n}, "
                f"{n_ctx:,} contexts, scale={self.topic_scale} [{built}]"
            )
        else:
            lines.append("  Topic: None")

        return "\n".join(lines)
