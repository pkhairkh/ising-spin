"""
Topic-level n-gram recall index for discourse-level coherence.

Context = sequence of topic assignments for context words.
With only ~16 topics, even 10-grams are well-populated, enabling
discourse-level recall that word/POS n-grams miss entirely.

The index maps: topic_context_tuple -> Counter({word_id: count})

Example:
  [SCIENCE, SCIENCE, SCIENCE, ...] (10 consecutive topic IDs) -> technical vocabulary
  This captures discourse-level coherence: words from the same topic cluster
  together, and the topic context strongly predicts which words follow.

The topic sequence is very compact (16 values) so even 10-grams are
well-populated -- 16^10 = ~1 trillion possible patterns, but in practice
only a tiny fraction appear, and they appear many times.
"""

import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import AbstractRecallIndex
from ..sampling.boltzmann import int_log2_fine


def _get_rss_mb() -> int:
    """Get current process RSS in MB (0 if unavailable)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    except Exception:
        return 0


class TopicNgramIndex(AbstractRecallIndex):
    """
    Topic-level n-gram index for discourse-level recall.

    Context = sequence of topic assignments for context words.
    max_n = 10 (very long context because there are only 16 topics).

    Maps: topic_context_tuple -> {word_id: count}

    Example: [SCIENCE, SCIENCE, SCIENCE, ...] -> technical vocabulary
    This captures discourse-level coherence that word/POS n-grams miss entirely.
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
            max_n:       Maximum topic n-gram length (default 10).
            min_count:   Minimum count for a continuation to be kept.
            n_topics:    Number of topics (default 16).
            word_topics: (vocab_size,) int8 array from TopicAssigner, mapping
                         each word ID to its topic ID. Can also be provided at
                         build time.
        """
        self.max_n = max_n
        self.min_count = min_count
        self.n_topics = n_topics
        self.word_topics = word_topics  # (vocab_size,) int8 array or None

        # Index: {k: {topic_context_tuple: Counter({word_id: count})}}
        self.index: Dict[int, Dict[Tuple, Counter]] = {
            k: {} for k in range(1, max_n + 1)
        }
        self.context_totals: Dict[int, Dict[Tuple, int]] = {
            k: {} for k in range(1, max_n + 1)
        }
        self._built = False
        self._higher_order_min = min_count

        # Populated by _finalize_index()
        self._unigram_totals: Dict[int, Tuple[int, int]] = {}
        self._kn_continuation: Dict[int, Dict[int, int]] = {}
        self._kn_totals: Dict[int, int] = {}

    def _word_to_topic(self, word_id: int) -> int:
        """Convert a word ID to its topic ID. Unknown words get topic 0."""
        if self.word_topics is not None and word_id < len(self.word_topics):
            return int(self.word_topics[word_id])
        return 0

    def _seq_to_topics(self, seq: List[int]) -> List[int]:
        """Convert a word ID sequence to a topic ID sequence."""
        return [self._word_to_topic(w) for w in seq]

    # =======================================================================
    # Building
    # =======================================================================

    def build(
        self,
        sequences: List[List[int]],
        word_topics: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """
        Build topic n-gram index from training sequences.

        word_topics: (vocab_size,) int8 array from TopicAssigner.
        If not provided here, must have been set in __init__.
        """
        if word_topics is not None:
            self.word_topics = word_topics

        if self.word_topics is None:
            raise ValueError(
                "TopicNgramIndex requires word_topics (from TopicAssigner). "
                "Provide via __init__ or build()."
            )

        for seq in sequences:
            start = 0
            for i, w in enumerate(seq):
                if w >= 4:
                    start = i
                    break

            # Convert to topic ID sequence
            topic_seq = self._seq_to_topics(seq)

            for t in range(start, len(seq)):
                continuation = seq[t]  # Record the WORD, not the topic
                if continuation < 4:
                    continue

                for k in range(1, self.max_n + 1):
                    if t - k < start:
                        break

                    # Context is topic IDs, not word IDs
                    topic_context = tuple(topic_seq[t - k:t])

                    if topic_context not in self.index[k]:
                        self.index[k][topic_context] = Counter()
                    self.index[k][topic_context][continuation] += 1
                    self.context_totals[k][topic_context] = (
                        self.context_totals[k].get(topic_context, 0) + 1
                    )

        # Prune low-count continuations
        self._prune_index(self.min_count)

        self._built = True
        self._finalize_index()

    def build_batched(
        self,
        sequences: List[List[int]],
        word_topics: Optional[np.ndarray] = None,
        batch_size: int = 200000,
        prune_interval: int = 1,
        adaptive_min_count: bool = True,
    ) -> None:
        """
        Memory-efficient topic n-gram index building with batched processing.
        Same OOM-aware pattern as WordNgramIndex.build_batched.
        """
        import gc

        if word_topics is not None:
            self.word_topics = word_topics

        if self.word_topics is None:
            raise ValueError(
                "TopicNgramIndex requires word_topics (from TopicAssigner). "
                "Provide via __init__ or build_batched()."
            )

        total_seqs = len(sequences)
        effective_min_count = self.min_count
        if adaptive_min_count and total_seqs > 500000:
            import math
            scale = max(
                self.min_count,
                min(5, int(math.log2(total_seqs / 500000)) + self.min_count),
            )
            effective_min_count = scale
            self._higher_order_min = effective_min_count + 1
        else:
            self._higher_order_min = effective_min_count

        if effective_min_count != self.min_count:
            print(
                f"    [TOPIC] Auto-scaled min_count: {self.min_count} -> {effective_min_count} "
                f"(corpus: {total_seqs:,} seqs, higher-order: {self._higher_order_min})"
            )

        n_batches = (total_seqs + batch_size - 1) // batch_size
        processed = 0
        t_start = time.time()

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total_seqs)
            batch = sequences[start:end]

            for seq in batch:
                s_start = 0
                for i, w in enumerate(seq):
                    if w >= 4:
                        s_start = i
                        break

                topic_seq = self._seq_to_topics(seq)

                for t in range(s_start, len(seq)):
                    continuation = seq[t]
                    if continuation < 4:
                        continue

                    for k in range(1, self.max_n + 1):
                        if t - k < s_start:
                            break
                        topic_context = tuple(topic_seq[t - k:t])
                        if topic_context not in self.index[k]:
                            self.index[k][topic_context] = Counter()
                        self.index[k][topic_context][continuation] += 1
                        self.context_totals[k][topic_context] = (
                            self.context_totals[k].get(topic_context, 0) + 1
                        )

            processed += len(batch)

            if (batch_idx + 1) % prune_interval == 0:
                self._prune_index(effective_min_count)
                gc.collect()

            n_ctx = sum(len(self.index[k]) for k in range(1, self.max_n + 1))
            n_cont = sum(
                sum(len(v) for v in self.index[k].values())
                for k in range(1, self.max_n + 1)
            )
            rss = _get_rss_mb()
            elapsed = time.time() - t_start
            mem_info = f", RSS={rss:,}MB" if rss > 0 else ""
            print(
                f"    [TOPIC] Batch {batch_idx + 1}/{n_batches}: {processed:,} seqs, "
                f"{n_ctx:,} contexts, {n_cont:,} continuations{mem_info} "
                f"({elapsed:.1f}s)"
            )

            if rss > 12000:
                print(f"    [TOPIC] WARNING HIGH MEMORY ({rss:,}MB) -- aggressive pruning...")
                self._prune_index(effective_min_count + 2)
                gc.collect()
                rss_after = _get_rss_mb()
                print(f"    [TOPIC] After aggressive prune: {rss_after:,}MB")

        self._prune_index(self.min_count)

        self._built = True
        self._finalize_index()

    def _prune_index(self, min_count: int) -> None:
        """Prune low-count n-gram entries from the index."""
        for k in range(1, self.max_n + 1):
            mc = (
                getattr(self, "_higher_order_min", min_count)
                if k >= 6  # Higher-order for topics uses higher threshold
                else min_count
            )
            for context in list(self.index[k].keys()):
                low_count = [
                    w for w, c in self.index[k][context].items() if c < mc
                ]
                for w in low_count:
                    del self.index[k][context][w]
                    self.context_totals[k][context] -= 1
                if not self.index[k][context]:
                    del self.index[k][context]
                    del self.context_totals[k][context]

    def _finalize_index(self) -> None:
        """Build unigram totals and KN continuation counts after index is built."""
        # Unigram totals: for topic 1-gram context, aggregate word counts
        self._unigram_totals = {}
        if 1 in self.index:
            total_N = sum(self.context_totals[1].values())
            for topic_context, continuations in self.index[1].items():
                for w, count in continuations.items():
                    if w not in self._unigram_totals:
                        self._unigram_totals[w] = (0, total_N)
                    existing_count, _ = self._unigram_totals[w]
                    self._unigram_totals[w] = (existing_count + count, total_N)

        # Kneser-Ney continuation counts
        self._kn_continuation = {}
        for k in range(2, self.max_n + 1):
            cont_count = Counter()
            for context, continuations in self.index[k].items():
                for w in continuations:
                    cont_count[w] += 1
            self._kn_continuation[k] = dict(cont_count)

        self._kn_totals = {}
        for k, cont_count in self._kn_continuation.items():
            self._kn_totals[k] = sum(cont_count.values())

        # Print summary
        for k in range(1, self.max_n + 1):
            n_ctx = len(self.index[k])
            n_cont = sum(len(v) for v in self.index[k].values())
            if n_ctx > 0:
                kn_info = (
                    f", KN cont={len(self._kn_continuation.get(k, {})):,}"
                    if k >= 2
                    else ""
                )
                print(
                    f"    [TOPIC] {k}-gram: {n_ctx:,} contexts, {n_cont:,} continuations{kn_info}"
                )

    # =======================================================================
    # Lookup
    # =======================================================================

    def lookup(self, context_ids: List[int]) -> Dict[int, List[Tuple[int, int, int]]]:
        """
        Look up n-gram continuations using topic context.

        context_ids: word IDs -- converted to topic IDs internally.
        Returns {k: [(word, count, total), ...]}.
        """
        # Convert word IDs to topic IDs
        topic_context = [self._word_to_topic(w) for w in context_ids]

        results = {}
        for k in range(min(self.max_n, len(topic_context)), 0, -1):
            topic_tuple = tuple(topic_context[-k:])
            if topic_tuple in self.index[k]:
                total = self.context_totals[k][topic_tuple]
                conts = self.index[k][topic_tuple].most_common()
                results[k] = [(word, count, total) for word, count in conts]
        return results

    # =======================================================================
    # Energy computation
    # =======================================================================

    def compute_energy(
        self,
        context_ids: List[int],
        candidate_words: np.ndarray,
        recall_scale: int = 100,
        context_weight_factor: int = 2,
        longest_only: bool = True,
        interpolated: bool = False,
        kn_backoff: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute energy from topic context match.

        context_ids are word IDs -- converted to topic IDs internally.
        Look up the topic context in the index. For matching continuations:
          E(w) = log2(total/count) * topic_recall_scale
        For non-matching: backoff energy from KN/unigram statistics.
        """
        n_candidates = len(candidate_words)
        max_energy = 20 * recall_scale
        recall_energies = np.full(n_candidates, max_energy, dtype=np.int64)

        # Backoff energy
        if self._built:
            if (
                kn_backoff
                and hasattr(self, "_kn_continuation")
                and self._kn_continuation
            ):
                best_kn_level = min(self._kn_continuation.keys())
                kn_cont = self._kn_continuation[best_kn_level]
                kn_total = self._kn_totals[best_kn_level]
                if kn_total > 0:
                    for i, w in enumerate(candidate_words):
                        w_int = int(w)
                        if w_int in kn_cont:
                            n_ctx_w = kn_cont[w_int]
                            if n_ctx_w > 0 and kn_total > n_ctx_w:
                                ratio = kn_total // n_ctx_w
                                if ratio >= 2:
                                    fine_log2 = int_log2_fine(ratio)
                                    recall_energies[i] = (
                                        fine_log2 * recall_scale * 2
                                    ) >> 8
            elif hasattr(self, "_unigram_totals") and self._unigram_totals:
                for i, w in enumerate(candidate_words):
                    w_int = int(w)
                    if w_int in self._unigram_totals:
                        count_w, total_N = self._unigram_totals[w_int]
                        if count_w > 0 and total_N > count_w:
                            ratio = total_N // count_w
                            if ratio >= 2:
                                fine_log2 = int_log2_fine(ratio)
                                recall_energies[i] = (
                                    fine_log2 * recall_scale
                                ) >> 8

        matches = self.lookup(context_ids)
        if not matches:
            return recall_energies

        if longest_only and not interpolated and matches:
            best_k = max(matches.keys())
            matches = {best_k: matches[best_k]}

        for k, continuations in matches.items():
            context_weight = context_weight_factor ** (k - 1)
            cont_lookup = {}
            for word, count, total in continuations:
                if count > 0 and total > 0:
                    ratio = total // max(1, count)
                    if ratio >= 2:
                        fine_log2 = int_log2_fine(ratio)
                        energy = (fine_log2 * recall_scale * context_weight) >> 8
                    else:
                        energy = 0
                else:
                    energy = max_energy
                if word not in cont_lookup or energy < cont_lookup[word]:
                    cont_lookup[word] = int(energy)

            for i, w in enumerate(candidate_words):
                if int(w) in cont_lookup:
                    w_int = int(w)
                    if interpolated:
                        if cont_lookup[w_int] < recall_energies[i]:
                            recall_energies[i] = cont_lookup[w_int]
                    else:
                        recall_energies[i] = cont_lookup[w_int]

        return recall_energies
