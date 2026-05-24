"""
Shared base class for all n-gram recall indexes.

Extracts the ~1000 lines of duplicated build/prune/finalize/energy logic
that was previously copy-pasted across WordNgramIndex, PosNgramIndex,
and TopicNgramIndex.

Subclasses only need to implement:
  - _context_to_key(context_words, k) → hashable key for the n-gram level
  - _should_skip_context(context_key, k) → bool, whether to skip this context
  - _label → str, prefix for log messages (e.g., "WORD", "POS", "TOPIC")
"""

from __future__ import annotations

import gc
import math
import time
from abc import abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..exceptions import IndexBuildError, ValidationError
from ..sampling.boltzmann import int_log2_fine
from ..utils import get_rss_mb, validate_array, validate_nonempty


class NgramIndexBase:
    """
    Base class for n-gram recall indexes with shared build/prune/energy logic.

    All three v17 recall indexes (word, POS, topic) share identical patterns:
      - Multi-level n-gram counting (index + context_totals)
      - Pruning low-count entries
      - Kneser-Ney continuation counts
      - Energy computation with KN backoff / unigram backoff
      - Batched building with OOM protection

    This base class factors out all that shared logic. Subclasses provide:
      - _context_to_key(): how to convert context word IDs to the index key
      - _should_skip_context(): whether to skip a context during indexing
      - _label: prefix for log messages
    """

    def __init__(
        self,
        max_n: int,
        min_count: int,
        higher_order_threshold: int = 4,
    ):
        """
        Args:
            max_n: Maximum n-gram order.
            min_count: Minimum continuation count to keep.
            higher_order_threshold: n-gram levels >= this use stricter pruning.
        """
        if max_n < 1:
            raise ValidationError(f"max_n must be >= 1, got {max_n}")
        if min_count < 1:
            raise ValidationError(f"min_count must be >= 1, got {min_count}")

        self.max_n = max_n
        self.min_count = min_count
        self._higher_order_threshold = higher_order_threshold
        self._higher_order_min = min_count

        # Core index: {k: {context_key: Counter({word_id: count})}}
        self.index: Dict[int, Dict[tuple, Counter]] = {
            k: {} for k in range(1, max_n + 1)
        }
        self.context_totals: Dict[int, Dict[tuple, int]] = {
            k: {} for k in range(1, max_n + 1)
        }
        self._built = False

        # Populated by _finalize_index()
        self._unigram_totals: Dict[int, Tuple[int, int]] = {}
        self._kn_continuation: Dict[int, Dict[int, int]] = {}
        self._kn_totals: Dict[int, int] = {}

    # ── Abstract interface ────────────────────────────────────────────────

    @property
    @abstractmethod
    def _label(self) -> str:
        """Label for log messages (e.g., "WORD", "POS", "TOPIC")."""

    @abstractmethod
    def _context_to_key(self, context_words: List[int], k: int) -> tuple | None:
        """
        Convert context word IDs to the n-gram index key for level k.

        Returns None if the context should not be indexed (e.g., too many
        unknown POS tags).
        """

    @abstractmethod
    def _should_skip_continuation(self, word_id: int) -> bool:
        """Return True if this continuation word should be skipped."""

    # ── Build ─────────────────────────────────────────────────────────────

    def build(self, sequences: List[List[int]], **kwargs) -> None:
        """
        Build n-gram index from tokenized sequences. Integer counting only.

        Args:
            sequences: List of tokenized sequences (list of word ID lists).

        Raises:
            IndexBuildError: if the index cannot be built.
        """
        if not sequences:
            raise IndexBuildError("Cannot build index from empty sequences")

        for seq in sequences:
            start = self._find_content_start(seq)
            self._index_sequence(seq, start)

        self._prune_index(self.min_count)
        self._built = True
        self._finalize_index()

    def build_batched(
        self,
        sequences: List[List[int]],
        batch_size: int = 200000,
        prune_interval: int = 1,
        adaptive_min_count: bool = True,
        oom_threshold_mb: int = 12000,
        **kwargs,
    ) -> None:
        """
        Memory-efficient n-gram index building with batched processing.

        Key optimizations vs build():
          1. Processes sequences in batches — limits peak n-gram dict size
          2. Prunes low-count entries after each batch — frees memory early
          3. Auto-scales min_count with corpus size
          4. Uses gc.collect() after each batch — returns memory to OS
          5. Memory monitoring with OOM early warning

        Args:
            sequences: Tokenized sequences.
            batch_size: Number of sequences per batch.
            prune_interval: Prune after this many batches.
            adaptive_min_count: Auto-scale min_count with corpus size.
            oom_threshold_mb: RSS threshold (MB) to trigger aggressive pruning.
                              Default 12000; set lower for constrained devices.
                              For 16GB Pi: 11000; for 8GB: 6000.

        Raises:
            IndexBuildError: if the index cannot be built.
        """
        if not sequences:
            raise IndexBuildError("Cannot build index from empty sequences")

        total_seqs = len(sequences)

        # Auto-scale min_count for large corpora
        effective_min_count = self.min_count
        if adaptive_min_count and total_seqs > 500000:
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
                f"    [{self._label}] Auto-scaled min_count: "
                f"{self.min_count} -> {effective_min_count} "
                f"(corpus: {total_seqs:,} seqs, "
                f"higher-order: {self._higher_order_min})"
            )

        n_batches = (total_seqs + batch_size - 1) // batch_size
        processed = 0
        t_start = time.time()

        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_seqs)
            batch = sequences[batch_start:batch_end]

            for seq in batch:
                start = self._find_content_start(seq)
                self._index_sequence(seq, start)

            processed += len(batch)

            if (batch_idx + 1) % prune_interval == 0:
                self._prune_index(effective_min_count)
                gc.collect()

            self._log_batch_progress(
                batch_idx, n_batches, processed, t_start
            )

            # OOM early warning — threshold is now configurable
            rss = get_rss_mb()
            if rss > oom_threshold_mb:
                print(
                    f"    [{self._label}] WARNING HIGH MEMORY "
                    f"({rss:,}MB > {oom_threshold_mb:,}MB) -- aggressive pruning..."
                )
                # Progressive pruning: increase min_count until under threshold
                extra_prune = 2
                while rss > oom_threshold_mb and extra_prune <= 8:
                    self._prune_index(effective_min_count + extra_prune)
                    gc.collect()
                    rss = get_rss_mb()
                    print(
                        f"    [{self._label}] Prune min_count+{extra_prune}: "
                        f"{rss:,}MB"
                    )
                    extra_prune += 2

        self._prune_index(self.min_count)
        self._built = True
        self._finalize_index()

    # ── Lookup ────────────────────────────────────────────────────────────

    def lookup(self, context_ids: List[int]) -> Dict[int, List[Tuple[int, int, int]]]:
        """
        Look up n-gram continuations for a context.

        Returns {k: [(word, count, total), ...]} sorted by count descending.
        """
        results = {}
        for k in range(min(self.max_n, len(context_ids)), 0, -1):
            context_key = self._context_to_key(context_ids, k)
            if context_key is not None and context_key in self.index[k]:
                total = self.context_totals[k][context_key]
                conts = self.index[k][context_key].most_common()
                results[k] = [(word, count, total) for word, count in conts]
        return results

    # ── Energy computation ────────────────────────────────────────────────

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
        Compute recall energy for candidate words. LOWER energy = more likely.

        Energy = log2(total / count) * recall_scale for matched words.
        Backoff energy from KN or unigram stats for unmatched words.
        Max energy for completely unseen words.

        Args:
            context_ids: Context identifiers (word IDs, converted internally).
            candidate_words: Array of candidate word IDs to score.
            recall_scale: Energy scale factor.
            context_weight_factor: Weight exponent for context length.
            longest_only: Use only the longest n-gram match.
            interpolated: Use interpolated smoothing (all levels vote).
            kn_backoff: Use Kneser-Ney backoff.

        Returns:
            int64 array of energies, shape (len(candidate_words),).
        """
        if len(candidate_words) == 0:
            return np.array([], dtype=np.int64)

        n_candidates = len(candidate_words)
        max_energy = 20 * recall_scale
        recall_energies = np.full(n_candidates, max_energy, dtype=np.int64)

        # Backoff energy
        if self._built:
            self._apply_backoff_energy(
                candidate_words, recall_energies, recall_scale, kn_backoff
            )

        # N-gram match energy
        matches = self.lookup(context_ids)
        if not matches:
            return recall_energies

        if longest_only and not interpolated and matches:
            best_k = max(matches.keys())
            matches = {best_k: matches[best_k]}

        for k, continuations in matches.items():
            context_weight = context_weight_factor ** (k - 1)
            cont_lookup = self._compute_continuation_energies(
                continuations, recall_scale, context_weight, max_energy
            )

            for i, w in enumerate(candidate_words):
                w_int = int(w)
                if w_int in cont_lookup:
                    if interpolated:
                        if cont_lookup[w_int] < recall_energies[i]:
                            recall_energies[i] = cont_lookup[w_int]
                    else:
                        recall_energies[i] = cont_lookup[w_int]

        return recall_energies

    # ── Internal helpers ──────────────────────────────────────────────────

    def _find_content_start(self, seq: List[int]) -> int:
        """Find the first non-special token index (word_id >= 4)."""
        for i, w in enumerate(seq):
            if w >= 4:
                return i
        return len(seq)

    def _index_sequence(self, seq: List[int], start: int) -> None:
        """Index n-grams from a single sequence."""
        for t in range(start, len(seq)):
            continuation = seq[t]
            if self._should_skip_continuation(continuation):
                continue

            for k in range(1, self.max_n + 1):
                if t - k < start:
                    break

                context_key = self._context_to_key(seq[t - k:t], k)
                if context_key is None:
                    continue

                if context_key not in self.index[k]:
                    self.index[k][context_key] = Counter()
                self.index[k][context_key][continuation] += 1
                self.context_totals[k][context_key] = (
                    self.context_totals[k].get(context_key, 0) + 1
                )

    def _prune_index(self, min_count: int) -> None:
        """Prune low-count n-gram entries from the index."""
        for k in range(1, self.max_n + 1):
            mc = (
                self._higher_order_min
                if k >= self._higher_order_threshold
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
        """Build unigram totals and KN continuation counts."""
        # Unigram totals
        self._unigram_totals = {}
        if 1 in self.index:
            total_N = sum(self.context_totals[1].values())
            for context, continuations in self.index[1].items():
                if context and len(context) == 1:
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

        # Log summary
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
                    f"    [{self._label}] {k}-gram: {n_ctx:,} contexts, "
                    f"{n_cont:,} continuations{kn_info}"
                )

    def _apply_backoff_energy(
        self,
        candidate_words: np.ndarray,
        recall_energies: np.ndarray,
        recall_scale: int,
        kn_backoff: bool,
    ) -> None:
        """Apply backoff energy from KN or unigram statistics."""
        if (
            kn_backoff
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
        elif self._unigram_totals:
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

    def _compute_continuation_energies(
        self,
        continuations: List[Tuple[int, int, int]],
        recall_scale: int,
        context_weight: int,
        max_energy: int,
    ) -> Dict[int, int]:
        """Compute energy for each continuation word. Returns {word_id: energy}."""
        cont_lookup: Dict[int, int] = {}
        for word, count, total in continuations:
            if count > 0 and total > 0:
                ratio = total // max(1, count)
                if ratio >= 2:
                    fine_log2 = int_log2_fine(ratio)
                    energy = (fine_log2 * recall_scale * context_weight) >> 8
                else:
                    energy = 0  # P ~ 0.5+, E ~ 0
            else:
                energy = max_energy
            if word not in cont_lookup or energy < cont_lookup[word]:
                cont_lookup[word] = int(energy)
        return cont_lookup

    def _log_batch_progress(
        self,
        batch_idx: int,
        n_batches: int,
        processed: int,
        t_start: float,
    ) -> None:
        """Log progress for batched building."""
        n_ctx = sum(len(self.index[k]) for k in range(1, self.max_n + 1))
        n_cont = sum(
            sum(len(v) for v in self.index[k].values())
            for k in range(1, self.max_n + 1)
        )
        rss = get_rss_mb()
        elapsed = time.time() - t_start
        mem_info = f", RSS={rss:,}MB" if rss > 0 else ""
        print(
            f"    [{self._label}] Batch {batch_idx + 1}/{n_batches}: "
            f"{processed:,} seqs, {n_ctx:,} contexts, "
            f"{n_cont:,} continuations{mem_info} ({elapsed:.1f}s)"
        )
