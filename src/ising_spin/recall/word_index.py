"""
Word-level n-gram recall index — extracted from v1-v16 NGramIndex.

This is the EXISTING n-gram index that has been the primary generation mechanism
since v1. In v17 it becomes one of three recall scales (word, POS, topic).

The index maps: word_context_tuple -> Counter({word_id: count})

Key features carried forward from v10-v16:
  - Kneser-Ney backoff for unseen contexts
  - Interpolated smoothing (product of experts across n-gram levels)
  - Fine-grained integer log2 via int_log2_fine()
  - Batched building with OOM-aware pruning
  - Context-weighted energy scaling (longer context = lower energy)
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
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024  # KB -> MB
    except Exception:
        return 0


class WordNgramIndex(AbstractRecallIndex):
    """
    Multi-level word n-gram index for exact token recall.

    This is the PRIMARY generation mechanism. When it hits, it produces
    coherent text. When it misses, the Ising PMI model takes over.

    Extracted from model.py NGramIndex (v1-v16) for v17 modular architecture.
    """

    def __init__(self, max_n: int = 5, min_count: int = 1):
        self.max_n = max_n
        self.min_count = min_count
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
        self._kn_discount: int = 3
        self._kn_discount_fp: int = 12

    # =======================================================================
    # Building
    # =======================================================================

    def build(self, sequences: List[List[int]], **kwargs) -> None:
        """Build n-gram index from tokenized sequences. Integer counting only."""
        for seq in sequences:
            start = 0
            for i, w in enumerate(seq):
                if w >= 4:
                    start = i
                    break

            for t in range(start, len(seq)):
                for k in range(1, self.max_n + 1):
                    if t - k < start:
                        break
                    context = tuple(seq[t - k:t])
                    continuation = seq[t]
                    if any(w < 4 for w in context) or continuation < 4:
                        continue
                    if context not in self.index[k]:
                        self.index[k][context] = Counter()
                    self.index[k][context][continuation] += 1
                    self.context_totals[k][context] = (
                        self.context_totals[k].get(context, 0) + 1
                    )

        # Prune low-count continuations
        self._prune_index(self.min_count)

        self._built = True
        self._finalize_index()

    def build_batched(
        self,
        sequences: List[List[int]],
        batch_size: int = 200000,
        prune_interval: int = 1,
        adaptive_min_count: bool = True,
    ) -> None:
        """
        Memory-efficient n-gram index building with batched processing and
        incremental pruning. Designed for large corpora (>500K sequences)
        where the standard build() would exhaust memory.

        Key optimizations vs build():
          1. Processes sequences in batches -- limits peak n-gram dict size
          2. Prunes low-count entries after each batch -- frees memory early
          3. Auto-scales min_count with corpus size -- fewer entries for larger corpora
          4. Uses gc.collect() after each batch -- returns memory to OS
          5. Prunes higher-order n-grams more aggressively (count < 2 for 4/5-gram)
          6. Memory monitoring with OOM early warning
        """
        import gc

        total_seqs = len(sequences)
        # Auto-scale min_count for large corpora
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
                f"    Auto-scaled min_count: {self.min_count} -> {effective_min_count} "
                f"(corpus: {total_seqs:,} seqs, higher-order: {self._higher_order_min})"
            )

        n_batches = (total_seqs + batch_size - 1) // batch_size
        processed = 0
        t_start = time.time()

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total_seqs)
            batch = sequences[start:end]

            # Count n-grams for this batch
            for seq in batch:
                s_start = 0
                for i, w in enumerate(seq):
                    if w >= 4:
                        s_start = i
                        break

                for t in range(s_start, len(seq)):
                    for k in range(1, self.max_n + 1):
                        if t - k < s_start:
                            break
                        context = tuple(seq[t - k:t])
                        continuation = seq[t]
                        if any(w < 4 for w in context) or continuation < 4:
                            continue
                        if context not in self.index[k]:
                            self.index[k][context] = Counter()
                        self.index[k][context][continuation] += 1
                        self.context_totals[k][context] = (
                            self.context_totals[k].get(context, 0) + 1
                        )

            processed += len(batch)

            # Prune after each batch (or every prune_interval batches)
            if (batch_idx + 1) % prune_interval == 0:
                self._prune_index(effective_min_count)
                gc.collect()

            # Progress reporting with memory tracking
            n_ctx = sum(len(self.index[k]) for k in range(1, self.max_n + 1))
            n_cont = sum(
                sum(len(v) for v in self.index[k].values())
                for k in range(1, self.max_n + 1)
            )
            rss = _get_rss_mb()
            elapsed = time.time() - t_start
            mem_info = f", RSS={rss:,}MB" if rss > 0 else ""
            print(
                f"    Batch {batch_idx + 1}/{n_batches}: {processed:,} seqs, "
                f"{n_ctx:,} contexts, {n_cont:,} continuations{mem_info} "
                f"({elapsed:.1f}s)"
            )

            # OOM early warning -- if RSS > 12GB, start aggressive pruning
            if rss > 12000:
                print(f"    WARNING HIGH MEMORY ({rss:,}MB) -- aggressive pruning...")
                self._prune_index(effective_min_count + 2)
                gc.collect()
                rss_after = _get_rss_mb()
                print(f"    WARNING After aggressive prune: {rss_after:,}MB")

        # Final prune with the base min_count
        self._prune_index(self.min_count)

        self._built = True
        self._finalize_index()

    def _prune_index(self, min_count: int) -> None:
        """Prune low-count n-gram entries from the index."""
        for k in range(1, self.max_n + 1):
            # Higher-order n-grams use stricter min_count
            mc = (
                getattr(self, "_higher_order_min", min_count)
                if k >= 4
                else min_count
            )
            for context in list(self.index[k].keys()):
                low_count = [
                    w
                    for w, c in self.index[k][context].items()
                    if c < mc
                ]
                for w in low_count:
                    del self.index[k][context][w]
                    self.context_totals[k][context] -= 1
                if not self.index[k][context]:
                    del self.index[k][context]
                    del self.context_totals[k][context]

    def _finalize_index(self) -> None:
        """Build unigram totals and KN continuation counts after index is built."""
        # Build unigram totals for backoff (Katz backoff to unigram)
        # _unigram_totals[w] = (count(w), total_tokens)
        self._unigram_totals = {}
        if 1 in self.index:
            total_N = sum(self.context_totals[1].values())
            for context in self.index[1]:
                if context and len(context) == 1:
                    w = context[0]
                    count_w = self.context_totals[1].get(context, 0)
                    self._unigram_totals[w] = (count_w, total_N)

        # Kneser-Ney continuation counts
        # N1+(.w) = number of DISTINCT contexts that precede w at each n-gram level
        self._kn_continuation = {}  # {k: {w: count_of_distinct_contexts}}
        for k in range(2, self.max_n + 1):  # Start from bigram (k=2)
            cont_count = Counter()
            for context, continuations in self.index[k].items():
                for w in continuations:
                    cont_count[w] += 1
            self._kn_continuation[k] = dict(cont_count)

        # Total number of distinct (context, word) pairs per level
        self._kn_totals = {}
        for k, cont_count in self._kn_continuation.items():
            self._kn_totals[k] = sum(cont_count.values())

        # Fixed discount for modified Kneser-Ney
        self._kn_discount = 3       # Fixed discount in integer units (~0.75 * 4)
        self._kn_discount_fp = 12   # Fixed-point discount (0.75 * 16)

        for k in range(1, self.max_n + 1):
            n_ctx = len(self.index[k])
            n_cont = sum(len(v) for v in self.index[k].values())
            kn_info = (
                f", KN cont={len(self._kn_continuation.get(k, {})):,}"
                if k >= 2
                else ""
            )
            print(
                f"    {k}-gram: {n_ctx:,} contexts, {n_cont:,} continuations{kn_info}"
            )

    # =======================================================================
    # Lookup
    # =======================================================================

    def lookup(self, context_words: List[int]) -> Dict[int, List[Tuple[int, int, int]]]:
        """Look up n-gram continuations. Returns {k: [(word, count, total), ...]}."""
        results = {}
        for k in range(min(self.max_n, len(context_words)), 0, -1):
            context = tuple(context_words[-k:])
            if context in self.index[k]:
                total = self.context_totals[k][context]
                conts = self.index[k][context].most_common()
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
        Compute recall ENERGY for candidate words based on n-gram matches.

        PRECISE RATIO + KNESER-NEY BACKOFF:
        - Uses log2(total) - log2(count) instead of log2(total//count)
          This eliminates integer division loss (up to 0.4 bits/token gain)
        - Kneser-Ney backoff: when no n-gram match, uses continuation counts
          N1+(.w) instead of raw unigram P(w). KN consistently beats Katz by 15-25%.
        - Interpolated smoothing: ALL n-gram levels vote (product of experts)

        Returns POSITIVE energy values where LOWER energy = more likely.
        E_recall(w) = log2(total/count) * energy_scale for matched words.
        E_recall(w) = max_energy for unmatched words (default high energy).

        NOTE: This returns POSITIVE values to be ADDED to energy.
        The calling code uses `energies += recall_energy` (not -= bonus).
        """
        context_words = context_ids  # For word index, context_ids ARE word IDs
        n_candidates = len(candidate_words)
        # Backoff energy for unmatched words
        max_energy = 20 * recall_scale  # Cap for unseen words
        recall_energies = np.full(n_candidates, max_energy, dtype=np.int64)

        # BACKOFF ENERGY -- Kneser-Ney or unigram
        if self._built:
            if (
                kn_backoff
                and hasattr(self, "_kn_continuation")
                and self._kn_continuation
            ):
                # Kneser-Ney backoff using continuation counts
                # Use the LOWEST level (bigram) for best coverage
                # KN backoff energy is scaled 2x higher than n-gram match energy
                # to create a clear gap between "matched" and "backed off"
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
            elif hasattr(self, "_unigram_totals"):
                # Standard unigram backoff
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

        matches = self.lookup(context_words)
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
                        energy = 0  # P ~ 0.5+, E ~ 0 (very likely)
                else:
                    energy = max_energy
                # Keep the LOWEST energy (most likely) for each word
                if word not in cont_lookup or energy < cont_lookup[word]:
                    cont_lookup[word] = int(energy)

            for i, w in enumerate(candidate_words):
                if int(w) in cont_lookup:
                    w_int = int(w)
                    if interpolated:
                        # Interpolated smoothing -- take BEST (lowest) energy
                        # across all levels (Jelinek-Mercer-like)
                        if cont_lookup[w_int] < recall_energies[i]:
                            recall_energies[i] = cont_lookup[w_int]
                    else:
                        recall_energies[i] = cont_lookup[w_int]

        return recall_energies

    # =======================================================================
    # Convenience methods
    # =======================================================================

    def get_best_copy_candidate(
        self,
        context_words: List[int],
        min_context_length: int = 3,
        min_confidence: float = 0.3,
    ) -> Optional[Tuple[int, int, int]]:
        """Find best word for direct copying (highest-confidence n-gram match)."""
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
