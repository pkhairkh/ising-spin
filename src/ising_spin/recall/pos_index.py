"""
POS-level n-gram recall index -- the KEY INNOVATION of v17.

Instead of matching exact word sequences, this index matches POS TAG sequences.
This gives much longer effective context because POS n-grams are far more
regular than word n-grams.

The index maps: POS_context_tuple -> Counter({word_id: count})

Example:
  Word 5-gram ["the", "big", "brown", "dog", "chased"] is probably unique.
  POS  5-gram [DET, ADJ, ADJ, NOUN, VERB]   has been seen thousands of times.
  POS 10-gram (full clause pattern)           still has meaningful counts.

When the exact 5-word n-gram is unseen, the POS pattern HAS been seen, and
it tells us: "after DET ADJ ADJ NOUN VERB, the next word is typically a
PREP, DET, or ADV" -- and which specific words follow.
"""

import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import AbstractRecallIndex
from ..sampling.boltzmann import int_log2_fine
from ..vocabulary.pos import POSTypeSystem, N_POS, POS2IDX


def _get_rss_mb() -> int:
    """Get current process RSS in MB (0 if unavailable)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    except Exception:
        return 0


# Tag priority for selecting a single POS tag per word
# (lower = higher priority for disambiguation)
_TAG_PRIORITY = {
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


class PosNgramIndex(AbstractRecallIndex):
    """
    POS-level n-gram index for abstract recall.

    Instead of context = [w1, w2, w3, w4], we use context = [pos1, pos2, pos3, pos4, ...]
    up to max_n=10 (much longer than word n-grams because POS sequences are far less sparse).

    The index maps: POS context tuple -> {word_id: count}

    This captures patterns like: "DET ADJ NOUN VERB PREP DET" -> {specific_nouns: counts}
    When the exact 5-word n-gram is unseen, this POS pattern HAS been seen thousands of times.
    """

    def __init__(
        self,
        max_n: int = 10,
        min_count: int = 2,
        pos_system: Optional[POSTypeSystem] = None,
    ):
        self.max_n = max_n  # Longer than word n-grams (10 vs 5)
        self.min_count = min_count
        self.pos_system = pos_system

        # word_pos_tags: dict mapping word_id -> primary POS tag (int)
        # Built from pos_system if not provided at build time
        self.word_pos_tags: Dict[int, int] = {}

        # Index: {k: {pos_context_tuple: Counter({word_id: count})}}
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

    def _derive_pos_tags(self) -> None:
        """Derive word_pos_tags from pos_system if not already set."""
        if self.word_pos_tags:
            return
        if self.pos_system is None:
            raise ValueError(
                "PosNgramIndex requires either word_pos_tags or pos_system "
                "to map words to POS tags."
            )
        # For each word, select its primary (highest-priority) POS tag
        for w, allowed in self.pos_system.allowed_types.items():
            if allowed:
                self.word_pos_tags[w] = min(allowed, key=lambda t: _TAG_PRIORITY.get(t, 99))

    def _word_to_pos(self, word_id: int) -> int:
        """Convert a word ID to its POS tag. Unknown words get X tag."""
        return self.word_pos_tags.get(word_id, POS2IDX["X"])

    def _seq_to_pos(self, seq: List[int]) -> List[int]:
        """Convert a word ID sequence to a POS tag sequence."""
        return [self._word_to_pos(w) for w in seq]

    # =======================================================================
    # Building
    # =======================================================================

    def build(
        self,
        sequences: List[List[int]],
        word_pos_tags: Optional[Dict[int, int]] = None,
        **kwargs,
    ) -> None:
        """
        Build from training sequences.

        word_pos_tags: dict mapping word_id -> primary POS tag (int).
        If not provided, derive from pos_system.

        For each position in each sequence:
          1. Get the POS tag for each context word
          2. Form POS context tuples of length 1..max_n
          3. Record the continuation WORD (not POS!) at each position

        This gives us: POS pattern -> which words follow
        """
        if word_pos_tags is not None:
            self.word_pos_tags = word_pos_tags
        self._derive_pos_tags()

        for seq in sequences:
            # v17.4: Handle <S> sentence boundaries
            NS = 5  # Number of special tokens
            start = 0
            for i, w in enumerate(seq):
                if w >= NS:
                    start = i
                    break
                elif w == 4:  # <S> token
                    start = i + 1

            # Convert to POS tag sequence
            pos_seq = self._seq_to_pos(seq)

            eff_start = start
            for t in range(start, len(seq)):
                continuation = seq[t]  # Record the WORD, not the POS tag
                if continuation < NS:
                    if continuation == 4:  # <S> — sentence boundary
                        eff_start = t + 1
                    continue

                for k in range(1, self.max_n + 1):
                    if t - k < eff_start:
                        break

                    # Context is POS tags, not word IDs
                    pos_context = tuple(pos_seq[t - k:t])

                    # Skip if context has too many X (unknown) tags
                    x_count = sum(1 for p in pos_context if p == POS2IDX["X"])
                    if x_count > k // 2:
                        continue

                    if pos_context not in self.index[k]:
                        self.index[k][pos_context] = Counter()
                    self.index[k][pos_context][continuation] += 1
                    self.context_totals[k][pos_context] = (
                        self.context_totals[k].get(pos_context, 0) + 1
                    )

        # Prune low-count continuations
        self._prune_index(self.min_count)

        self._built = True
        self._finalize_index()

    def build_batched(
        self,
        sequences: List[List[int]],
        word_pos_tags: Optional[Dict[int, int]] = None,
        batch_size: int = 200000,
        prune_interval: int = 1,
        adaptive_min_count: bool = True,
    ) -> None:
        """
        Memory-efficient POS n-gram index building with batched processing.
        Same OOM-aware pattern as WordNgramIndex.build_batched.
        """
        import gc

        if word_pos_tags is not None:
            self.word_pos_tags = word_pos_tags
        self._derive_pos_tags()

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
                f"    [POS] Auto-scaled min_count: {self.min_count} -> {effective_min_count} "
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
                # v17.4: Handle <S> sentence boundaries
                NS = 5
                s_start = 0
                for i, w in enumerate(seq):
                    if w >= NS:
                        s_start = i
                        break
                    elif w == 4:  # <S> token
                        s_start = i + 1

                pos_seq = self._seq_to_pos(seq)

                eff_start = s_start
                for t in range(s_start, len(seq)):
                    continuation = seq[t]
                    if continuation < NS:
                        if continuation == 4:  # <S> — sentence boundary
                            eff_start = t + 1
                        continue

                    for k in range(1, self.max_n + 1):
                        if t - k < eff_start:
                            break
                        pos_context = tuple(pos_seq[t - k:t])
                        x_count = sum(1 for p in pos_context if p == POS2IDX["X"])
                        if x_count > k // 2:
                            continue
                        if pos_context not in self.index[k]:
                            self.index[k][pos_context] = Counter()
                        self.index[k][pos_context][continuation] += 1
                        self.context_totals[k][pos_context] = (
                            self.context_totals[k].get(pos_context, 0) + 1
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
                f"    [POS] Batch {batch_idx + 1}/{n_batches}: {processed:,} seqs, "
                f"{n_ctx:,} contexts, {n_cont:,} continuations{mem_info} "
                f"({elapsed:.1f}s)"
            )

            if rss > 12000:
                print(f"    [POS] WARNING HIGH MEMORY ({rss:,}MB) -- aggressive pruning...")
                self._prune_index(effective_min_count + 2)
                gc.collect()
                rss_after = _get_rss_mb()
                print(f"    [POS] After aggressive prune: {rss_after:,}MB")

        self._prune_index(self.min_count)

        self._built = True
        self._finalize_index()

    def _prune_index(self, min_count: int) -> None:
        """Prune low-count n-gram entries from the index."""
        for k in range(1, self.max_n + 1):
            mc = (
                getattr(self, "_higher_order_min", min_count)
                if k >= 8  # Higher-order for POS uses higher threshold (8+)
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
        # Unigram totals: for POS 1-gram context, count words that follow
        self._unigram_totals = {}
        if 1 in self.index:
            total_N = sum(self.context_totals[1].values())
            for pos_context, continuations in self.index[1].items():
                if pos_context and len(pos_context) == 1:
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
                    f"    [POS] {k}-gram: {n_ctx:,} contexts, {n_cont:,} continuations{kn_info}"
                )

    # =======================================================================
    # Lookup
    # =======================================================================

    def lookup(self, context_ids: List[int]) -> Dict[int, List[Tuple[int, int, int]]]:
        """
        Look up n-gram continuations using POS context.

        context_ids: word IDs -- converted to POS tags internally.
        Returns {k: [(word, count, total), ...]}.
        """
        # v17.4: Truncate context at last <S> (sentence boundary)
        SENT_IDX = 4
        last_sent = -1
        for i, w in enumerate(context_ids):
            if w == SENT_IDX:
                last_sent = i
        if last_sent >= 0:
            context_ids = context_ids[last_sent + 1:]

        # Convert word IDs to POS tags
        pos_context = [self._word_to_pos(w) for w in context_ids]

        results = {}
        for k in range(min(self.max_n, len(pos_context)), 0, -1):
            pos_tuple = tuple(pos_context[-k:])
            if pos_tuple in self.index[k]:
                total = self.context_totals[k][pos_tuple]
                conts = self.index[k][pos_tuple].most_common()
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
        Compute energy from POS context match.

        context_ids are word IDs -- converted to POS tags internally.
        Look up the POS context in the index. For matching continuations:
          E(w) = log2(total/count) * pos_recall_scale
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
            # v17.4 FIX: Cap context_weight (same as word_index)
            raw_weight = context_weight_factor ** (k - 1)
            context_weight = min(raw_weight, 16)
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
