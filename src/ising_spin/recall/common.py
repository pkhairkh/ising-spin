"""
Common base class for all n-gram recall indexes.

``BaseNgramIndex`` consolidates ~800 lines of near-identical code that was
previously duplicated across ``WordNgramIndex``, ``PosNgramIndex``, and
``TopicNgramIndex``.  The shared logic includes:

* Index building (single-pass and batched with OOM monitoring)
* Pruning of low-count continuations
* Finalisation (Kneser-Ney continuation counts, unigram totals, summary)
* Lookup with sentence-boundary truncation
* Energy computation (KN backoff, unigram backoff, context-weighted match)

Subclasses specialise the *key-extraction* step — i.e. how raw word-ID
sequences are mapped to context keys and continuation IDs — by implementing
four abstract methods.
"""

from __future__ import annotations

import math
import time
from abc import abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import AbstractRecallIndex
from ..errors import IndexBuildError, ValidationError
from ..sampling.boltzmann import int_log2_fine
from ..shared.boundaries import find_effective_start, truncate_at_sentence_boundary
from ..shared.constants import SENT_TOKEN_IDX, SPECIAL_TOKEN_COUNT
from ..shared.memory import get_rss_mb


# ───────────────────────────────────────────────────────────────────────
# Base class
# ───────────────────────────────────────────────────────────────────────


class BaseNgramIndex(AbstractRecallIndex):
    """
    Common base for all n-gram recall indexes.

    Subclasses must implement:

      * ``_build_context_key(seq, t, k) -> tuple``
            Extract the context key for n-gram order *k* at position *t*.
      * ``_build_continuation(seq, t) -> int``
            Extract the continuation word ID at position *t*.
      * ``_should_skip_context(context_key, k) -> bool``
            Whether to skip an invalid context (default: ``False``).
      * ``_context_to_lookup_keys(context_ids) -> list``
            Convert context word IDs to the key space used in the index
            (identity for word-level, POS tags, topic IDs, …).

    Subclasses may also override:

      * ``_prepare_for_build()``              — pre-build validation / setup
      * ``_build_unigram_totals()``           — custom unigram-totals logic
      * ``_higher_order_threshold`` (class)   — k ≥ this uses stricter pruning
      * ``_label`` (class)                    — prefix for log messages
    """

    # ── Class-level configuration (override in subclasses) ────────────

    #: N-gram orders k ≥ this value use ``_higher_order_min`` during pruning.
    _higher_order_threshold: int = 4

    #: Label prepended to log messages (e.g. ``"[POS]"``, ``"[TOPIC]"``).
    _label: str = ""

    # ── Constructor ───────────────────────────────────────────────────

    def __init__(self, max_n: int = 5, min_count: int = 1) -> None:
        if max_n < 1:
            raise ValidationError(f"max_n must be ≥ 1, got {max_n}")
        if min_count < 1:
            raise ValidationError(f"min_count must be ≥ 1, got {min_count}")

        self.max_n: int = max_n
        self.min_count: int = min_count

        # {k: {context_tuple: Counter({word_id: count})}}
        self.index: Dict[int, Dict[Tuple, Counter]] = {
            k: {} for k in range(1, max_n + 1)
        }
        # {k: {context_tuple: total_count}}
        self.context_totals: Dict[int, Dict[Tuple, int]] = {
            k: {} for k in range(1, max_n + 1)
        }
        self._built: bool = False
        self._higher_order_min: int = min_count

        # Populated by _finalize_index()
        self._unigram_totals: Dict[int, Tuple[int, int]] = {}
        self._kn_continuation: Dict[int, Dict[int, int]] = {}
        self._kn_totals: Dict[int, int] = {}
        self._kn_discount: int = 3
        self._kn_discount_fp: int = 12

    # ═══════════════════════════════════════════════════════════════════
    # Abstract methods — subclass hooks
    # ═══════════════════════════════════════════════════════════════════

    @abstractmethod
    def _build_context_key(self, seq: List[int], t: int, k: int) -> Tuple:
        """Return the context key tuple for n-gram order *k* at position *t*.

        Args:
            seq: Original word-ID sequence.
            t:   Current position (the continuation position).
            k:   N-gram order (1 = one context token, 2 = two, …).

        Returns:
            Hashable tuple serving as the context key in ``self.index[k]``.
        """

    @abstractmethod
    def _build_continuation(self, seq: List[int], t: int) -> int:
        """Return the continuation word ID at position *t*.

        For all current indexes this is simply ``seq[t]``, but the hook
        allows subclasses to transform or filter if needed.
        """

    def _should_skip_context(self, context_key: Tuple, k: int) -> bool:
        """Return ``True`` if *context_key* should be skipped at order *k*.

        Default implementation never skips.  Override in subclasses to
        reject invalid contexts (e.g. too many unknown POS tags).
        """
        return False

    @abstractmethod
    def _context_to_lookup_keys(self, context_ids: List[int]) -> List[int]:
        """Convert context word IDs to the key space used by the index.

        For ``WordNgramIndex`` this is the identity mapping.
        For ``PosNgramIndex`` it maps each word ID to its POS tag.
        For ``TopicNgramIndex`` it maps each word ID to its topic ID.
        """

    # ═══════════════════════════════════════════════════════════════════
    # Optional hooks — override in subclasses as needed
    # ═══════════════════════════════════════════════════════════════════

    def _prepare_for_build(self) -> None:
        """Run subclass-specific setup / validation before the build loop.

        Called once at the start of :meth:`build` and :meth:`build_batched`.
        Raise :class:`~ising_spin.errors.ValidationError` or
        :class:`~ising_spin.errors.IndexBuildError` on failure.
        """

    def _build_unigram_totals(self) -> None:
        """Populate ``self._unigram_totals`` after the index is built.

        Default implementation aggregates continuation-word counts across
        all 1-gram contexts (works for POS and Topic indexes).
        ``WordNgramIndex`` may override to use context-word counts instead.
        """
        self._unigram_totals = {}
        if 1 not in self.index:
            return
        total_N = sum(self.context_totals[1].values())
        for _context, continuations in self.index[1].items():
            for w, count in continuations.items():
                if w not in self._unigram_totals:
                    self._unigram_totals[w] = (0, total_N)
                existing_count, _ = self._unigram_totals[w]
                self._unigram_totals[w] = (existing_count + count, total_N)

    # ═══════════════════════════════════════════════════════════════════
    # Building
    # ═══════════════════════════════════════════════════════════════════

    def build(self, sequences: List[List[int]], **kwargs) -> None:
        """Build n-gram index from tokenised sequences.

        Iterates over every position in every sequence, extracts context
        keys and continuation IDs via the abstract hooks, and accumulates
        counts.  After counting, prunes low-count entries and finalises
        KN / unigram statistics.

        Args:
            sequences: List of tokenised sequences (word-ID lists).
            **kwargs:  Subclass-specific parameters (e.g. ``word_pos_tags``).

        Raises:
            ValidationError: If *sequences* is empty.
            IndexBuildError: If subclass pre-build validation fails.
        """
        if not sequences:
            raise ValidationError("sequences must not be empty")

        self._prepare_for_build()

        NS = SPECIAL_TOKEN_COUNT
        for seq in sequences:
            if not seq:
                continue
            eff_start = find_effective_start(seq, NS)
            for t in range(eff_start, len(seq)):
                continuation = self._build_continuation(seq, t)
                if continuation < NS:
                    if seq[t] == SENT_TOKEN_IDX:
                        eff_start = t + 1
                    continue
                for k in range(1, self.max_n + 1):
                    if t - k < eff_start:
                        break
                    context_key = self._build_context_key(seq, t, k)
                    if self._should_skip_context(context_key, k):
                        continue
                    if context_key not in self.index[k]:
                        self.index[k][context_key] = Counter()
                    self.index[k][context_key][continuation] += 1
                    self.context_totals[k][context_key] = (
                        self.context_totals[k].get(context_key, 0) + 1
                    )

        self._prune_index(self.min_count)
        self._built = True
        self._finalize_index()

    def build_batched(
        self,
        sequences: List[List[int]],
        batch_size: int = 200_000,
        prune_interval: int = 1,
        adaptive_min_count: bool = True,
        **kwargs,
    ) -> None:
        """Memory-efficient n-gram index building with batched processing.

        Designed for large corpora (>500 K sequences) where the standard
        :meth:`build` would exhaust memory.  Key features:

        1. Processes sequences in batches — limits peak n-gram dict size.
        2. Prunes low-count entries after each batch — frees memory early.
        3. Auto-scales ``min_count`` with corpus size for larger corpora.
        4. Calls ``gc.collect()`` after each batch — returns memory to OS.
        5. Monitors RSS and triggers aggressive pruning when memory is high.

        Args:
            sequences:          List of tokenised sequences.
            batch_size:         Number of sequences per batch.
            prune_interval:     Prune every N batches (default 1).
            adaptive_min_count: Auto-scale min_count for large corpora.
            **kwargs:           Subclass-specific parameters.

        Raises:
            ValidationError: If *sequences* is empty.
            IndexBuildError: If subclass pre-build validation fails.
        """
        import gc

        if not sequences:
            raise ValidationError("sequences must not be empty")

        self._prepare_for_build()

        total_seqs = len(sequences)

        # ── Auto-scale min_count for large corpora ────────────────────
        effective_min_count = self.min_count
        if adaptive_min_count and total_seqs > 500_000:
            scale = max(
                self.min_count,
                min(5, int(math.log2(total_seqs / 500_000)) + self.min_count),
            )
            effective_min_count = scale
            self._higher_order_min = effective_min_count + 1
        else:
            self._higher_order_min = effective_min_count

        label = self._label
        if effective_min_count != self.min_count:
            print(
                f"    {label} Auto-scaled min_count: "
                f"{self.min_count} -> {effective_min_count} "
                f"(corpus: {total_seqs:,} seqs, "
                f"higher-order: {self._higher_order_min})"
            )

        n_batches = (total_seqs + batch_size - 1) // batch_size
        processed = 0
        t_start = time.time()

        NS = SPECIAL_TOKEN_COUNT

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total_seqs)
            batch = sequences[start:end]

            # ── Count n-grams for this batch ──────────────────────────
            for seq in batch:
                if not seq:
                    continue
                s_start = find_effective_start(seq, NS)
                eff_start = s_start
                for t in range(s_start, len(seq)):
                    continuation = self._build_continuation(seq, t)
                    if continuation < NS:
                        if seq[t] == SENT_TOKEN_IDX:
                            eff_start = t + 1
                        continue
                    for k in range(1, self.max_n + 1):
                        if t - k < eff_start:
                            break
                        context_key = self._build_context_key(seq, t, k)
                        if self._should_skip_context(context_key, k):
                            continue
                        if context_key not in self.index[k]:
                            self.index[k][context_key] = Counter()
                        self.index[k][context_key][continuation] += 1
                        self.context_totals[k][context_key] = (
                            self.context_totals[k].get(context_key, 0) + 1
                        )

            processed += len(batch)

            # ── Prune after each batch ────────────────────────────────
            if (batch_idx + 1) % prune_interval == 0:
                self._prune_index(effective_min_count)
                gc.collect()

            # ── Progress reporting with memory tracking ───────────────
            n_ctx = sum(len(self.index[k]) for k in range(1, self.max_n + 1))
            n_cont = sum(
                sum(len(v) for v in self.index[k].values())
                for k in range(1, self.max_n + 1)
            )
            rss = get_rss_mb()
            elapsed = time.time() - t_start
            mem_info = f", RSS={rss:,}MB" if rss > 0 else ""
            print(
                f"    {label} Batch {batch_idx + 1}/{n_batches}: "
                f"{processed:,} seqs, {n_ctx:,} contexts, "
                f"{n_cont:,} continuations{mem_info} ({elapsed:.1f}s)"
            )

            # ── OOM early warning ─────────────────────────────────────
            if rss > 12_000:
                print(
                    f"    {label} WARNING HIGH MEMORY "
                    f"({rss:,}MB) -- aggressive pruning..."
                )
                self._prune_index(effective_min_count + 2)
                gc.collect()
                rss_after = get_rss_mb()
                print(f"    {label} After aggressive prune: {rss_after:,}MB")

        # Final prune with the base min_count
        self._prune_index(self.min_count)
        self._built = True
        self._finalize_index()

    # ═══════════════════════════════════════════════════════════════════
    # Pruning
    # ═══════════════════════════════════════════════════════════════════

    def _prune_index(self, min_count: int) -> None:
        """Prune low-count n-gram entries from the index.

        Higher-order n-grams (k ≥ :attr:`_higher_order_threshold`) use
        the stricter ``_higher_order_min`` count threshold.
        """
        hot = self._higher_order_threshold
        for k in range(1, self.max_n + 1):
            mc = (
                getattr(self, "_higher_order_min", min_count)
                if k >= hot
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

    # ═══════════════════════════════════════════════════════════════════
    # Finalisation
    # ═══════════════════════════════════════════════════════════════════

    def _finalize_index(self) -> None:
        """Build unigram totals and KN continuation counts after indexing.

        Subclasses that need custom unigram-totals logic should override
        :meth:`_build_unigram_totals` rather than this method.
        """
        # ── Unigram totals (delegated to hook) ────────────────────────
        self._build_unigram_totals()

        # ── Kneser-Ney continuation counts ────────────────────────────
        # N1+(.w) = number of DISTINCT contexts that precede w at each
        # n-gram level.
        self._kn_continuation: Dict[int, Dict[int, int]] = {}
        for k in range(2, self.max_n + 1):
            cont_count: Counter = Counter()
            for _context, continuations in self.index[k].items():
                for w in continuations:
                    cont_count[w] += 1
            self._kn_continuation[k] = dict(cont_count)

        # Total number of distinct (context, word) pairs per level
        self._kn_totals: Dict[int, int] = {}
        for k, cont_count in self._kn_continuation.items():
            self._kn_totals[k] = sum(cont_count.values())

        # Fixed discount for modified Kneser-Ney
        self._kn_discount = 3  # ~0.75 * 4
        self._kn_discount_fp = 12  # 0.75 * 16

        # ── Summary ───────────────────────────────────────────────────
        label = self._label
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
                    f"    {label} {k}-gram: {n_ctx:,} contexts, "
                    f"{n_cont:,} continuations{kn_info}"
                )

    # ═══════════════════════════════════════════════════════════════════
    # Lookup
    # ═══════════════════════════════════════════════════════════════════

    def lookup(
        self, context_ids: List[int]
    ) -> Dict[int, List[Tuple[int, int, int]]]:
        """Look up n-gram continuations for the given context.

        The context is first truncated at the last ``<S>`` token to
        prevent cross-sentence lookups, then converted to the index's
        key space via :meth:`_context_to_lookup_keys`.

        Args:
            context_ids: Context word IDs (may contain ``<S>``).

        Returns:
            ``{k: [(word_id, count, total), …]}`` for each matching
            n-gram level *k*, ordered from most to least common.
        """
        context_ids = truncate_at_sentence_boundary(context_ids)
        lookup_keys = self._context_to_lookup_keys(context_ids)

        results: Dict[int, List[Tuple[int, int, int]]] = {}
        for k in range(min(self.max_n, len(lookup_keys)), 0, -1):
            context_tuple = tuple(lookup_keys[-k:])
            if context_tuple in self.index[k]:
                total = self.context_totals[k][context_tuple]
                conts = self.index[k][context_tuple].most_common()
                results[k] = [(word, count, total) for word, count in conts]
        return results

    # ═══════════════════════════════════════════════════════════════════
    # Energy computation
    # ═══════════════════════════════════════════════════════════════════

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
        """Compute recall ENERGY for candidate words based on n-gram matches.

        Uses precise ratio + Kneser-Ney backoff:

        * ``log2(total) - log2(count)`` instead of ``log2(total // count)``
          — eliminates integer-division loss (up to 0.4 bits/token gain).
        * KN backoff: when no n-gram match, uses continuation counts
          ``N1+(.w)`` instead of raw unigram P(w).
        * Interpolated smoothing: all n-gram levels vote (product of
          experts).

        Returns **positive** energy values where **lower** energy = more
        likely.  The calling code uses ``energies += recall_energy``.

        Args:
            context_ids:         Context word IDs (converted internally).
            candidate_words:     Array of candidate word IDs to score.
            recall_scale:        Energy scale factor.
            context_weight_factor: Exponential weight per context length.
            longest_only:        Use only the longest matching n-gram.
            interpolated:        Combine all n-gram levels (best energy).
            kn_backoff:          Use KN continuation counts for backoff.
            **kwargs:            Ignored (for subclass compatibility).

        Returns:
            ``np.ndarray`` of int64 energies, shape ``(len(candidate_words),)``.
        """
        if len(candidate_words) == 0:
            return np.array([], dtype=np.int64)

        n_candidates = len(candidate_words)
        max_energy = 20 * recall_scale
        recall_energies = np.full(n_candidates, max_energy, dtype=np.int64)

        # ── Backoff energy (KN or unigram) ────────────────────────────
        if self._built:
            self._compute_backoff_energy(
                candidate_words, recall_energies, recall_scale, kn_backoff
            )

        # ── Context lookup + match application ────────────────────────
        matches = self.lookup(context_ids)
        if not matches:
            return recall_energies

        if longest_only and not interpolated and matches:
            best_k = max(matches.keys())
            matches = {best_k: matches[best_k]}

        self._apply_matches(
            matches,
            candidate_words,
            recall_energies,
            recall_scale,
            context_weight_factor,
            interpolated,
            max_energy,
        )

        return recall_energies

    # ── Backoff helpers ────────────────────────────────────────────────

    def _compute_backoff_energy(
        self,
        candidate_words: np.ndarray,
        recall_energies: np.ndarray,
        recall_scale: int,
        kn_backoff: bool,
    ) -> None:
        """Fill *recall_energies* with backoff values (mutates in-place).

        Kneser-Ney backoff uses continuation counts and is scaled 2×
        higher than n-gram match energy to create a clear gap between
        "matched" and "backed off" words.  Falls back to unigram
        statistics when KN is unavailable or disabled.
        """
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

    # ── Match application ─────────────────────────────────────────────

    @staticmethod
    def _apply_matches(
        matches: Dict[int, List[Tuple[int, int, int]]],
        candidate_words: np.ndarray,
        recall_energies: np.ndarray,
        recall_scale: int,
        context_weight_factor: int,
        interpolated: bool,
        max_energy: int,
    ) -> None:
        """Apply n-gram match energies to *recall_energies* (mutates in-place).

        For each matching n-gram level *k*:

        1. Compute a context weight ``min(factor^(k-1), 16)``.
        2. For each continuation word, compute
           ``energy = log2(total/count) * scale * weight``.
        3. Assign the lowest (most likely) energy to each candidate.

        With *interpolated* = True, each level can only **lower** the
        existing energy (Jelinek-Mercer-like product of experts).
        Without interpolation, the last level wins outright.
        """
        for k, continuations in matches.items():
            # v17.4 FIX: Cap context_weight to prevent exponential energy
            # blowup.  With factor=2, a 10-gram match got 2^9=512× the
            # base scale.  Cap at 2^4=16 so high-order matches still get
            # significant weight but don't overwhelm the energy landscape.
            raw_weight = context_weight_factor ** (k - 1)
            context_weight = min(raw_weight, 16)

            cont_lookup: Dict[int, int] = {}
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
                w_int = int(w)
                if w_int in cont_lookup:
                    if interpolated:
                        # Interpolated smoothing — take BEST (lowest)
                        # energy across all levels
                        if cont_lookup[w_int] < recall_energies[i]:
                            recall_energies[i] = cont_lookup[w_int]
                    else:
                        recall_energies[i] = cont_lookup[w_int]
