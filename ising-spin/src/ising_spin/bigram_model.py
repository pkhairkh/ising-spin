"""
Pure integer bigram language model with Jelinek-Mercer interpolation.

v90 FIX: The original model used Laplace smoothing with alpha=1.0, which
with V=2000 allocates 2000 pseudo-observations — dominating the probability
for most context words (95%+ of mass is smoothing when row_total < 2000).
This made the base model nearly equivalent to a unigram model (PPL=27.74).

Two fundamental changes:
1. Laplace alpha reduced from 1.0 to 0.01 (2000→20 pseudo-observations)
2. Jelinek-Mercer interpolation: P(w|prev) = λ·P_bigram(w|prev) + (1-λ)·P_unigram(w)
   where λ = total[prev] / (total[prev] + alpha*V) — contexts with more
   observations trust the bigram more; rare contexts back off to unigram.

This gives the energy model a MUCH better base to correct. Expected base PPL
improvement: 27.74 → ~15-18, making the energy model's job far easier.

No neural nets. No torch. No float32 in the hot path. Integer counting + interpolation.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


class BigramModel:
    """
    Pure integer bigram language model with Jelinek-Mercer interpolation.

    Stores:
      - bigram_counts[i][j]: number of times word j follows word i
      - unigram_counts[j]: number of times word j appears
      - row_totals[i]: sum of bigram_counts[i] (for fast normalization)

    Uses Jelinek-Mercer interpolation with Laplace smoothing for unseen bigrams.
    The interpolation weight lambda depends on the context word's total count,
    providing automatic backoff to unigram for rare contexts.
    """

    def __init__(
        self,
        vocab_size: int,
        smoothing_alpha: float = 0.01,
        log_prob_scale: int = 100,
        seed: int = 42,
    ):
        """
        Args:
            vocab_size: Number of words in vocabulary.
            smoothing_alpha: Laplace smoothing parameter. 0.01 = light smoothing.
                             v89 used 1.0 which was catastrophically over-smoothed.
            log_prob_scale: Scale factor for converting log-probs to integer energies.
                           energy = -log_prob * scale
            seed: Random seed.
        """
        self.V = vocab_size
        self.alpha = smoothing_alpha
        self.log_prob_scale = log_prob_scale
        self.seed = seed

        # Bigram count matrix: V x V int32
        # 16 MB for V=2000 — perfectly fine
        self.bigram_counts = np.zeros((vocab_size, vocab_size), dtype=np.int32)
        self.unigram_counts = np.zeros(vocab_size, dtype=np.int32)
        self.row_totals = np.zeros(vocab_size, dtype=np.int64)
        self._built = False

        # Precomputed unigram distribution (set after build)
        self._unigram_total: int = 0
        self._unigram_probs: Optional[np.ndarray] = None

    def build(self, sequences: List[List[int]]) -> "BigramModel":
        """
        Build bigram counts from tokenized sequences.

        This is pure integer counting — no float ops.
        """
        for seq in sequences:
            for pos in range(1, len(seq)):
                prev = seq[pos - 1]
                target = seq[pos]
                if 0 <= prev < self.V and 0 <= target < self.V:
                    self.bigram_counts[prev, target] += 1
                    self.unigram_counts[target] += 1

        # Row totals for fast normalization
        self.row_totals = self.bigram_counts.sum(axis=1).astype(np.int64)

        # Also count first words as unigrams
        for seq in sequences:
            if len(seq) > 0 and 0 <= seq[0] < self.V:
                self.unigram_counts[seq[0]] += 1

        # Precompute unigram distribution
        self._unigram_total = max(1, int(self.unigram_counts.sum()))
        self._unigram_probs = (self.unigram_counts.astype(np.float64) + self.alpha) / \
                              (self._unigram_total + self.alpha * self.V)

        self._built = True
        return self

    def _compute_interpolated_probs(self, prev: int) -> np.ndarray:
        """
        Compute Jelinek-Mercer interpolated probabilities for all words given prev.

        P(w|prev) = λ · P_bigram(w|prev) + (1 - λ) · P_unigram(w)
        where λ = total[prev] / (total[prev] + alpha * V)

        This automatically backs off to unigram for rare contexts.
        """
        total = self.row_totals[prev]

        # Interpolation weight: more observations → trust bigram more
        lam = total / (total + self.alpha * self.V)

        # Bigram probability (Laplace-smoothed)
        bigram_probs = (self.bigram_counts[prev].astype(np.float64) + self.alpha) / \
                       (total + self.alpha * self.V)

        # Interpolation: P = λ * P_bigram + (1-λ) * P_unigram
        probs = lam * bigram_probs + (1.0 - lam) * self._unigram_probs

        return probs

    def get_top_k(
        self,
        context_ids: List[int],
        k: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get top-K candidate words with log-probabilities.

        Uses Jelinek-Mercer interpolation for better probability estimates.

        Args:
            context_ids: List of context word IDs (uses only the last one).
            k: Number of candidates to return.

        Returns:
            Tuple of (candidates, log_probs) where:
              - candidates: np.ndarray of word IDs, shape (K,)
              - log_probs: np.ndarray of log-probabilities, shape (K,)
        """
        if not self._built:
            raise RuntimeError("BigramModel not built — call build() first")

        if len(context_ids) == 0:
            return self._unigram_top_k(k)

        prev = context_ids[-1]
        if prev < 0 or prev >= self.V:
            return self._unigram_top_k(k)

        total = self.row_totals[prev]
        if total == 0:
            return self._unigram_top_k(k)

        # Jelinek-Mercer interpolated probabilities
        probs = self._compute_interpolated_probs(prev)

        # Get top-K by probability
        k_actual = min(k, self.V)
        top_indices = np.argsort(probs)[-k_actual:][::-1]
        top_probs = probs[top_indices]

        # Convert to log-probabilities
        log_probs = np.log(np.maximum(top_probs, 1e-300))

        # Filter out special tokens (IDs 0-3)
        valid_mask = top_indices >= 4
        if not np.any(valid_mask):
            return self._unigram_top_k(k)

        return top_indices[valid_mask], log_probs[valid_mask]

    def _unigram_top_k(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get top-K candidates from unigram distribution (no context)."""
        if self._unigram_probs is not None:
            probs = self._unigram_probs
        else:
            total = max(1, int(self.unigram_counts.sum()))
            probs = (self.unigram_counts.astype(np.float64) + self.alpha) / \
                    (total + self.alpha * self.V)

        k_actual = min(k, self.V)
        top_indices = np.argsort(probs)[-k_actual:][::-1]
        top_probs = probs[top_indices]

        log_probs = np.log(np.maximum(top_probs, 1e-300))

        valid_mask = top_indices >= 4
        if not np.any(valid_mask):
            candidates = np.arange(4, min(4 + k, self.V))
            lp = np.log(1.0 / max(1, self.V - 4))
            log_probs_arr = np.full(len(candidates), lp)
            return candidates, log_probs_arr

        return top_indices[valid_mask], log_probs[valid_mask]

    def compute_log_prob(self, context_ids: List[int], target: int) -> float:
        """
        Compute log P(target | context) for perplexity evaluation.

        Uses Jelinek-Mercer interpolation.
        """
        if not self._built:
            return -10.0

        if target < 0 or target >= self.V:
            return -10.0

        # Unigram probability
        u_p = float(self._unigram_probs[target]) if self._unigram_probs is not None else \
              (self.unigram_counts[target] + self.alpha) / (self._unigram_total + self.alpha * self.V)

        if len(context_ids) == 0:
            return float(np.log(max(u_p, 1e-300)))

        prev = context_ids[-1]
        if prev < 0 or prev >= self.V:
            return float(np.log(max(u_p, 1e-300)))

        total = self.row_totals[prev]
        if total == 0:
            return float(np.log(max(u_p, 1e-300)))

        # Jelinek-Mercer interpolation
        lam = total / (total + self.alpha * self.V)
        bigram_p = (self.bigram_counts[prev, target] + self.alpha) / (total + self.alpha * self.V)
        p = lam * bigram_p + (1.0 - lam) * u_p

        return float(np.log(max(p, 1e-300)))

    def compute_sequence_log_prob(self, ids: List[int]) -> float:
        """Compute total log-probability of a sequence."""
        if len(ids) < 2:
            return 0.0
        total = 0.0
        for pos in range(1, len(ids)):
            total += self.compute_log_prob(ids[:pos], ids[pos])
        return total

    def statistics(self) -> Dict:
        """Return bigram model statistics."""
        if not self._built:
            return {"built": False}

        total_bigrams = int(self.bigram_counts.sum())
        nonzero_bigrams = int(np.count_nonzero(self.bigram_counts))
        possible_bigrams = self.V * self.V

        return {
            "built": True,
            "vocab_size": self.V,
            "total_bigrams": total_bigrams,
            "nonzero_bigrams": nonzero_bigrams,
            "bigram_density": nonzero_bigrams / possible_bigrams,
            "smoothing_alpha": self.alpha,
            "memory_mb": self.bigram_counts.nbytes / (1024 * 1024),
        }
