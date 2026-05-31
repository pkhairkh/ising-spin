"""
Pure integer bigram language model.

The base model for the Integer Language Model. No neural nets, no torch,
no float32 matrix multiplies. Just integer counting and Laplace smoothing.

For a V=2000 word vocabulary, the full bigram matrix is 2000x2000 int32
= 16 MB — trivial on any modern device including a Pi 5.

Generation: P(word | prev) = (count[prev][word] + alpha) / (total[prev] + alpha*V)
This is the unigram-bigram backed-off model that serves as the foundation
for the energy-guided decoding in IntegerLM.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


class BigramModel:
    """
    Pure integer bigram language model.

    Stores:
      - bigram_counts[i][j]: number of times word j follows word i
      - unigram_counts[j]: number of times word j appears
      - row_totals[i]: sum of bigram_counts[i] (for fast normalization)

    Uses Laplace (add-alpha) smoothing for unseen bigrams.
    Falls back to unigram distribution when bigram count is zero.
    """

    def __init__(
        self,
        vocab_size: int,
        smoothing_alpha: float = 1.0,
        log_prob_scale: int = 100,
        seed: int = 42,
    ):
        """
        Args:
            vocab_size: Number of words in vocabulary.
            smoothing_alpha: Laplace smoothing parameter. 1.0 = standard Laplace.
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

        self._built = True
        return self

    def get_top_k(
        self,
        context_ids: List[int],
        k: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get top-K candidate words with log-probabilities.

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
            # No context: use unigram distribution
            return self._unigram_top_k(k)

        prev = context_ids[-1]
        if prev < 0 or prev >= self.V:
            return self._unigram_top_k(k)

        # Compute log-prob for each word given prev
        total = self.row_totals[prev]
        if total == 0:
            return self._unigram_top_k(k)

        # Laplace-smoothed probabilities
        probs = (self.bigram_counts[prev].astype(np.float64) + self.alpha) / \
                (total + self.alpha * self.V)

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
        total = max(1, int(self.unigram_counts.sum()))
        probs = (self.unigram_counts.astype(np.float64) + self.alpha) / \
                (total + self.alpha * self.V)

        k_actual = min(k, self.V)
        top_indices = np.argsort(probs)[-k_actual:][::-1]
        top_probs = probs[top_indices]

        log_probs = np.log(np.maximum(top_probs, 1e-300))

        valid_mask = top_indices >= 4
        if not np.any(valid_mask):
            # Fallback: return any non-special tokens
            candidates = np.arange(4, min(4 + k, self.V))
            lp = np.log(1.0 / max(1, self.V - 4))
            log_probs_arr = np.full(len(candidates), lp)
            return candidates, log_probs_arr

        return top_indices[valid_mask], log_probs[valid_mask]

    def compute_log_prob(self, context_ids: List[int], target: int) -> float:
        """
        Compute log P(target | context) for perplexity evaluation.

        Not in the generation hot path — uses float for accuracy.
        """
        if not self._built:
            return -10.0

        if len(context_ids) == 0:
            total = max(1, int(self.unigram_counts.sum()))
            p = (self.unigram_counts[target] + self.alpha) / (total + self.alpha * self.V)
            return float(np.log(max(p, 1e-300)))

        prev = context_ids[-1]
        if prev < 0 or prev >= self.V or target < 0 or target >= self.V:
            return -10.0

        total = self.row_totals[prev]
        if total == 0:
            # Fall back to unigram
            u_total = max(1, int(self.unigram_counts.sum()))
            p = (self.unigram_counts[target] + self.alpha) / (u_total + self.alpha * self.V)
            return float(np.log(max(p, 1e-300)))

        p = (self.bigram_counts[prev, target] + self.alpha) / (total + self.alpha * self.V)
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
