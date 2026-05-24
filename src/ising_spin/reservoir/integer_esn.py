"""
Integer Echo State Network for long-range temporal dynamics.

Architecture:
  1. IntegerESN: Maintains an int16 reservoir state vector of dimension D=512.
     The state evolves via a leaky integrator with fixed random input weights:
       h(t) = clip(alpha_q15 * h(t-1) >> 15 + W_in[:, w_t], -2^15, 2^15)
     where alpha_q15 ≈ 31130 (≈0.95 in Q15) provides an exponential decay
     with effective memory of ~50 tokens:
       half-life = -ln(2) / ln(alpha) ≈ 0.693 / 0.051 ≈ 13.6 steps
       95% decay: ~43 steps; 99% decay: ~90 steps

  2. Readout matrix R: Pre-aggregated during training by accumulating
     reservoir states h(t-1) for each target word w_t. Normalized by
     word count in Q8 fixed point, stored as int16.
       R[w] = Q8 * mean(h(t) over all positions where w follows h(t))

  3. Energy computation: For each candidate word w:
       E_reservoir(w) = -(h(t) · R[w]) * reservoir_scale / max(1, max_abs_dot)
     Normalized to Q10 before scaling, same pattern as Dense AM.

Key insight: Standard n-gram recall has a HARD window (5 tokens for word,
10 for POS/topic). The ESN provides a SOFT window via exponential decay.
Tokens from 50 positions ago still have ~5% influence on the reservoir state.
This is the "temporal dynamics" that transformers get from self-attention,
but implemented as a fixed random recurrent network with integer arithmetic.

Memory budget:
  - W_in: (512, V) int8 ≈ 25 MB for V=49K
  - R:    (V, 512) int16 ≈ 50 MB for V=49K
  Total: ~75 MB (well within Pi 5's 16 GB RAM)

Computation per token:
  - State update: 512 integer multiply-accumulates (alpha * h + input)
  - Energy: 512 integer multiply-accumulates per candidate (dot product h · R[w])
  Total: ~512 * n_candidates MACs, comparable to Dense AM
"""

import numpy as np
from typing import Optional, List

from ..exceptions import ValidationError


class IntegerESN:
    """
    Integer Echo State Network for long-range temporal dynamics.

    The ESN maintains a fading memory of the entire document history.
    Unlike n-gram recall which only looks at the last 5-10 tokens,
    the ESN's exponential decay provides ~50 token effective lookback
    (with spectral radius alpha ≈ 0.95 in Q15).

    State update (per token):
        h(t) = clip(alpha_q15 * h(t-1) >> 15 + W_in[:, w_t], -2^15, 2^15)

    where:
        - h(t) is the D-dimensional int16 reservoir state
        - alpha_q15 is the decay factor in Q15 (31130 ≈ 0.95)
        - W_in[:, w_t] looks up the input column for word w_t

    The readout matrix R is precomputed via training pre-aggregation:
        R[w] = Q8 * mean(h(t) over all positions where w follows h(t))

    Energy: E_reservoir(w) = -(h(t) · R[w]) * reservoir_scale / norm_factor

    Memory: W_in is (D, V) int8 ≈ 25 MB, R is (V, D) int16 ≈ 50 MB.
    """

    # Q15 fixed-point for alpha (spectral radius)
    Q15 = 32768

    # Default alpha ≈ 0.95 in Q15: 0.95 * 32768 = 31129.6 ≈ 31130
    # This gives effective memory of ~50 tokens:
    #   0.95^n < 0.05 → n > ln(0.05)/ln(0.95) ≈ 58
    #   0.95^n < 0.01 → n > ln(0.01)/ln(0.95) ≈ 90
    DEFAULT_ALPHA_Q15 = 31130

    # Clip range for reservoir state
    INT16_MIN = -32768
    INT16_MAX = 32767

    # Q8 normalization for count-normalized readout
    COUNT_NORM_Q = 256

    # Q10 normalization for dot products (same as Dense AM)
    DOT_NORM_Q = 1024
    DOT_FLOOR = 100
    MAX_Q30 = (1 << 30) - 1

    def __init__(
        self,
        vocab_size: int,
        reservoir_dim: int = 512,
        alpha_q15: int = 31130,
        seed: int = 42,
    ):
        """
        Initialize the Integer ESN.

        Args:
            vocab_size: Number of words in vocabulary (V).
            reservoir_dim: Reservoir state dimension D (default 512).
            alpha_q15: Decay factor in Q15 (default 31130 ≈ 0.95).
                       Controls memory length: higher = longer memory.
            seed: Random seed for deterministic weight generation.
        """
        self.vocab_size = vocab_size
        self.reservoir_dim = reservoir_dim
        self.alpha_q15 = alpha_q15
        self.seed = seed

        # Fixed random input weight matrix: (D, V) int8
        # Sparse ternary {-1, 0, +1} with ~33% each
        # This maps each word to a D-dimensional input signal.
        rng = np.random.RandomState(seed)
        n_input_cols = min(vocab_size, 50000)
        self.W_in = rng.choice(
            [-1, 0, 1],
            size=(reservoir_dim, n_input_cols),
            p=[0.33, 0.34, 0.33],
        ).astype(np.int8)

        # Reservoir state: D-dimensional int16 vector
        self.h = np.zeros(reservoir_dim, dtype=np.int16)

        # Precomputed readout matrix: (V, D) int16
        # R[w] = Q8 * mean(h before w) over all training positions
        self.R: Optional[np.ndarray] = None

        # Word counts from pre-aggregation
        self._word_counts: Optional[np.ndarray] = None

        # Whether readout has been built
        self._built = False

    def reset(self) -> None:
        """
        Reset reservoir state for a new document.

        Sets h(t) = zero vector. Called at the start of each new
        generation or when processing a new training sequence.
        """
        self.h = np.zeros(self.reservoir_dim, dtype=np.int16)

    def step(self, word_id: int) -> np.ndarray:
        """
        Advance reservoir state by one token. Pure integer arithmetic.

        h(t) = clip(alpha_q15 * h(t-1) >> 15 + W_in[:, w_t], -2^15, 2^15)

        The Q15 multiplication ensures the decay is computed in fixed-point:
          alpha_q15 * h >> 15 ≈ 0.95 * h  (within int16 range)

        The input contribution W_in[:, w_t] is a fixed random column
        of sparse ternary values, providing a unique "fingerprint" for
        each word in the reservoir's state space.

        Args:
            word_id: Integer token ID of the current word.

        Returns:
            Updated reservoir state, int16, shape (D,).
        """
        if word_id < 0:
            return self.h.copy()  # Ignore invalid word IDs
        # Decay: alpha_q15 * h(t-1) >> 15
        # h is int16, alpha is int, product fits in int32
        # Shift right by 15 brings it back to int16 scale
        decayed = (self.alpha_q15 * self.h.astype(np.int32)) >> 15  # (D,) int32

        # Input: W_in[:, word_id]
        if 0 <= word_id < self.W_in.shape[1]:
            input_vec = self.W_in[:, word_id]  # (D,) int8
        else:
            input_vec = np.zeros(self.reservoir_dim, dtype=np.int8)

        # Update: decayed + input
        updated = decayed + input_vec.astype(np.int32)

        # Clip to int16 range
        self.h = np.clip(updated, self.INT16_MIN, self.INT16_MAX).astype(np.int16)

        return self.h.copy()

    def build(
        self,
        sequences: List[List[int]],
        max_sequences: Optional[int] = None,
    ) -> "IntegerESN":
        """
        Build the readout matrix R from training sequences.

        For each training sequence, we run the reservoir forward and
        accumulate the reservoir state for each target word:

            R_sum[w] += h(t) for all positions where seq[t] = w

        where h(t) is the reservoir state BEFORE feeding word w
        (i.e., h(t) encodes the context seq[0..t-1]).

        After accumulation, normalize by word count:
            R[w] = R_sum[w] * Q8 / max(1, count[w])

        This gives Q8 * mean(h_before_w) per word in int16 range.
        For a word seen K times with int16 h values in [-32768, 32767]:
            |R_sum| <= K * 32767
            |R_norm| <= K * 32767 * 256 / K = 8,388,352

        Since 8,388,352 > 32767, we need to be more careful. The actual
        R_sum values will typically be much smaller because h values are
        distributed around zero (not all at the extremes). We clip to
        int16 range after normalization.

        Args:
            sequences: List of training sequences (lists of word IDs).
            max_sequences: Cap on number of sequences to process (None = all).

        Returns:
            self
        """
        V = self.vocab_size
        D = self.reservoir_dim

        # Accumulate in int32 (safe for typical training sizes)
        R_sum = np.zeros((V, D), dtype=np.int32)
        word_counts = np.zeros(V, dtype=np.int32)

        n_seqs = len(sequences)
        if max_sequences is not None:
            n_seqs = min(n_seqs, max_sequences)

        print(f"    Building ESN readout ({n_seqs} sequences, D={D})...")

        for seq_idx in range(n_seqs):
            seq = sequences[seq_idx]
            self.reset()  # Reset for each new document

            for pos in range(len(seq)):
                word_id = seq[pos]

                # Record reservoir state BEFORE feeding this word
                # h(t) encodes the context seq[0..pos-1]
                # Skip pos=0 (no context yet, h is zero)
                if pos > 0 and 0 <= word_id < V:
                    R_sum[word_id] += self.h.astype(np.int32)
                    word_counts[word_id] += 1

                # Advance reservoir with this word
                self.step(word_id)

        # Count-normalize: R[w] = R_sum[w] * Q8 / max(1, count[w])
        # Vectorized: avoid Python for-loop over V words
        counts_safe = np.maximum(word_counts, 1)[:, np.newaxis]  # (V, 1)
        normalized = (R_sum * self.COUNT_NORM_Q) // counts_safe   # (V, D) int32
        R_norm = np.clip(normalized, -32768, 32767).astype(np.int16)
        zero_mask = word_counts == 0
        R_norm[zero_mask] = 0

        self.R = R_norm
        self._word_counts = word_counts
        self._built = True

        mem_R = self.R.nbytes / (1024 * 1024)
        mem_W = self.W_in.nbytes / (1024 * 1024)
        n_nonzero = int(np.sum(word_counts > 0))
        print(f"    ESN readout R: shape={self.R.shape}, dtype=int16, "
              f"memory={mem_R:.1f} MB, {n_nonzero} words with features")
        print(f"    ESN input W_in: shape={self.W_in.shape}, dtype=int8, "
              f"memory={mem_W:.1f} MB")

        # Reset state after building
        self.reset()

        return self

    def compute_energy(
        self,
        candidate_words: np.ndarray,
        reservoir_scale: int = 800,
    ) -> np.ndarray:
        """
        Compute reservoir energy for candidate words.

        E_reservoir(w) = -(h(t) · R[w]) * reservoir_scale / max(1, max_abs_dot)

        The dot product h · R[w] measures how well the current reservoir
        state matches the expected pre-word state for candidate w. Words
        that are consistent with the long-range context get lower energy
        (more likely under Boltzmann sampling).

        The normalization by max_abs_dot ensures the energy magnitude
        is controlled by reservoir_scale, regardless of the raw dot
        product magnitude. This follows the same Q10 pattern as Dense AM.

        Args:
            candidate_words: Array of candidate word IDs.
            reservoir_scale: Energy scaling factor (default 800).

        Returns:
            np.ndarray of int64 energies, shape (len(candidate_words),).
            LOWER energy = better match with reservoir state = more likely.
        """
        n_candidates = len(candidate_words)

        if not self._built or self.R is None:
            return np.zeros(n_candidates, dtype=np.int64)

        # Look up readout vectors for candidates
        safe_candidates = np.clip(candidate_words, 0, self.vocab_size - 1)
        R_candidates = self.R[safe_candidates]  # (n, D) int16

        # Dot product: h · R[w] for each candidate
        # int16 · int16 -> int32 per element, sum over D dimensions
        dots = R_candidates.astype(np.int32) @ self.h.astype(np.int32)  # (n,) int32

        # Normalize dot products to Q10 fixed point (same as Dense AM)
        max_abs_dot = max(self.DOT_FLOOR, int(np.max(np.abs(dots))))
        Q = self.DOT_NORM_Q  # 1024

        norm_dots = (dots.astype(np.int64) * Q) // max_abs_dot  # (n,) int64

        # Energy = -(norm_dot * reservoir_scale) / Q
        # This gives: max dot → energy ≈ -reservoir_scale, min dot → energy ≈ 0
        energies = -(norm_dots * reservoir_scale) // Q

        return energies.astype(np.int64)

    @property
    def built(self) -> bool:
        """Whether the readout matrix has been built."""
        return self._built

    @property
    def word_counts(self) -> Optional[np.ndarray]:
        """Access word counts from pre-aggregation."""
        return self._word_counts

    @property
    def state(self) -> np.ndarray:
        """Access the current reservoir state h(t)."""
        return self.h.copy()
