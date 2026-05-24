"""
Multi-Timescale Reservoir — Emergent long-range coherence via learned couplings.

In a real spin glass, macro-spin structure EMERGES from the couplings J_ij,
not from hand-coded rules. This module implements the physics correctly:
instead of telling the model what "entities" or "phases" are, we give it
multiple timescales and let it LEARN what persists at each scale.

Architecture:
  Three (or more) Integer ESN reservoirs with different decay constants α:
    Fast:   α=0.85  →  ξ ≈ 10 tokens   (local n-gram-like patterns)
    Medium: α=0.95  →  ξ ≈ 50 tokens   (paragraph-level coherence)
    Slow:   α=0.997 →  ξ ≈ 500 tokens  (document-level structure)

  Each reservoir has its OWN readout matrix R, learned from training data.
  The model discovers what matters at each timescale through the readout
  weights — no hand-coded entity lists, no phase triggers, no scene keywords.

  Energy contribution:
    E_mtr(w) = E_fast(w) + E_medium(w) + E_slow(w)

  where each E_k(w) = -(h_k(t) · R_k[w]) * scale_k / norm_k

Key insight: At α=0.997, information from position 0 has 30% influence
at position 400 (0.997^400 ≈ 0.30). This gives correlation length
ξ >> 400 tokens — the missing ingredient for long-range coherence.

Memory budget (V=2000, D=128 per reservoir):
  - W_in: 3 × (128 × 2000 × 1 byte) = 768 KB
  - R:    3 × (2000 × 128 × 2 bytes) = 1.5 MB
  Total: ~2.3 MB (trivial on Pi 5)

This is TRUE spin glass physics: multi-timescale dynamics with learned
couplings. No hand-coded rules. No domain-specific knowledge. Just
the natural emergence of slow modes from the data.
"""

import numpy as np
from typing import Dict, List, Optional


class MultiTimescaleReservoir:
    """
    Multi-timescale reservoir for emergent long-range coherence.

    Instead of hand-coded macro-spin rules, this uses multiple reservoirs
    at different timescales. The slow reservoir maintains information across
    400+ tokens (α=0.997), giving correlation length ξ >> 400.

    All readout matrices are LEARNED from training data. The model
    discovers what persists at each timescale — entities, themes,
    narrative structure — through the readout weights, not through
    hand-coded rules.

    All arithmetic is integer-only.
    """

    # Default timescale configurations: (name, alpha_q15, dim, scale)
    # alpha_q15 = alpha * 32768
    TIMESCALES = [
        ("fast",   27853, 128, 400),   # α≈0.85, ξ≈10,  captures local patterns
        ("medium", 31130, 128, 600),   # α≈0.95, ξ≈50,  captures paragraph structure
        ("slow",   32667, 128, 1000),  # α≈0.997, ξ≈500, captures document structure
    ]

    INT16_MIN = -32768
    INT16_MAX = 32767
    Q15 = 32768
    COUNT_NORM_Q = 256   # Q8 for count normalization
    DOT_NORM_Q = 1024    # Q10 for dot product normalization
    DOT_FLOOR = 100

    def __init__(
        self,
        vocab_size: int,
        timescales: Optional[List[tuple]] = None,
        seed: int = 42,
    ):
        """
        Initialize Multi-Timescale Reservoir.

        Args:
            vocab_size: Vocabulary size V.
            timescales: List of (name, alpha_q15, dim, scale) tuples.
                Default: fast/medium/slow as defined above.
            seed: Random seed for deterministic weight generation.
        """
        self.vocab_size = vocab_size
        self.timescales = timescales or self.TIMESCALES
        self.seed = seed

        # Create per-timescale state
        self.n_scales = len(self.timescales)
        self.names = [ts[0] for ts in self.timescales]
        self.alphas_q15 = [ts[1] for ts in self.timescales]
        self.dims = [ts[2] for ts in self.timescales]
        self.scales = [ts[3] for ts in self.timescales]

        # Input weight matrices: W_in[k] shape (dim_k, V) int8
        # Sparse ternary {-1, 0, +1}
        rng = np.random.RandomState(seed)
        self.W_in = []
        n_input_cols = min(vocab_size, 50000)
        for k in range(self.n_scales):
            D = self.dims[k]
            W = rng.choice([-1, 0, 1], size=(D, n_input_cols), p=[0.33, 0.34, 0.33]).astype(np.int8)
            self.W_in.append(W)

        # Reservoir states: h[k] shape (dim_k,) int16
        self.h = [np.zeros(self.dims[k], dtype=np.int16) for k in range(self.n_scales)]

        # Readout matrices: R[k] shape (V, dim_k) int16
        # Learned from training data
        self.R: List[Optional[np.ndarray]] = [None] * self.n_scales
        self._word_counts: List[Optional[np.ndarray]] = [None] * self.n_scales

        # Whether readouts have been built
        self._built = False

    def reset(self) -> None:
        """Reset all reservoir states for a new document."""
        for k in range(self.n_scales):
            self.h[k].fill(0)

    def step(self, word_id: int) -> None:
        """
        Advance all reservoir states by one token.

        h_k(t) = clip(α_k * h_k(t-1) >> 15 + W_in_k[:, w_t], -2^15, 2^15)

        The different α values create different timescales:
        - Fast (α≈0.85): quick adaptation, short memory
        - Slow (α≈0.997): slow adaptation, long memory (~500 tokens)

        Args:
            word_id: Integer token ID of the current word.
        """
        for k in range(self.n_scales):
            if word_id < 0:
                continue

            alpha = self.alphas_q15[k]
            D = self.dims[k]

            # Decay: alpha * h >> 15
            decayed = (alpha * self.h[k].astype(np.int32)) >> 15

            # Input: W_in[:, word_id]
            if 0 <= word_id < self.W_in[k].shape[1]:
                input_vec = self.W_in[k][:, word_id]
            else:
                input_vec = np.zeros(D, dtype=np.int8)

            # Update and clip
            updated = decayed + input_vec.astype(np.int32)
            self.h[k] = np.clip(updated, self.INT16_MIN, self.INT16_MAX).astype(np.int16)

    def build(
        self,
        sequences: List[List[int]],
        max_sequences: Optional[int] = None,
    ) -> "MultiTimescaleReservoir":
        """
        Build all readout matrices from training sequences.

        For each timescale k, we run reservoir k forward through all
        training sequences and accumulate the reservoir state for each
        target word:

            R_k_sum[w] += h_k(t) for all positions where seq[t] = w

        After accumulation, normalize by word count:
            R_k[w] = R_k_sum[w] * Q8 / max(1, count[w])

        The readout matrix captures what the reservoir state looks like
        BEFORE each word. During generation, words whose readout vectors
        match the current reservoir state get lower energy (more likely).

        At different timescales, the readout captures different patterns:
        - Fast: "after recent words X,Y,Z, word W typically follows"
        - Slow: "in a document about X, with the long-range context Y,
                word W is likely"

        This is ENTIRELY LEARNED from data. No hand-coded rules.

        Args:
            sequences: List of training sequences (lists of word IDs).
            max_sequences: Cap on number of sequences (None = all).

        Returns:
            self
        """
        V = self.vocab_size
        n_seqs = len(sequences)
        if max_sequences is not None:
            n_seqs = min(n_seqs, max_sequences)

        # Build readout for each timescale
        for k in range(self.n_scales):
            D = self.dims[k]
            name = self.names[k]

            R_sum = np.zeros((V, D), dtype=np.int32)
            word_counts = np.zeros(V, dtype=np.int32)

            print(f"    Building MTR readout [{name}] (α_q15={self.alphas_q15[k]}, "
                  f"D={D}, {n_seqs} seqs)...")

            for seq_idx in range(n_seqs):
                seq = sequences[seq_idx]
                self._reset_scale(k)  # Reset only this scale

                for pos in range(len(seq)):
                    word_id = seq[pos]
                    if pos > 0 and 0 <= word_id < V:
                        R_sum[word_id] += self.h[k].astype(np.int32)
                        word_counts[word_id] += 1

                    # Advance this reservoir
                    self._step_scale(k, word_id)

            # Count-normalize: R[w] = R_sum[w] * Q8 / max(1, count[w])
            counts_safe = np.maximum(word_counts, 1)[:, np.newaxis]
            normalized = (R_sum * self.COUNT_NORM_Q) // counts_safe
            R_norm = np.clip(normalized, self.INT16_MIN, self.INT16_MAX).astype(np.int16)
            zero_mask = word_counts == 0
            R_norm[zero_mask] = 0

            self.R[k] = R_norm
            self._word_counts[k] = word_counts

            n_nonzero = int(np.sum(word_counts > 0))
            mem_kb = R_norm.nbytes / 1024
            print(f"      [{name}] {n_nonzero} words with features, "
                  f"memory={mem_kb:.1f} KB")

        self._built = True

        # Reset all states after building
        self.reset()

        return self

    def _reset_scale(self, k: int) -> None:
        """Reset a single reservoir scale."""
        self.h[k].fill(0)

    def _step_scale(self, k: int, word_id: int) -> None:
        """Advance a single reservoir scale."""
        if word_id < 0:
            return

        alpha = self.alphas_q15[k]
        D = self.dims[k]

        decayed = (alpha * self.h[k].astype(np.int32)) >> 15
        if 0 <= word_id < self.W_in[k].shape[1]:
            input_vec = self.W_in[k][:, word_id]
        else:
            input_vec = np.zeros(D, dtype=np.int8)

        updated = decayed + input_vec.astype(np.int32)
        self.h[k] = np.clip(updated, self.INT16_MIN, self.INT16_MAX).astype(np.int16)

    def compute_energy(
        self,
        candidate_words: np.ndarray,
    ) -> np.ndarray:
        """
        Compute combined multi-timescale reservoir energy for candidates.

        E_mtr(w) = Σ_k  -(h_k(t) · R_k[w]) * scale_k / max(1, max_abs_dot_k)

        Each timescale votes independently. Words that match the slow
        reservoir's state get a long-range bias (document coherence).
        Words that match the fast reservoir get a local bias (n-gram-like).

        The scales control the relative strength:
        - Fast:   400  (weaker — local patterns already covered by n-grams)
        - Medium: 600  (moderate — paragraph-level coherence)
        - Slow:  1000  (strong — document-level coherence, the key innovation)

        Args:
            candidate_words: Array of candidate word IDs.

        Returns:
            np.ndarray of int64 energies, shape (n_candidates,).
            LOWER energy = more likely under multi-timescale coupling.
        """
        n_candidates = len(candidate_words)
        if not self._built:
            return np.zeros(n_candidates, dtype=np.int64)

        total_energy = np.zeros(n_candidates, dtype=np.int64)
        safe_candidates = np.clip(candidate_words, 0, self.vocab_size - 1)

        for k in range(self.n_scales):
            if self.R[k] is None:
                continue

            scale = self.scales[k]

            # Look up readout vectors for candidates
            R_candidates = self.R[k][safe_candidates]  # (n, D_k) int16

            # Dot product: h_k · R_k[w] for each candidate
            dots = R_candidates.astype(np.int32) @ self.h[k].astype(np.int32)

            # Normalize to Q10
            max_abs_dot = max(self.DOT_FLOOR, int(np.max(np.abs(dots))))
            norm_dots = (dots.astype(np.int64) * self.DOT_NORM_Q) // max_abs_dot

            # Energy = -(norm_dot * scale) / Q
            energies = -(norm_dots * scale) // self.DOT_NORM_Q

            total_energy += energies.astype(np.int64)

        return total_energy

    @property
    def built(self) -> bool:
        """Whether all readout matrices have been built."""
        return self._built

    def get_diagnostics(self) -> Dict:
        """Get diagnostic information about multi-timescale state."""
        result = {"n_scales": self.n_scales, "built": self._built}
        for k in range(self.n_scales):
            name = self.names[k]
            alpha_float = self.alphas_q15[k] / self.Q15
            # Effective memory: tokens until influence drops to 5%
            import math
            if alpha_float > 0 and alpha_float < 1:
                xi = math.log(0.05) / math.log(alpha_float)
            else:
                xi = float('inf')
            result[name] = {
                "alpha": round(alpha_float, 4),
                "alpha_q15": self.alphas_q15[k],
                "dim": self.dims[k],
                "scale": self.scales[k],
                "xi_95pct": round(xi, 0),
                "h_norm": int(np.linalg.norm(self.h[k].astype(np.float32))),
                "n_words": int(np.sum(self._word_counts[k] > 0)) if self._word_counts[k] is not None else 0,
            }
        return result
