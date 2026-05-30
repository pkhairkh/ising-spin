"""
VSA Permutation Binding Context — Compositional Word-Order Representation.

Implements Vector Symbolic Architecture (VSA) binding via permutation:
  bind(a, hash(b)) = a XOR rotate(a, hash(b))
  unbind(a, hash(b)) = rotate(a, D - hash(b))

This encodes word ORDER into the DAM context by composing each word's
SDR with its positional role. The resulting M_bind is used for:
  1. Attractor dynamics context (augmenting the DAM's context field)
  2. Binding energy bonus (separate from DAM energy — v45 reverted)

Key design decisions (v52):
  - Positional VSA context encoding is PRIMARY (in DAM training)
  - Binding context is SECONDARY (runtime order signal only)
  - M_bind uses attractor dynamics, NOT DAM energy (v45 reverted)
  - Recency weighting was removed (hurt PPL)

ALL INTEGER ARITHMETIC. Runs on Raspberry Pi 5.
"""

import numpy as np
from collections import deque
from typing import Optional


class BindingContext:
    """
    VSA permutation binding context for compositional word-order encoding.

    Each word's active bits are bound with a hash of the previous word,
    creating a superposition of bigram-level bindings. Multi-step unbinding
    recovers partial order information for energy computation.

    Args:
        D: SDR dimension (typically 512).
        k: Number of active bits per SDR (typically 10).
        window: Number of recent bigram bindings to maintain.
        bind_weight: Energy weight for binding bonus.
        n_unbind_words: Number of recent words for multi-step unbinding.
        target_density: Target density of M_bind in active bits (0=auto=2*k).
    """

    def __init__(
        self,
        D: int = 512,
        k: int = 10,
        window: int = 8,
        bind_weight: int = 15,
        n_unbind_words: int = 3,
        target_density: int = 0,
    ):
        self.D = D
        self.k = k
        self.window = window
        self.bind_weight = bind_weight
        self.n_unbind_words = n_unbind_words

        # Auto-compute target density: 2*k gives ~20 active bits in M_bind
        if target_density <= 0:
            self.target_density = 2 * k  # auto
        else:
            self.target_density = target_density

        # M_bind: the binding memory SDR — superposition of all bound pairs
        self.M_bind = np.zeros(D, dtype=np.int32)

        # Recent word active bits (deque of arrays) for multi-step unbinding
        self._recent_words: deque = deque(maxlen=window)

        # Precompute rotation lookup: for each hash value h in [0, D-1],
        # roll_indices[h] is the array of indices such that
        # rolled[i] = original[(i - h) % D]
        # This enables fast vectorized rotation via fancy indexing.
        self._roll_indices = np.zeros((D, D), dtype=np.int64)
        for h in range(D):
            self._roll_indices[h] = np.arange(D)

        # kWTA threshold for sparsifying M_bind after each update
        self._kwta_threshold = 0  # Will be set dynamically

        self._step_count = 0

    def reset(self) -> None:
        """Clear all binding state."""
        self.M_bind[:] = 0
        self._recent_words.clear()
        self._step_count = 0

    def add_word(self, active_bits: np.ndarray) -> None:
        """
        Add a word to the binding context.

        The word is bound with the hash of the previous word (if any),
        and the resulting binding is superposed into M_bind.

        Args:
            active_bits: Array of active bit indices for the word (k elements).
        """
        if len(active_bits) == 0:
            return

        # Compute hash of the word: sum of active bits mod D
        word_hash = int(np.sum(active_bits)) % self.D

        # Create a sparse SDR for the word
        word_sdr = np.zeros(self.D, dtype=np.int32)
        word_sdr[active_bits] = 1

        # Bind with previous word's hash (if we have a previous word)
        if len(self._recent_words) > 0:
            prev_bits = self._recent_words[-1]
            prev_hash = int(np.sum(prev_bits)) % self.D

            # VSA permutation binding: bind(a, h) = a XOR rotate(a, h)
            # This creates a representation that encodes BOTH words
            bound = self._bind(word_sdr, prev_hash)
        else:
            # First word: no binding, just use the word SDR
            bound = word_sdr

        # Superpose into M_bind
        self.M_bind += bound.astype(np.int32)

        # kWTA sparsification: keep only the top-target_density values
        # This prevents M_bind from becoming too dense over time
        self._sparsify_bind()

        # Store the word's active bits for future binding
        self._recent_words.append(active_bits.copy())
        self._step_count += 1

    def _bind(self, sdr: np.ndarray, hash_val: int) -> np.ndarray:
        """
        VSA permutation binding: bind(a, h) = a XOR rotate(a, h).

        The rotation shifts the SDR by hash_val positions, then XOR
        combines it with the original. This creates a distributed
        representation that approximately preserves similarity.

        Args:
            sdr: Binary SDR vector (D,) int32.
            hash_val: Rotation amount (hash of the binding key).

        Returns:
            Bound SDR vector (D,) int32.
        """
        # Rotate the SDR by hash_val positions
        rotated = np.roll(sdr, hash_val)

        # XOR binding: this is the VSA bind operation
        bound = sdr ^ rotated

        return bound

    def _unbind(self, sdr: np.ndarray, hash_val: int) -> np.ndarray:
        """
        VSA permutation unbinding: unbind(a, h) = rotate(a, D - h).

        This is the inverse of binding: rotating back by the same
        amount recovers the original signal (approximately).

        Args:
            sdr: Binary SDR vector (D,) int32.
            hash_val: Rotation amount (hash of the binding key).

        Returns:
            Unbound SDR vector (D,) int32.
        """
        # Unbinding: rotate in the opposite direction
        return np.roll(sdr, self.D - hash_val)

    def _sparsify_bind(self) -> None:
        """
        kWTA sparsification of M_bind.

        Keeps only the top `target_density` values, setting the rest to 0.
        This prevents M_bind from becoming too dense as more bindings
        are superposed.
        """
        # Find the threshold: keep the top target_density values
        nonzero_vals = self.M_bind[self.M_bind > 0]
        if len(nonzero_vals) <= self.target_density:
            return  # Already sparse enough

        # Sort and find the threshold
        sorted_vals = np.sort(nonzero_vals)[::-1]  # descending
        threshold = int(sorted_vals[min(self.target_density, len(sorted_vals) - 1)])

        if threshold <= 0:
            return

        # Zero out everything below threshold
        self.M_bind[self.M_bind < threshold] = 0

    def compute_binding_energy(
        self,
        candidate_words: np.ndarray,
        sdr_encoder,
    ) -> np.ndarray:
        """
        Compute binding energy bonus for each candidate word.

        Multi-step unbinding: for each of the n_unbind_words most recent
        words, unbind M_bind with that word's hash, then compute overlap
        with the candidate word's SDR. Higher overlap → more negative
        energy → more likely.

        This provides ORDER-SENSITIVE scoring: words that are consistent
        with the recent binding context get an energy bonus.

        Args:
            candidate_words: Array of candidate word indices.
            sdr_encoder: SDREncoder for converting word indices to SDRs.

        Returns:
            Binding energy array (n_candidates,) int64.
        """
        if len(self._recent_words) == 0 or np.sum(self.M_bind) == 0:
            return np.zeros(len(candidate_words), dtype=np.int64)

        n_candidates = len(candidate_words)
        total_bind_energy = np.zeros(n_candidates, dtype=np.int64)

        # Multi-step unbinding: check against n_unbind_words recent contexts
        recent_list = list(self._recent_words)
        n_unbind = min(self.n_unbind_words, len(recent_list))

        for step in range(n_unbind):
            # Get the hash for this step's word
            idx = len(recent_list) - 1 - step
            if idx < 0:
                break
            word_bits = recent_list[idx]
            word_hash = int(np.sum(word_bits)) % self.D

            # Unbind M_bind with this word's hash
            unbound = self._unbind(self.M_bind, word_hash)

            # Compute overlap of unbound M_bind with each candidate's SDR
            for i, w_idx in enumerate(candidate_words):
                w_idx = int(w_idx)
                if w_idx < 0 or w_idx >= sdr_encoder.vocab_size:
                    continue

                # Get candidate word's SDR
                candidate_sdr = sdr_encoder.encode(w_idx)

                # Overlap: sum of element-wise product
                # Higher overlap → word is consistent with binding context
                overlap = int(np.sum(unbound * candidate_sdr))

                # Energy bonus: negative (more likely) for higher overlap
                # Recency decay: more recent words get stronger signal
                decay = max(1, n_unbind - step)
                total_bind_energy[i] -= overlap * self.bind_weight * decay // n_unbind

        return total_bind_energy

    def get_context_or(self, context_sdr: np.ndarray) -> np.ndarray:
        """
        Return M_bind if it has content, otherwise return the given context SDR.

        This is used for attractor dynamics: M_bind augments the context
        field when there's binding information, but falls back to the
        standard context when M_bind is empty (e.g., at the start).

        Args:
            context_sdr: Fallback context SDR (D,) int32.

        Returns:
            Context SDR for attractor dynamics (D,) int32.
        """
        if np.sum(self.M_bind) > 0:
            # Combine M_bind with the context SDR
            combined = context_sdr.copy()
            # Add binding bits (union)
            bind_mask = self.M_bind > 0
            combined[bind_mask] = 1
            return combined
        else:
            return context_sdr

    def get_diagnostics(self) -> dict:
        """Return diagnostics for the binding context."""
        m_nnz = int(np.sum(self.M_bind > 0))
        m_max = int(np.max(self.M_bind)) if m_nnz > 0 else 0
        m_sum = int(np.sum(self.M_bind))

        return {
            'step_count': self._step_count,
            'M_bind_nnz': m_nnz,
            'M_bind_max': m_max,
            'M_bind_sum': m_sum,
            'M_bind_density': m_nnz / self.D if self.D > 0 else 0.0,
            'target_density': self.target_density,
            'recent_words': len(self._recent_words),
            'window': self.window,
            'bind_weight': self.bind_weight,
            'n_unbind_words': self.n_unbind_words,
        }
