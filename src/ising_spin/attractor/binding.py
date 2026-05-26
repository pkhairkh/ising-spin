"""
VSA Compositional Binding — INTEGER ONLY, NO hand-coded grammar roles.

Implements data-dependent, non-commutative binding for order-sensitive
composition. The machine discovers its own compositional structure
through attractor dynamics — no POS tags, no role labels needed.

BINDING: bind(a, b) = rot(a, hash(b))
  - a is the "content" word (the one being encoded)
  - b is the "key" word (determines the permutation)
  - hash(b) = sum(active_bits_of_b) mod D — full [0, D-1] spread
  - Result has SAME density as input (k=10 bits) — no density explosion
  - Non-commutative: bind(a, b) != bind(b, a) because hash(a) != hash(b)
    in general (different SDRs produce different rotations)

UNBINDING: unbind(bound, b) = rot(bound, D - hash(b))
  - EXACT inverse: unbind(bind(a, b), b) = a
  - No noise, no approximation — just reverse the rotation
  - Query "what came after word b?" -> unbind(M_bind, b) has high overlap
    with words that were bound as content with b as key

CONTEXT VECTOR: M_bind = OR-superposition of recent bigram bindings + kWTA
  - Encodes bigram ORDER: (w0->w1), (w1->w2), (w2->w3), ...
  - For bigram (prev->current): bind(sdr[current], sdr[prev])
    Unbinding with sdr[prev] reveals sdr[current] — the word that FOLLOWED prev
  - kWTA sparsification keeps density at ~2k = 20 bits for DAM compatibility
  - Sliding window of W recent bindings prevents unbounded growth

ENERGY BONUS:
  For each candidate word c, unbind M_bind with the last N words
  (default N=3 for multi-step unbinding) and sum overlaps:
    bonus = sum over w in recent_words: overlap(sdr[c], unbind(M_bind, w)) * BIND_WEIGHT
  Added as negative bonus (lower energy = more likely).
  Multi-word unbinding gives richer context beyond bigrams —
  trigram and longer patterns emerge from the binding structure.

WHY PERMUTATION (not XOR) BINDING:
  The original proposal used a XOR rot(a, hash(b)), which:
    - Produces ~2k active bits (density explosion with OR-superposition)
    - Is NOT trivially invertible (self-referential unbinding formula)
    - Requires cleanup memory even for single-pair retrieval
  Permutation binding rot(a, hash(b)) is cleaner:
    - Same density as input (k=10, not 2k=20)
    - EXACTLY invertible: unbind = rotate back
    - OR-superposition + kWTA is well-defined
    - Standard VSA approach (Plate 1995, Kanerva 2009)

WHY SUM-OF-POSITIONS HASH (not popcount):
  popcount(b AND MASK) with k=10 active bits gives rotation range [0, 10].
  This is nearly commutative — most word pairs would rotate by the same
  amount, defeating the purpose. Sum-of-positions gives range [0, D-1]
  with excellent spread for D=512.

  Example: SDR with active bits {3, 47, 128, 256, 300, 400, 450, 480, 490, 500}
    sum = 3054, hash = 3054 % 512 = 498
  Different SDRs produce different hashes because their active bit positions
  are drawn from different regions of the random projection space.

INTEGRATION WITH DAM:
  - v39: M_bind is NOT OR'd into context SDR for DAM energy computation.
    The DAM was trained on standard context SDRs (word superposition + kWTA).
    Injecting binding bits that the DAM never saw adds noise to coupling energy.
    Instead, M_bind is used ONLY for the binding energy bonus (separate signal).
  - M_bind IS still used for attractor dynamics (step_all) context field
    to help the attractor settle toward binding-consistent states.
  - Binding energy bonus is added as a separate term in total energies
  - Beta calibration automatically includes binding energy
  - RG flow and UV checks are UNCHANGED — binding is a runtime overlay

COMPUTATIONAL OVERHEAD (on Pi 5):
  - Per word: 1 hash (sum of k=10 values) + 1 rotation (k modular adds)
  - Per candidate: 1 overlap (k popcount) + 1 multiply
  - Total: ~100 integer ops per candidate — negligible vs DAM dot product

Based on:
  Plate (1995): Holographic Reduced Representations — permutation binding
  Kanerva (2009): Hyperdimensional Computing — VSA foundations
  Gallant & Okaywe (2013): Representing verbs with permutation + superposition
  Rachkovskij (2015): Binary Sparse Distributed Representations — binding by shift

ALL INTEGER ARITHMETIC. Runs on Raspberry Pi 5.
"""

import numpy as np
from typing import Optional
from collections import deque


def sdr_hash(active_bits: np.ndarray, D: int) -> int:
    """Compute rotation amount from SDR content.

    Uses sum of active bit positions modulo D.
    This gives full range [0, D-1] for rotation, unlike
    popcount(b AND MASK) which only gives [0, k] with k=10.

    For k=10 active bits in D=512:
      - Minimum sum: 0+1+2+...+9 = 45  -> 45 % 512 = 45
      - Maximum sum: 503+504+...+511 = 4585  -> 4585 % 512 = 497
      - Good spread across [0, 511] — nearly uniform

    Different words produce different rotation amounts because
    their SDRs have different active bit positions (determined
    by the deterministic random projection + kWTA encoding).

    Args:
        active_bits: Array of active bit indices (k values).
        D: SDR dimension.

    Returns:
        Rotation amount in [0, D-1].
    """
    return int(np.sum(active_bits.astype(np.int64))) % D


def rotate_sdr(active_bits: np.ndarray, shift: int, D: int) -> np.ndarray:
    """Cyclic rotation of SDR active bit indices.

    rot(v, k): each active bit position i -> (i + shift) % D
    This is equivalent to cyclic bit rotation of the dense vector.

    Inverse: rotate_sdr(result, D - shift, D) undoes the rotation.

    All integer — just modular arithmetic on k=10 values.

    Args:
        active_bits: Array of active bit indices (k values).
        shift: Rotation amount (positive = right shift in index space).
        D: SDR dimension.

    Returns:
        Rotated active bit indices (k values).
    """
    shift = shift % D  # Normalize
    return np.array([(int(b) + shift) % D for b in active_bits], dtype=np.int16)


def bind_pair(a_bits: np.ndarray, b_bits: np.ndarray, D: int) -> np.ndarray:
    """VSA binding: bind(a, b) = rot(a, hash(b)).

    Non-commutative: bind(a, b) = rot(a, h(b)) != rot(b, h(a)) = bind(b, a)
    in general, because h(a) != h(b) for different words.

    Result has exactly k active bits (same density as input).
    No density explosion — compatible with existing DAM.

    Semantic: "a in the context of b" or "a followed b"
    To query "what followed b?", unbind with b to get a.

    Args:
        a_bits: Active bit indices of the "content" word (k values).
        b_bits: Active bit indices of the "key" word (k values).
        D: SDR dimension.

    Returns:
        Active bit indices of the bound vector (k values).
    """
    shift = sdr_hash(b_bits, D)
    return rotate_sdr(a_bits, shift, D)


def unbind_pair(bound_bits: np.ndarray, b_bits: np.ndarray, D: int) -> np.ndarray:
    """VSA unbinding: unbind(bound, b) = rot(bound, D - hash(b)).

    Exact inverse: unbind(bind(a, b), b) = a.
    No noise, no approximation — just reverse the rotation.

    This is the key advantage of permutation binding over XOR binding:
    unbinding is an EXACT operation, not an approximate one that
    requires cleanup memory.

    Args:
        bound_bits: Active bit indices of the bound vector.
        b_bits: Active bit indices of the "key" word used in binding.
        D: SDR dimension.

    Returns:
        Active bit indices of the unbound vector (= original a_bits).
    """
    shift = sdr_hash(b_bits, D)
    return rotate_sdr(bound_bits, D - shift, D)


class BindingContext:
    """Maintains compositional context via VSA bigram binding.

    At each time step, the binding context encodes the ORDER-SENSITIVE
    bigram structure of the recent word sequence. This provides:

    1. Word-order sensitivity: "cat sat" != "sat cat"
       because bind(sdr["sat"], sdr["cat"]) != bind(sdr["cat"], sdr["sat"])

    2. Phrase-level attractors: frequent bigrams form stable patterns
       in M_bind. The DAM will learn these as attractors because
       M_bind is OR'd into the context field.

    3. Expectation signal: "what followed the last word?" is queryable
       via unbinding. This gives a primitive but genuine form of
       linguistic expectation beyond raw n-gram statistics.

    The context is maintained as an OR-superposition of recent binding
    pairs, with kWTA sparsification to maintain compatibility with the
    DAM's sparse input expectations.

    Memory overhead: O(W * k) integers where W = window size, k = 10.
    Computational overhead: O(k) per word (rotation + OR + kWTA).
    """

    def __init__(
        self,
        D: int = 512,
        k: int = 10,
        window: int = 8,
        target_density: int = 0,  # 0 = auto (2*k)
        bind_weight: int = 30,
        n_unbind_words: int = 3,
    ):
        """
        Args:
            D: SDR dimension.
            k: Number of active bits per word SDR.
            window: Number of recent bigram bindings to maintain.
                    With window=8, M_bind covers the last 8 bigrams.
            target_density: Target number of active bits in M_bind after kWTA.
                           0 = auto (2*k = 20, ~4% density).
                           Must be >= k and <= D.
            bind_weight: Weight for binding energy bonus (BIND_WEIGHT).
                        Higher = stronger influence of binding context.
                        v39: With LOG2_NORM=512, dE ~ O(200-300),
                        bind_weight=30 gives typical bonus ~60 (20-30% of dE).
                        Formula: bonus = overlap * bind_weight (NO integer div).
            n_unbind_words: Number of recent words to unbind with for
                           multi-step unbinding (default=3 for trigram context).
                           Unbinds with each of the last N words and sums bonuses.
        """
        self.D = D
        self.k = k
        self.window = window
        self.target_density = max(k, target_density if target_density > 0 else 2 * k)
        self.bind_weight = bind_weight
        self.n_unbind_words = n_unbind_words

        # Dense accumulator for M_bind (rebuild from bindings each time)
        self._acc = np.zeros(D, dtype=np.int32)

        # Recent binding pairs for rebuilding after window slides
        self._bindings: deque = deque(maxlen=window)

        # Current M_bind as dense binary vector
        self.M_bind: np.ndarray = np.zeros(D, dtype=np.uint8)

        # Recent word active bits (for multi-step unbinding)
        # Stores up to n_unbind_words recent word SDRs
        self._recent_words: deque = deque(maxlen=n_unbind_words)

        # Last word's active bits (for creating next bigram binding)
        self._last_word_bits: Optional[np.ndarray] = None

        # Statistics
        self._n_bindings = 0

    def reset(self) -> None:
        """Reset binding context for a new sequence."""
        self._acc = np.zeros(self.D, dtype=np.int32)
        self._bindings.clear()
        self._recent_words.clear()
        self.M_bind = np.zeros(self.D, dtype=np.uint8)
        self._last_word_bits = None
        self._n_bindings = 0

    def add_word(self, word_active_bits: np.ndarray) -> None:
        """Add a word to the binding context.

        Creates a bigram binding with the previous word (if any):
          bind(current_word, previous_word) = rot(sdr[current], hash(sdr[prev]))

        Semantic: "current word FOLLOWED previous word"
        To query "what followed prev?", unbind with sdr[prev].

        After adding, rebuilds M_bind from all bindings in the window
        using OR-superposition + kWTA.

        Args:
            word_active_bits: Active bit indices of the new word's SDR.
        """
        if self._last_word_bits is not None:
            # Bind: rot(current_word, hash(previous_word))
            # Convention: a = current word (content), b = previous word (key)
            # Query "what came after prev?" -> unbind(M, prev) gives current
            bound = bind_pair(word_active_bits, self._last_word_bits, self.D)

            # Add to sliding window
            self._bindings.append(bound)
            self._n_bindings += 1

            # Rebuild M_bind from all bindings in window
            self._rebuild()

        # Remember this word for the next bigram AND for multi-step unbinding
        self._last_word_bits = word_active_bits.copy()
        self._recent_words.append(word_active_bits.copy())

    def _rebuild(self) -> None:
        """Rebuild M_bind from current binding window with kWTA.

        OR-superposition: each binding contributes k active bits.
        With W=8 bindings, raw OR has up to 8*k = 80 active bits.
        kWTA reduces to target_density = 2*k = 20 bits (~4% density).

        Bits that appear in multiple bindings get higher accumulator
        values and are preferentially kept by kWTA. This means
        frequently-occurring bigram patterns are preserved even
        under sparsification — emergent phrase-level structure.

        v43: UNIFORM weighting (reverted from v42 recency weighting).
        Recency weighting made M_bind carry only the most recent bigram,
        breaking multi-step unbinding (n_unbind=3). With uniform weights,
        bits shared across multiple bindings survive kWTA, giving useful
        signal when unbinding with ANY of the last N words.
        """
        self._acc = np.zeros(self.D, dtype=np.int32)

        for bound_bits in self._bindings:
            for b in bound_bits:
                idx = int(b)
                if 0 <= idx < self.D:
                    self._acc[idx] += 1

        # kWTA: keep top target_density bits
        if np.max(self._acc) > 0:
            n_active = min(self.target_density, int(np.sum(self._acc > 0)))
            if n_active > 0:
                top_k = np.argpartition(self._acc, -n_active)[-n_active:]
                self.M_bind = np.zeros(self.D, dtype=np.uint8)
                self.M_bind[top_k] = 1
            else:
                self.M_bind = np.zeros(self.D, dtype=np.uint8)
        else:
            self.M_bind = np.zeros(self.D, dtype=np.uint8)

    def compute_binding_energy(
        self,
        candidate_words: np.ndarray,
        sdr_encoder,  # SDREncoder
    ) -> np.ndarray:
        """Compute binding energy bonus for each candidate word.

        v39: Multi-step unbinding — unbind M_bind with each of the last
        N words (n_unbind_words, default=3) and sum the overlap bonuses.
        This gives richer context beyond bigrams: trigram and longer
        patterns emerge from the binding structure.

        For each candidate c and each recent word w:
          overlap(sdr[c], unbind(M_bind, w)) * bind_weight
        Total bonus = sum over all recent words.

        v39: Removed `// 10` integer division that was losing precision.
        Now uses direct `overlap * bind_weight` for full integer precision.

        Higher overlap = lower energy (more likely) = negative bonus.

        Args:
            candidate_words: Array of candidate word IDs.
            sdr_encoder: SDREncoder for word->SDR mapping.

        Returns:
            Energy array (len(candidate_words),) int64.
            Values are <= 0 (binding bonus reduces energy).
        """
        n_cand = len(candidate_words)
        energies = np.zeros(n_cand, dtype=np.int64)

        if len(self._recent_words) == 0 or np.sum(self.M_bind) == 0:
            return energies

        M_bind_active = np.where(self.M_bind > 0)[0]
        if len(M_bind_active) == 0:
            return energies

        # Pre-compute valid candidate mask and SDRs
        valid_mask = (candidate_words >= 0) & (candidate_words < sdr_encoder.vocab_size)
        valid_words = candidate_words[valid_mask]

        if len(valid_words) == 0:
            return energies

        candidate_sdrs = sdr_encoder.word_sdrs[valid_words]  # (n, D) uint8

        # Multi-step unbinding: unbind with each recent word and sum bonuses
        total_overlaps = np.zeros(len(valid_words), dtype=np.int64)

        for word_bits in self._recent_words:
            # Unbind M_bind with this word: rot(M_bind_active, D - hash(word))
            shift = sdr_hash(word_bits, self.D)
            unbound_active = rotate_sdr(M_bind_active, self.D - shift, self.D)

            unbound_dense = np.zeros(self.D, dtype=np.uint8)
            for b in unbound_active:
                idx = int(b)
                if 0 <= idx < self.D:
                    unbound_dense[idx] = 1

            # Compute overlap with each candidate's SDR
            overlaps = candidate_sdrs.astype(np.int32) @ unbound_dense.astype(np.int32)
            total_overlaps += overlaps.astype(np.int64)

        # Energy bonus: negative (lower energy = more likely)
        # v39: Direct `overlap * bind_weight` — NO integer division that loses precision
        binding_bonus = -(total_overlaps * self.bind_weight)
        energies[valid_mask] = binding_bonus.astype(np.int64)

        return energies

    def get_context_or(self, context_sdr: np.ndarray) -> np.ndarray:
        """OR M_bind with the standard context SDR for ATTRACTOR DYNAMICS.

        v39: This is used ONLY for step_all() context field (attractor dynamics),
        NOT for compute_word_energies() DAM energy computation. The DAM was
        trained on standard context SDRs — injecting binding bits that the DAM
        never saw adds noise to the coupling energy.

        The resulting combined context has at most k + target_density
        active bits (e.g., 10 + 20 = 30 for default settings).
        This is still sparse enough for the DAM (~6% density).

        The M_bind bits carry ORDER information that the standard
        context SDR doesn't have (it uses superposition which is
        order-insensitive). This helps attractor dynamics settle
        toward binding-consistent states.

        Args:
            context_sdr: Standard context SDR (D,) uint8.

        Returns:
            Combined context SDR (D,) uint8 = context_sdr | M_bind.
        """
        return (context_sdr | self.M_bind).astype(np.uint8)

    def get_diagnostics(self) -> dict:
        """Return binding context diagnostics.

        v42: Reports BOTH configured parameters and runtime state.
        - window: CONFIGURED capacity (e.g., 8)
        - window_fill: CURRENT number of bindings in deque
        - target_density: CONFIGURED kWTA target (e.g., 20)
        - m_bind_density: CURRENT active bits in M_bind
        """
        return {
            'n_bindings': self._n_bindings,
            'window': self.window,                    # CONFIGURED capacity (8)
            'window_fill': len(self._bindings),       # current fill level
            'n_recent_words': len(self._recent_words),
            'target_density': self.target_density,      # CONFIGURED kWTA target (20)
            'm_bind_density': int(np.sum(self.M_bind)), # current active bits
            'bind_weight': self.bind_weight,
            'n_unbind_words': self.n_unbind_words,
            'has_last_word': self._last_word_bits is not None,
        }
