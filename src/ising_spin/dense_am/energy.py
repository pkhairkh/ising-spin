"""
Dense Associative Memory energy with random feature pre-aggregation.

Architecture:
  1. RandomFeatureProjector: Maps a token context (list of word IDs) to a
     D=256 dimensional integer feature vector using random projections with
     cosine nonlinearity (Random Fourier Features approximation).

  2. DenseAMEnergy: Pre-aggregates feature vectors per word during training,
     then computes energy as a single D-dim dot product per candidate at
     inference. A polynomial nonlinearity F(x) controls energy sharpness:
       - degree=1: F(x) = x  (linear, standard Hopfield)
       - degree=2: F(x) = x^2 (Dense AM, sharper basins, capacity ~N)

Key insight: Standard Hopfield networks have capacity ~0.14N patterns.
Dense AM with F(x) = x^2 increases capacity to ~N patterns by creating
much sharper energy basins. This means:
  - Correct completions get MUCH lower energy
  - Incorrect completions get MUCH higher energy
  - The energy landscape has deeper, more separated wells

Random feature trick: Instead of storing all N patterns and computing
similarity with each one (O(N*D)), pre-aggregate per word:
  Phi(w) = sum of phi(ctx_mu) for all training contexts where w follows ctx_mu
Then energy is a single dot product: E(w) = -F(phi(ctx) . Phi(w))

Memory budget: Phi is (V, D) int16 = ~25 MB for V=49K, D=256.
Computation: 256 integer multiply-accumulates per candidate word.
All integer, no float operations in hot path.
"""

import numpy as np
from typing import Optional, List


class RandomFeatureProjector:
    """
    Maps a token context to a D-dimensional integer feature vector.

    Uses random projections with cosine nonlinearity (Random Fourier Features).
    The feature map approximates a shift-invariant kernel:
        k(x, y) ~ phi(x) . phi(y)

    Algorithm:
      1. Look up a random hash vector for each context word from a
         precomputed table (vocab_size x context_hash_dim, int8)
      2. Superpose (sum) these hash vectors with position-dependent weights
         to create a context_hash vector of dimension context_hash_dim
      3. Project through a fixed random integer matrix W (D x context_hash_dim)
      4. Add random bias b (D,)
      5. Apply cosine via 256-entry LUT -> phi(ctx) in int8^D

    The cosine nonlinearity is crucial: it makes the random features
    approximate a Gaussian kernel, which gives good similarity structure
    even for inputs that differ in only a few positions.

    All integer arithmetic. The cos LUT uses math.cos() during __init__
    (one-time setup) — NO float operations in the hot path (project()).
    """

    # Context window for hashing (last K tokens)
    CONTEXT_WINDOW = 10

    def __init__(
        self,
        vocab_size: int,
        D: int = 256,
        context_hash_dim: int = 32,
        seed: int = 42,
    ):
        """
        Initialize the random feature projector.

        Args:
            vocab_size: Number of words in vocabulary (V).
            D: Feature vector dimension (default 256).
            context_hash_dim: Intermediate hash dimension (default 32).
            seed: Random seed for deterministic generation.
        """
        self.vocab_size = vocab_size
        self.D = D
        self.context_hash_dim = context_hash_dim
        self.seed = seed

        rng = np.random.RandomState(seed)

        # Word hash vectors: (min(vocab_size, 50000), context_hash_dim) int8
        # Each word gets a random sparse vector for context hashing.
        # Sparse ternary {-1, 0, +1} keeps the hash compact and fast.
        n_hash_rows = min(vocab_size, 50000)
        self.word_hashes = rng.choice(
            [-1, 0, 1],
            size=(n_hash_rows, context_hash_dim),
            p=[0.33, 0.34, 0.33],
        ).astype(np.int8)

        # Random projection matrix: (D, context_hash_dim) int8
        # Sparse ternary for efficient integer matrix-vector multiply.
        self.W = rng.choice(
            [-1, 0, 1],
            size=(D, context_hash_dim),
            p=[0.33, 0.34, 0.33],
        ).astype(np.int8)

        # Random bias: (D,) uint8
        self.b = rng.randint(0, 256, size=D, dtype=np.uint8)

        # Cosine lookup table (init-time only, uses math.cos)
        self._cos_lut = self._build_cos_lut()

    def _build_cos_lut(self) -> np.ndarray:
        """
        Build 256-entry cosine lookup table.

        Maps phase index i in [0, 255] to cos(2*pi*i/256) * 127,
        clipped to int8 range [-127, 127].

        Properties:
            LUT[0]   = 127  (cos(0) = 1)
            LUT[64]  ~ 0    (cos(pi/2) = 0)
            LUT[128] = -127 (cos(pi) = -1)
            LUT[192] ~ 0    (cos(3*pi/2) = 0)

        Note: Uses math.cos() during INITIALIZATION ONLY.
        The LUT is then used at inference time via integer lookup.

        Returns:
            np.ndarray of shape (256,) with int8 values.
        """
        import math

        lut = np.zeros(256, dtype=np.int8)
        for i in range(256):
            val = round(math.cos(2 * math.pi * i / 256) * 127)
            lut[i] = max(-127, min(127, val))

        return lut

    def project(self, context_words: List[int]) -> np.ndarray:
        """
        Map a list of context word IDs to a D-dimensional feature vector.

        This is the core random feature computation. It maps a variable-length
        context to a FIXED-SIZE feature vector, enabling kernel-like similarity
        comparisons via simple dot products.

        Steps:
          1. Superpose word hash vectors with position weighting
          2. Project through random matrix W
          3. Add random bias b
          4. Apply cosine via LUT -> int8 feature vector

        Args:
            context_words: List of word IDs forming the context.

        Returns:
            np.ndarray of shape (D,) with dtype int8, values in [-127, 127].
        """
        if not context_words:
            return np.zeros(self.D, dtype=np.int8)

        # --- Step 1: Compute context hash ---
        # Superpose word hashes with position-dependent weights.
        # More recent words get higher weight, giving the features
        # a recency bias that matches linguistic locality.
        h = np.zeros(self.context_hash_dim, dtype=np.int32)
        recent = context_words[-self.CONTEXT_WINDOW:]

        for pos, w_id in enumerate(recent):
            if 0 <= w_id < len(self.word_hashes):
                # Position weight: 1 for oldest, CONTEXT_WINDOW for newest
                weight = pos + 1
                h += self.word_hashes[w_id].astype(np.int32) * weight

        # Clip to int8 range for matrix multiply
        h_clipped = np.clip(h, -127, 127).astype(np.int8)

        # --- Step 2: Random projection W @ h ---
        # (D, context_hash_dim) @ (context_hash_dim,) -> (D,)
        # int8 @ int8 -> int32 (each element: sum of ~32 products of ~1*~1)
        raw = self.W.astype(np.int32) @ h_clipped.astype(np.int32)  # (D,) int32

        # --- Step 3: Add random bias ---
        raw += self.b.astype(np.int32)

        # --- Step 4: Apply cosine via LUT ---
        # Map raw values to [0, 255] for LUT indexing (modular arithmetic)
        indices = (raw % 256).astype(np.uint8)
        phi = self._cos_lut[indices]  # (D,) int8

        return phi

    @property
    def cos_lut(self) -> np.ndarray:
        """Access the cosine lookup table for testing."""
        return self._cos_lut


class DenseAMEnergy:
    """
    Dense Associative Memory energy using random feature pre-aggregation.

    The Dense AM replaces the linear energy function E(x) = x with a
    polynomial nonlinearity E(x) = F(x), where F(x) = x^degree.

    For degree=1: Standard linear energy (like Hopfield network).
      Capacity ~ 0.14N, shallow energy basins.

    For degree=2: Dense AM energy (quadratic sharpening).
      Capacity ~ N, much sharper energy basins. The quadratic
      nonlinearity amplifies differences: good matches get MUCH lower
      energy, bad matches get MUCH higher energy. This is the key
      expressivity gain.

    Pre-aggregation: During training, we compute:
      Phi(w) = sum of phi(ctx_mu) for all (ctx_mu, w) pairs

    This is normalized by word count to give the MEAN feature vector
    per word (Q8 fixed-point), preventing frequent words from dominating.

    Inference: For each candidate word w:
      E_dense_am(w) = -F(phi(ctx) . Phi(w))

    where phi(ctx) is the random feature projection of the current context,
    and the dot product phi . Phi is a single D-dimensional inner product.

    Memory: Phi is (V, D) int16 ~ 25 MB for V=49K, D=256.
    Computation: D=256 integer multiply-accumulates per candidate.
    """

    # Q-format for count normalization (multiply by 256 before divide)
    # This keeps the mean feature vector in int16 range.
    COUNT_NORM_Q = 256

    # Normalization Q-format for dot products (Q10 = 1024)
    DOT_NORM_Q = 1024

    # Floor for max_abs_dot to prevent amplifying noise
    DOT_FLOOR = 100

    # Maximum energy value (Q30)
    MAX_Q30 = (1 << 30) - 1

    def __init__(
        self,
        projector: RandomFeatureProjector,
        vocab_size: int,
        degree: int = 2,
        dense_am_scale: int = 1200,
    ):
        """
        Initialize Dense AM energy module.

        Args:
            projector: RandomFeatureProjector instance.
            vocab_size: Number of words in vocabulary (V).
            degree: Polynomial degree for nonlinearity (default 2).
                    degree=1: linear (standard), degree=2: Dense AM (sharp).
            dense_am_scale: Energy scaling factor (default 1200).
                           Controls the magnitude of Dense AM energy relative
                           to other energy terms (recall_scale=1600, vsa_scale=800).
        """
        self.projector = projector
        self.vocab_size = vocab_size
        self.degree = degree
        self.dense_am_scale = dense_am_scale

        # Pre-aggregated readout matrix: (V, D) int16
        # Phi[w] = Q8 * mean(phi(ctx)) over all contexts where w follows
        self.Phi: Optional[np.ndarray] = None

        # Word counts for normalization
        self._word_counts: Optional[np.ndarray] = None

        # Whether pre-aggregation has been done
        self._built = False

    def preaggregate(
        self,
        sequences: List[List[int]],
        max_sequences: Optional[int] = None,
    ) -> "DenseAMEnergy":
        """
        Build the Phi matrix from training sequences.

        For each (context, target) pair in training:
          Phi[target] += phi(context)

        After accumulation, normalize by word count to get the MEAN
        feature vector per word (Q8 fixed-point). This prevents
        frequent words from having enormous Phi values that would
        dominate the dot products.

        The count normalization is:
          Phi_norm[w] = Phi_sum[w] * 256 // max(1, count[w])

        This gives Q8 * mean(phi) in int16 range. For a word seen K
        times with int8 phi values in [-127, 127]:
          |Phi_sum| <= K * 127
          |Phi_norm| <= K * 127 * 256 / K = 32512 -> fits in int16

        Args:
            sequences: List of training sequences (lists of word IDs).
            max_sequences: Cap on number of sequences to process (None = all).

        Returns:
            self
        """
        V = self.vocab_size
        D = self.projector.D

        # Accumulate in int32 (safe for up to ~16K additions of int8 values)
        Phi_sum = np.zeros((V, D), dtype=np.int32)
        word_counts = np.zeros(V, dtype=np.int32)

        n_seqs = len(sequences)
        if max_sequences is not None:
            n_seqs = min(n_seqs, max_sequences)

        print(f"    Pre-aggregating Dense AM Phi matrix ({n_seqs} sequences, D={D})...")

        for seq_idx in range(n_seqs):
            seq = sequences[seq_idx]
            for pos in range(1, len(seq)):
                context = seq[:pos]
                target = seq[pos]
                if 0 <= target < V:
                    phi = self.projector.project(context)  # (D,) int8
                    Phi_sum[target] += phi.astype(np.int32)
                    word_counts[target] += 1

        # Count-normalize: Phi_norm = Phi_sum * Q / max(1, count)
        # This gives Q8 * mean(phi) per word
        Phi_norm = np.zeros((V, D), dtype=np.int16)
        for w in range(V):
            if word_counts[w] > 0:
                # Q8 normalization: multiply by 256, then divide by count
                normalized = Phi_sum[w] * self.COUNT_NORM_Q // word_counts[w]
                Phi_norm[w] = np.clip(normalized, -32768, 32767).astype(np.int16)

        self.Phi = Phi_norm
        self._word_counts = word_counts
        self._built = True

        mem_mb = self.Phi.nbytes / (1024 * 1024)
        n_nonzero = int(np.sum(word_counts > 0))
        print(f"    Dense AM Phi: shape={self.Phi.shape}, dtype=int16, "
              f"memory={mem_mb:.1f} MB, {n_nonzero} words with features")

        return self

    def compute_energy(
        self,
        context_words: List[int],
        candidate_words: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Dense AM energy for candidate words.

        E_dense_am(w) = -F(phi(ctx) . Phi[w])

        where:
          - phi(ctx) is the D-dimensional random feature vector for the context
          - Phi[w] is the pre-aggregated feature vector for word w
          - F is the polynomial nonlinearity (degree=1: linear, degree=2: quadratic)

        The dot product is normalized to Q10 fixed point before applying F,
        which keeps the energy in a predictable range regardless of the
        raw dot product magnitude. This makes the dense_am_scale parameter
        meaningful and comparable across different training sets.

        Normalization:
          norm_dot = dot * Q10 / max(1, max_abs_dot)
          Then: degree=1: E = -(norm_dot * scale) / Q10
                degree=2: E = -(norm_dot^2 * scale) / Q10^2

        This ensures energies are always in range [-scale, 0] approximately,
        with the best candidate getting energy closest to -scale and the
        worst getting energy closest to 0.

        Args:
            context_words: List of context word IDs.
            candidate_words: Array of candidate word IDs.

        Returns:
            np.ndarray of int64 energies, shape (len(candidate_words),).
            LOWER energy = better match = more likely under Boltzmann.
        """
        n_candidates = len(candidate_words)

        if not self._built or self.Phi is None:
            return np.zeros(n_candidates, dtype=np.int64)

        # Project current context to feature space
        phi_ctx = self.projector.project(context_words)  # (D,) int8

        # Look up pre-aggregated Phi for candidates
        safe_candidates = np.clip(candidate_words, 0, self.vocab_size - 1)
        Phi_candidates = self.Phi[safe_candidates]  # (n, D) int16

        # Dot product: phi_ctx . Phi[w] for each candidate
        # int8 . int16 -> int32 per element, sum over D dimensions
        dots = Phi_candidates.astype(np.int32) @ phi_ctx.astype(np.int32)  # (n,) int32

        # Normalize dot products to Q10 fixed point
        # This prevents the raw dot product magnitude from causing overflow
        # when applying the polynomial nonlinearity.
        max_abs_dot = max(self.DOT_FLOOR, int(np.max(np.abs(dots))))
        Q = self.DOT_NORM_Q  # 1024

        norm_dots = (dots.astype(np.int64) * Q) // max_abs_dot  # (n,) int64

        # Apply polynomial nonlinearity F and compute energy
        if self.degree == 1:
            # Linear: F(x) = x
            # E = -(norm_dot * dense_am_scale) / Q
            energies = -(norm_dots * self.dense_am_scale) // Q
        else:
            # Dense AM: F(x) = x^2 (quadratic sharpening)
            # norm_dots are in Q10, so norm_dots^2 is in Q20
            F_x = norm_dots * norm_dots  # int64, Q20

            # Clip to Q30 range to prevent overflow
            F_x = np.clip(F_x, -self.MAX_Q30, self.MAX_Q30)

            # E = -(F_x * dense_am_scale) / Q^2
            # Q^2 = 1024^2 = 1,048,576
            Q_squared = Q * Q
            energies = -(F_x * self.dense_am_scale) // Q_squared

        return energies.astype(np.int64)

    @property
    def built(self) -> bool:
        """Whether the Phi matrix has been pre-aggregated."""
        return self._built

    @property
    def word_counts(self) -> Optional[np.ndarray]:
        """Access word counts from pre-aggregation."""
        return self._word_counts
