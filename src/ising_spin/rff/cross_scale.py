"""
Cross-Scale Random Fourier Features for ISG-LM v18.3.

Architecture:
  1. CrossScaleRFF: Maps a multi-scale context (word IDs + POS IDs + topic
     IDs) to a D=256 dimensional integer feature vector using random
     projections with cosine nonlinearity (Random Fourier Features).

  2. Pre-aggregates feature vectors per word during training into a Theta
     matrix, then computes energy as a single D-dim dot product per
     candidate at inference.

Key difference from Dense AM:
  - Dense AM uses ONLY word-level context hashes.
  - Cross-Scale RFF combines word + POS + topic into a SINGLE feature vector.
  - This captures cross-scale interactions that independent scales miss,
    e.g. "NOUN in SPORTS context" vs "NOUN in POLITICS context".

  The combined hash h is:
    h = sum_i weight_i * (word_hashes[w_i] + pos_hashes[p_i] + topic_hashes[t_i])

  This means two contexts that share the same words but differ in POS or
  topic will produce DIFFERENT feature vectors — the random projection
  mixes all three scales, creating joint features that no single scale
  can represent.

Memory budget: Theta is (V, D) int8 = ~12.5 MB for V=49K, D=256.
Computation: 256 integer multiply-accumulates per candidate word.
All integer, no float operations in hot path.
"""

import numpy as np
from typing import Optional, List


class CrossScaleRFF:
    """
    Cross-Scale Random Fourier Feature energy module.

    Combines word, POS, and topic context hashes into a single random
    feature vector, capturing cross-scale interactions that independent
    per-scale energy terms miss.

    Pre-aggregation: During training, compute:
      Theta[w] = sum of phi(ctx_mu) for all (ctx_mu, w) pairs

    Normalized by word count to give the MEAN feature vector per word
    (Q7 fixed-point for int8), preventing frequent words from dominating.

    Inference: For each candidate word w:
      E_rff(w) = -(phi(ctx) . Theta[w]) * rff_scale / Q10

    where phi(ctx) is the cross-scale random feature projection of the
    current context (word+POS+topic), and the dot product is a single
    D-dimensional inner product.

    Memory: Theta is (V, D) int8 ~ 12.5 MB for V=49K, D=256.
    Computation: D=256 integer multiply-accumulates per candidate.
    """

    # Context window for hashing (last K tokens)
    CONTEXT_WINDOW = 10

    # Q-format for count normalization (multiply by 128 before divide)
    # This keeps the mean feature vector in int8 range.
    # For a word seen K times with int8 phi values in [-127, 127]:
    #   |Theta_sum| <= K * 127
    #   |Theta_norm| <= K * 127 * 128 / K = 16256 -> fits in int8 [-127,127]
    # after clipping.
    COUNT_NORM_Q = 128

    # Normalization Q-format for dot products (Q10 = 1024)
    DOT_NORM_Q = 1024

    # Floor for max_abs_dot to prevent amplifying noise
    DOT_FLOOR = 100

    def __init__(
        self,
        vocab_size: int,
        n_pos: int = 13,
        n_topics: int = 16,
        D: int = 256,
        context_hash_dim: int = 32,
        seed: int = 42,
        rff_scale: int = 600,
    ):
        """
        Initialize the Cross-Scale RFF module.

        Args:
            vocab_size: Number of words in vocabulary (V).
            n_pos: Number of POS types (default 13).
            n_topics: Number of topic types (default 16).
            D: Feature vector dimension (default 256).
            context_hash_dim: Intermediate hash dimension (default 32).
            seed: Random seed for deterministic generation.
            rff_scale: Energy scaling factor (default 600).
                       Smaller than Dense AM (1200) because cross-scale
                       features are more informative per dimension.
        """
        self.vocab_size = vocab_size
        self.n_pos = n_pos
        self.n_topics = n_topics
        self.D = D
        self.context_hash_dim = context_hash_dim
        self.seed = seed
        self.rff_scale = rff_scale

        rng = np.random.RandomState(seed)

        # Word hash vectors: (min(vocab_size, 50000), context_hash_dim) int8
        # Each word gets a random sparse ternary vector for context hashing.
        n_hash_rows = min(vocab_size, 50000)
        self.word_hashes = rng.choice(
            [-1, 0, 1],
            size=(n_hash_rows, context_hash_dim),
            p=[0.33, 0.34, 0.33],
        ).astype(np.int8)

        # POS hash vectors: (n_pos, context_hash_dim) int8
        # Each POS tag gets a random sparse ternary vector.
        # This allows the feature to distinguish e.g. "run" as VERB vs NOUN
        # even when the word hash is the same.
        self.pos_hashes = rng.choice(
            [-1, 0, 1],
            size=(n_pos, context_hash_dim),
            p=[0.33, 0.34, 0.33],
        ).astype(np.int8)

        # Topic hash vectors: (n_topics, context_hash_dim) int8
        # Each topic gets a random sparse ternary vector.
        # This allows the feature to distinguish e.g. "goal" in SPORTS
        # vs "goal" in POLITICS.
        self.topic_hashes = rng.choice(
            [-1, 0, 1],
            size=(n_topics, context_hash_dim),
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
        self.cos_lut = self._build_cos_lut()

        # Pre-aggregated readout matrix: (V, D) int8
        # Theta[w] = Q7 * mean(phi(ctx)) over all contexts where w follows
        self.Theta: Optional[np.ndarray] = None

        # Word counts for normalization
        self._word_counts: Optional[np.ndarray] = None

        # Whether pre-aggregation has been done
        self._built = False

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

    def project(
        self,
        context_word_ids: List[int],
        context_pos_ids: List[int],
        context_topic_ids: List[int],
    ) -> np.ndarray:
        """
        Map a multi-scale context to a D-dimensional cross-scale feature vector.

        This is the core random feature computation. Unlike Dense AM which
        only uses word-level hashes, this method combines word, POS, and
        topic hashes into a single context vector, creating features that
        capture cross-scale interactions.

        Steps:
          1. Superpose word+POS+topic hash vectors with position weighting
          2. Project through random matrix W
          3. Add random bias b
          4. Apply cosine via LUT -> int8 feature vector

        The cross-scale combination is additive in the hash space:
          h += word_hashes[w] * weight + pos_hashes[p] * weight + topic_hashes[t] * weight

        The random projection W then mixes these components, creating
        joint features that are sensitive to all three scales simultaneously.

        Args:
            context_word_ids: List of word IDs forming the context.
            context_pos_ids: List of POS IDs (same length as context_word_ids).
            context_topic_ids: List of topic IDs (same length as context_word_ids).

        Returns:
            np.ndarray of shape (D,) with dtype int8, values in [-127, 127].
        """
        if not context_word_ids:
            return np.zeros(self.D, dtype=np.int8)

        # --- Step 1: Compute cross-scale context hash ---
        # Superpose word+POS+topic hash vectors with position-dependent weights.
        # More recent words get higher weight, giving the features
        # a recency bias that matches linguistic locality.
        h = np.zeros(self.context_hash_dim, dtype=np.int32)

        n_ctx = len(context_word_ids)
        start = max(0, n_ctx - self.CONTEXT_WINDOW)
        recent_words = context_word_ids[start:]
        recent_pos = context_pos_ids[start:]
        recent_topics = context_topic_ids[start:]

        for pos, (w_id, p_id, t_id) in enumerate(
            zip(recent_words, recent_pos, recent_topics)
        ):
            # Position weight: 1 for oldest in window, CONTEXT_WINDOW for newest
            weight = pos + 1

            # Word hash contribution
            if 0 <= w_id < len(self.word_hashes):
                h += self.word_hashes[w_id].astype(np.int32) * weight

            # POS hash contribution
            if 0 <= p_id < self.n_pos:
                h += self.pos_hashes[p_id].astype(np.int32) * weight

            # Topic hash contribution
            if 0 <= t_id < self.n_topics:
                h += self.topic_hashes[t_id].astype(np.int32) * weight

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
        phi = self.cos_lut[indices]  # (D,) int8

        return phi

    def build(
        self,
        sequences: List[List[int]],
        word_pos_tags=None,
        word_topics=None,
        max_sequences: Optional[int] = None,
    ) -> "CrossScaleRFF":
        """
        Build the Theta matrix from training sequences.

        For each (context, target) pair in training:
          Theta[target] += phi(context)

        where the context includes word, POS, and topic information.

        After accumulation, normalize by word count to get the MEAN
        feature vector per word (Q7 fixed-point). This prevents
        frequent words from having enormous Theta values that would
        dominate the dot products.

        The count normalization is:
          Theta_norm[w] = Theta_sum[w] * 128 // max(1, count[w])

        This gives Q7 * mean(phi) in int8 range. For a word seen K
        times with int8 phi values in [-127, 127]:
          |Theta_sum| <= K * 127
          |Theta_norm| <= K * 127 * 128 / K = 16256
          Clipped to [-127, 127] for int8 storage.

        Args:
            sequences: List of training sequences (lists of word IDs).
            word_pos_tags: Either a dict {word_id: pos_id} for per-word lookup,
                           or a list of POS tag sequences parallel to `sequences`.
                           If dict, POS IDs are looked up per word.
            word_topics: Either a numpy array (vocab_size,) of per-word topic
                         assignments, or a list of topic sequences parallel
                         to `sequences`. If array, topic IDs are looked up per word.
            max_sequences: Cap on number of sequences to process (None = all).

        Returns:
            self
        """
        V = self.vocab_size
        D = self.D

        # Accumulate in int32 (safe for up to ~16K additions of int8 values)
        Theta_sum = np.zeros((V, D), dtype=np.int32)
        word_counts = np.zeros(V, dtype=np.int32)

        n_seqs = len(sequences)
        if max_sequences is not None:
            n_seqs = min(n_seqs, max_sequences)

        print(
            f"    Pre-aggregating RFF Theta matrix ({n_seqs} sequences, D={D})..."
        )

        # Determine whether pos/topics are dict/array (per-word) or list (per-sequence)
        pos_is_dict = isinstance(word_pos_tags, dict)
        topic_is_array = word_topics is not None and hasattr(word_topics, '__len__') and not isinstance(word_topics, list)

        for seq_idx in range(n_seqs):
            seq = sequences[seq_idx]

            # Get per-sequence POS and topic arrays
            if pos_is_dict:
                pos_seq = [word_pos_tags.get(w, 0) for w in seq]
            elif isinstance(word_pos_tags, list) and seq_idx < len(word_pos_tags):
                pos_seq = word_pos_tags[seq_idx]
            else:
                pos_seq = [0] * len(seq)

            if topic_is_array:
                # word_topics is a numpy array of shape (vocab_size,)
                topic_seq = [int(word_topics[w]) if 0 <= w < len(word_topics) else 0 for w in seq]
            elif isinstance(word_topics, list) and seq_idx < len(word_topics):
                topic_seq = word_topics[seq_idx]
            else:
                topic_seq = [0] * len(seq)

            for pos in range(1, len(seq)):
                context_words = seq[:pos]
                context_pos = pos_seq[:pos]
                context_topics = topic_seq[:pos]
                target = seq[pos]
                if 0 <= target < V:
                    phi = self.project(
                        context_words, context_pos, context_topics
                    )  # (D,) int8
                    Theta_sum[target] += phi.astype(np.int32)
                    word_counts[target] += 1

        # Count-normalize: Theta_norm = Theta_sum * Q7 / max(1, count)
        # This gives Q7 * mean(phi) per word
        # Vectorized: avoid Python for-loop over V words
        counts_safe = np.maximum(word_counts, 1)[:, np.newaxis]  # (V, 1)
        normalized = (Theta_sum * self.COUNT_NORM_Q) // counts_safe  # (V, D) int32
        Theta_norm = np.clip(normalized, -127, 127).astype(np.int8)
        zero_mask = word_counts == 0
        Theta_norm[zero_mask] = 0

        self.Theta = Theta_norm
        self._word_counts = word_counts
        self._built = True

        mem_mb = self.Theta.nbytes / (1024 * 1024)
        n_nonzero = int(np.sum(word_counts > 0))
        print(
            f"    RFF Theta: shape={self.Theta.shape}, dtype=int8, "
            f"memory={mem_mb:.1f} MB, {n_nonzero} words with features"
        )

        return self

    def compute_energy(
        self,
        context_word_ids: List[int],
        context_pos_ids: List[int],
        context_topic_ids: List[int],
        candidate_words: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cross-scale RFF energy for candidate words.

        E_rff(w) = -(phi(ctx) . Theta[w]) * rff_scale / Q10

        where:
          - phi(ctx) is the D-dimensional cross-scale random feature vector
            for the context (word+POS+topic)
          - Theta[w] is the pre-aggregated feature vector for word w (int8)
          - The dot product is normalized to Q10 fixed point

        The normalization ensures energies are in a predictable range
        regardless of the raw dot product magnitude, making the
        rff_scale parameter meaningful and comparable across different
        training sets.

        Normalization:
          norm_dot = dot * Q10 / max(1, max_abs_dot)
          Then: E = -(norm_dot * rff_scale) / Q10

        This ensures energies are always in range [-rff_scale, 0] approximately,
        with the best candidate getting energy closest to -rff_scale and the
        worst getting energy closest to 0.

        Args:
            context_word_ids: List of context word IDs.
            context_pos_ids: List of context POS IDs.
            context_topic_ids: List of context topic IDs.
            candidate_words: Array of candidate word IDs.

        Returns:
            np.ndarray of int64 energies, shape (len(candidate_words),).
            LOWER energy = better match = more likely under Boltzmann.
        """
        n_candidates = len(candidate_words)

        if not self._built or self.Theta is None:
            return np.zeros(n_candidates, dtype=np.int64)

        # Project current context to feature space
        phi_ctx = self.project(
            context_word_ids, context_pos_ids, context_topic_ids
        )  # (D,) int8

        # Look up pre-aggregated Theta for candidates
        safe_candidates = np.clip(candidate_words, 0, self.vocab_size - 1)
        Theta_candidates = self.Theta[safe_candidates]  # (n, D) int8

        # Dot product: phi_ctx . Theta[w] for each candidate
        # int8 . int8 -> int32 per element, sum over D dimensions
        dots = Theta_candidates.astype(np.int32) @ phi_ctx.astype(
            np.int32
        )  # (n,) int32

        # Normalize dot products to Q10 fixed point
        # This prevents the raw dot product magnitude from causing overflow.
        max_abs_dot = max(self.DOT_FLOOR, int(np.max(np.abs(dots))))
        Q = self.DOT_NORM_Q  # 1024

        norm_dots = (dots.astype(np.int64) * Q) // max_abs_dot  # (n,) int64

        # Linear energy: E = -(norm_dot * rff_scale) / Q
        energies = -(norm_dots * self.rff_scale) // Q

        return energies.astype(np.int64)

    @property
    def built(self) -> bool:
        """Whether the Theta matrix has been pre-aggregated."""
        return self._built

    @property
    def word_counts(self) -> Optional[np.ndarray]:
        """Access word counts from pre-aggregation."""
        return self._word_counts
