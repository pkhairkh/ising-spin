"""
Quantized Fourier Holographic Reduced Representations (qFHRR).

Implements a compositional VSA using modular arithmetic on uint8 phase vectors.
In Fourier HRR, vectors represent complex numbers via quantized phases.
Binding = circular convolution in spatial domain = element-wise addition mod Q
  in the phase domain. This is FAR cheaper than binary/bipolar VSA binding.

Architecture:
  - Phase vectors: uint8 arrays of shape (D,), D=512, Q=256
  - bind(a, b)    = (a + b) mod 256        (element-wise modular addition)
  - unbind(a, b)  = (a - b) mod 256        (element-wise modular subtraction)
  - superpose(a,b) = clip(a + b, 0, 255)   (saturating addition = bundling)
  - similarity(a,b) = sum of phase_diff_LUT[|a - b|]  (int32)

For the Ising Spin Glass LM:
  - Each word gets a random hash vector:  hash_word
  - Role vectors for POS and Topic:       role_pos, role_topic
  - Encoded token = superpose(bind(hash_word, role_pos), bind(hash_topic, role_topic))
  - Readout matrix R[w] precomputed for all V words
  - Energy: E_vsa(w) = -sim(context_encoding, R[w])

Memory budget:  R is (V, 512) uint8 = ~25 MB for V=49000.
Computation:    512 additions + 512 LUT lookups per candidate = ~0.5 ms for 5K candidates.
"""

import numpy as np
from typing import Optional


class QFHRRVectors:
    """
    Core qFHRR vector operations.

    Phase vectors are uint8 arrays of dimension D. Each component is a
    quantized angle in [0, 255] representing a phase in [0, 2*pi).
    Q=256 levels give sufficient resolution for similarity computation.

    The phase-difference lookup table maps |a_i - b_i| (mod 256) to a
    similarity contribution. Maximum similarity at phase diff = 0,
    minimum at phase diff = 128 (opposite phases).
    """

    # Number of quantization levels for phases
    Q = 256

    # Maximum similarity value per dimension (at phase diff = 0)
    # This determines the scale of the similarity function.
    # With D=512 and MAX_SIM_PER_DIM=256, max total similarity = 512*256 = 131072
    # which fits comfortably in int32 (2^31 = 2,147,483,648).
    MAX_SIM_PER_DIM = 256

    def __init__(self, dimension: int = 512, seed: int = 42):
        """
        Initialize qFHRR vector operations.

        Args:
            dimension: Vector dimension D (default 512).
            seed: Random seed for deterministic generation.
        """
        self.D = dimension
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        # Build the phase-difference similarity lookup table.
        # For phase difference d in [0, 255]:
        #   similarity(d) = MAX_SIM_PER_DIM * cos(2*pi*d/256) clipped to [0, MAX_SIM_PER_DIM]
        # In integer: cos(2*pi*d/256) ≈ 1 when d≈0, ≈ -1 when d≈128
        # We clip negatives to 0 so that opposite phases contribute 0 (not negative).
        # This makes similarity a measure of alignment, not correlation.
        self._phase_diff_lut = self._build_phase_diff_lut()

    def _build_phase_diff_lut(self) -> np.ndarray:
        """
        Build the phase-difference similarity lookup table.

        For phase difference d (0 to 255):
            sim(d) = max(0, round(MAX_SIM_PER_DIM * cos(2*pi*d/256)))

        Properties:
            LUT[0]   = MAX_SIM_PER_DIM  (same phase, max similarity)
            LUT[64]  ≈ 0                (90 degrees, zero similarity)
            LUT[128] = 0                (opposite phase, zero similarity)
            Monotonically decreasing from 0 to 128, then increasing back.

        Note: This uses math.cos() during INITIALIZATION ONLY (one-time setup).
        The LUT is then used at inference time via integer lookup — NO float
        operations in the hot path.

        Returns:
            np.ndarray of shape (256,) with int32 similarity values.
        """
        import math

        lut = np.zeros(self.Q, dtype=np.int32)

        for d in range(self.Q):
            # cos(2*pi*d/256) — computed once during init, not in hot path
            cos_val = math.cos(2 * math.pi * d / 256)

            # Convert to similarity: max(0, round(cos_val * MAX_SIM_PER_DIM))
            sim_val = round(cos_val * self.MAX_SIM_PER_DIM)
            sim_val = max(0, sim_val)  # clip negatives to 0

            lut[d] = sim_val

        return lut

    def generate(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate n random phase vectors.

        Args:
            n: Number of vectors to generate.
            seed: Optional seed override (uses instance seed if None).

        Returns:
            np.ndarray of shape (n, D) with dtype uint8, values in [0, 255].
        """
        rng = np.random.RandomState(seed if seed is not None else self.seed)
        return rng.randint(0, self.Q, size=(n, self.D), dtype=np.uint8)

    def generate_one(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a single random phase vector.

        Args:
            seed: Optional seed override.

        Returns:
            np.ndarray of shape (D,) with dtype uint8.
        """
        rng = np.random.RandomState(seed if seed is not None else self.seed)
        return rng.randint(0, self.Q, size=self.D, dtype=np.uint8)

    @staticmethod
    def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Bind two phase vectors via element-wise modular addition.

        bind(a, b) = (a + b) mod 256

        This is the VSA binding operation. In Fourier HRR, binding
        corresponds to circular convolution, which in the phase domain
        is simply addition modulo the number of phase levels.

        Properties:
            Commutative:  bind(a, b) == bind(b, a)
            Associative:  bind(bind(a, b), c) == bind(a, bind(b, c))
            Identity:     bind(a, zero) == a  (where zero is all-zeros vector)

        Args:
            a: Phase vector(s), uint8, shape (D,) or (n, D).
            b: Phase vector(s), uint8, same shape as a.

        Returns:
            Bound vector(s), uint8, same shape as input.
        """
        # uint8 addition automatically wraps mod 256 in numpy
        return (a.astype(np.uint16) + b.astype(np.uint16)).astype(np.uint8)

    @staticmethod
    def unbind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Unbind a phase vector by subtracting the binding key.

        unbind(a, b) = (a - b) mod 256

        This is the approximate inverse of bind. If a = bind(x, b),
        then unbind(a, b) ≈ x (up to noise from superposition).

        Args:
            a: Bound phase vector(s), uint8.
            b: Key vector(s), uint8.

        Returns:
            Unbound vector(s), uint8, same shape as input.
        """
        # Use int16 to handle negative values, then mod 256
        result = (a.astype(np.int16) - b.astype(np.int16)) % 256
        return result.astype(np.uint8)

    @staticmethod
    def superpose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Superpose (bundle) two phase vectors via saturating addition.

        superpose(a, b) = clip(a + b, 0, 255)

        This is the VSA bundling operation. Unlike binding (modular addition),
        superposition uses SATURATING addition. This preserves the phase
        structure when multiple vectors are bundled together.

        For proper VSA bundling with phase vectors, one should use
        weighted circular mean. However, saturating addition is a
        reasonable integer approximation that:
        1. Preserves the dominant phase when vectors agree
        2. Moves toward 255 (saturation) when vectors disagree
        3. Is extremely cheap to compute

        Args:
            a: Phase vector(s), uint8.
            b: Phase vector(s), uint8, same shape as a.

        Returns:
            Superposed vector(s), uint8, same shape as input.
        """
        result = a.astype(np.uint16) + b.astype(np.uint16)
        return np.clip(result, 0, 255).astype(np.uint8)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> int:
        """
        Compute similarity between two phase vectors using the LUT.

        similarity(a, b) = sum of phase_diff_LUT[|a_i - b_i|_circular]

        where |x|_circular = min(x, 256 - x) is the circular distance
        in the phase domain.

        Higher similarity = more aligned = lower energy.

        Args:
            a: Phase vector, uint8, shape (D,).
            b: Phase vector, uint8, shape (D,).

        Returns:
            Integer similarity score (int32 range). Self-similarity is maximum.
        """
        # Compute circular phase difference
        diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
        # Circular: min(d, 256-d) — but our LUT is symmetric, so just use d
        # Actually, we need circular distance: min(d, 256-d)
        diff_circular = np.minimum(diff, 256 - diff).astype(np.int32)

        # Look up similarity contributions
        sim_contributions = self._phase_diff_lut[diff_circular]

        return int(sim_contributions.sum())

    def similarity_batch(
        self,
        a: np.ndarray,
        B: np.ndarray,
    ) -> np.ndarray:
        """
        Compute similarity of one vector against a matrix of vectors.

        This is the critical operation for VSA energy computation:
        compare the context encoding against the readout matrix R.

        Args:
            a: Single phase vector, uint8, shape (D,).
            B: Matrix of phase vectors, uint8, shape (n, D).

        Returns:
            np.ndarray of int32 similarities, shape (n,).
        """
        # Compute circular phase differences for all rows at once
        diff = np.abs(a.astype(np.int16) - B.astype(np.int16))  # (n, D) int16
        diff_circular = np.minimum(diff, 256 - diff).astype(np.int32)  # (n, D) int32

        # LUT lookup for all elements
        sim_contributions = self._phase_diff_lut[diff_circular]  # (n, D) int32

        # Sum along dimension axis
        return sim_contributions.sum(axis=1)  # (n,) int32

    @property
    def phase_diff_lut(self) -> np.ndarray:
        """Access the phase-difference lookup table."""
        return self._phase_diff_lut


class VSAEncoder:
    """
    VSA encoder that creates compositional codes for tokens.

    Each token is encoded as a bound superposition of:
      1. Word identity hash vector
      2. POS role binding
      3. Topic role binding

    Encoding formula:
      encode(w, pos, topic) = superpose(
          bind(hash_word[w], role_pos[pos]),
          bind(hash_topic[w], role_topic[topic])
      )

    The word hash captures WHAT the word is. The POS binding captures
    HOW it functions syntactically. The topic binding captures WHERE
    it belongs semantically. Binding (not superposition) is used for
    roles so that the same word in different syntactic/semantic roles
    gets different codes.

    The readout matrix R[w] stores the precomputed encoding for every
    vocabulary word using its dominant POS and topic assignments.
    At inference, we compute:
      E_vsa(w) = -similarity(context_encoding, R[w]) * vsa_scale
    where context_encoding is a superposition of recent token encodings.
    """

    def __init__(
        self,
        vocab_size: int,
        n_pos: int = 13,
        n_topics: int = 16,
        dimension: int = 512,
        seed: int = 42,
    ):
        """
        Initialize VSA encoder.

        Args:
            vocab_size: Number of words in vocabulary (V).
            n_pos: Number of POS tags (default 13 for coarse POS).
            n_topics: Number of topics (default 16).
            dimension: Phase vector dimension D (default 512).
            seed: Random seed for deterministic vector generation.
        """
        self.vocab_size = vocab_size
        self.n_pos = n_pos
        self.n_topics = n_topics
        self.dimension = dimension
        self.seed = seed

        # Core qFHRR operations
        self.qfhrr = QFHRRVectors(dimension=dimension, seed=seed)

        # --- Generate base vectors ---
        # Word identity hash vectors: one per vocabulary word
        # Using seeded generation for reproducibility
        self.hash_words = self._generate_seeded_matrix(
            vocab_size, dimension, seed_base=seed
        )

        # POS role vectors: one per POS tag
        self.role_pos = self._generate_seeded_matrix(
            n_pos, dimension, seed_base=seed + 10000
        )

        # Topic role vectors: one per topic
        self.role_topic = self._generate_seeded_matrix(
            n_topics, dimension, seed_base=seed + 20000
        )

        # Readout matrix: computed during build()
        self._readout_matrix: Optional[np.ndarray] = None  # (V, D) uint8
        self._built = False

    def _generate_seeded_matrix(
        self, n: int, d: int, seed_base: int
    ) -> np.ndarray:
        """
        Generate a matrix of random phase vectors with a specific seed.

        Args:
            n: Number of vectors.
            d: Dimension of each vector.
            seed_base: Base seed for this generation.

        Returns:
            np.ndarray of shape (n, d), dtype uint8.
        """
        rng = np.random.RandomState(seed_base)
        return rng.randint(0, 256, size=(n, d), dtype=np.uint8)

    def encode(
        self,
        word_id: int,
        pos_id: int,
        topic_id: int,
    ) -> np.ndarray:
        """
        Encode a token as a compositional VSA code.

        encode(w, pos, topic) = superpose(
            bind(hash_word[w], role_pos[pos]),
            bind(hash_topic_for_word[w], role_topic[topic])
        )

        Args:
            word_id: Word index in [0, vocab_size).
            pos_id: POS tag index in [0, n_pos).
            topic_id: Topic index in [0, n_topics).

        Returns:
            Phase vector, uint8, shape (D,).
        """
        # Clamp indices to valid range
        w = max(0, min(word_id, self.vocab_size - 1))
        p = max(0, min(pos_id, self.n_pos - 1))
        t = max(0, min(topic_id, self.n_topics - 1))

        # Bind word identity with POS role
        word_pos_bound = self.qfhrr.bind(self.hash_words[w], self.role_pos[p])

        # For topic, we use the same word hash but bind with topic role
        # This means the topic binding captures "this word in this topic context"
        word_topic_bound = self.qfhrr.bind(self.hash_words[w], self.role_topic[t])

        # Superpose (bundle) the two bound representations
        return self.qfhrr.superpose(word_pos_bound, word_topic_bound)

    def build(
        self,
        pos_system=None,
        word_topics: Optional[np.ndarray] = None,
    ) -> "VSAEncoder":
        """
        Build the readout matrix R for all vocabulary words.

        R[w] = encode(w, dominant_pos[w], dominant_topic[w])

        Uses the word's primary POS assignment and topic assignment
        to precompute the encoding vector for each vocabulary word.

        Args:
            pos_system: POSTypeSystem with allowed_types mapping.
                        If None, defaults to NOUN (pos_id=0) for all words.
            word_topics: (vocab_size,) int8 array of topic assignments.
                         If None, defaults to topic 0 for all words.

        Returns:
            self
        """
        V = self.vocab_size
        D = self.dimension

        print(f"  Building VSA readout matrix (V={V}, D={D})...")

        # Precompute dominant POS for each word
        dominant_pos = np.zeros(V, dtype=np.int32)
        if pos_system is not None and hasattr(pos_system, 'allowed_types'):
            for w_id, types in pos_system.allowed_types.items():
                if 0 <= w_id < V and types:
                    # Use the most common POS as dominant
                    dominant_pos[w_id] = min(types)  # lowest index = most specific

        # Precompute topic assignments
        word_topic_ids = np.zeros(V, dtype=np.int32)
        if word_topics is not None:
            for w_id in range(min(V, len(word_topics))):
                t = int(word_topics[w_id])
                if 0 <= t < self.n_topics:
                    word_topic_ids[w_id] = t

        # Build readout matrix
        self._readout_matrix = np.zeros((V, D), dtype=np.uint8)

        # Batch computation for efficiency
        # Step 1: Bind all word hashes with their POS roles
        # hash_words[w] shape: (V, D), role_pos[p] shape: (D,)
        # We need: bind(hash_words[w], role_pos[dominant_pos[w]]) for each w
        for w_id in range(V):
            p = int(dominant_pos[w_id])
            t = int(word_topic_ids[w_id])

            # Bind word hash with POS role
            word_pos = self.qfhrr.bind(self.hash_words[w_id], self.role_pos[p])

            # Bind word hash with topic role
            word_topic = self.qfhrr.bind(self.hash_words[w_id], self.role_topic[t])

            # Superpose
            self._readout_matrix[w_id] = self.qfhrr.superpose(word_pos, word_topic)

        mem_mb = self._readout_matrix.nbytes / (1024 * 1024)
        print(f"    Readout matrix: shape={self._readout_matrix.shape}, "
              f"dtype=uint8, memory={mem_mb:.1f} MB")

        self._built = True
        return self

    def compute_context_encoding(
        self,
        context_word_ids: list,
        context_pos_ids: Optional[list] = None,
        context_topic_ids: Optional[list] = None,
        window: int = 10,
    ) -> np.ndarray:
        """
        Compute the VSA encoding for the current context.

        This creates a superposition of the last `window` token encodings.
        More recent tokens could be weighted more heavily, but for integer
        simplicity we use uniform superposition with truncation.

        Args:
            context_word_ids: List of word IDs in the context.
            context_pos_ids: Optional list of POS IDs (same length).
            context_topic_ids: Optional list of topic IDs (same length).
            window: Number of recent tokens to include (default 10).

        Returns:
            Context encoding vector, uint8, shape (D,).
        """
        if not context_word_ids:
            return np.zeros(self.dimension, dtype=np.uint8)

        # Take the last `window` tokens
        recent = context_word_ids[-window:]
        recent_pos = context_pos_ids[-window:] if context_pos_ids else None
        recent_topics = context_topic_ids[-window:] if context_topic_ids else None

        # Encode each token and superpose
        # For integer simplicity: just superpose all encodings equally.
        # Saturating addition naturally handles overflow.
        result = np.zeros(self.dimension, dtype=np.uint16)  # use uint16 to track overflows

        for i, w_id in enumerate(recent):
            p_id = recent_pos[i] if recent_pos and i < len(recent_pos) else 0
            t_id = recent_topics[i] if recent_topics and i < len(recent_topics) else 0

            encoding = self.encode(w_id, p_id, t_id)
            result += encoding.astype(np.uint16)

        # Clip to uint8 range
        return np.clip(result, 0, 255).astype(np.uint8)

    def compute_vsa_energy(
        self,
        context_encoding: np.ndarray,
        candidate_words: np.ndarray,
        vsa_scale: int = 800,
    ) -> np.ndarray:
        """
        Compute VSA energy for candidate words.

        E_vsa(w) = -similarity(context_encoding, R[w]) * vsa_scale / max_sim

        The negative sign means: words MORE similar to the context get
        LOWER energy (more likely under the Boltzmann distribution).

        The vsa_scale parameter controls the energy magnitude to be
        comparable with other energy terms (recall, state, etc.).

        Args:
            context_encoding: Context VSA code, uint8, shape (D,).
            candidate_words: Array of candidate word IDs.
            vsa_scale: Energy scaling factor (default 800).

        Returns:
            np.ndarray of int64 energies, shape (len(candidate_words),).
            LOWER energy = more similar to context = more likely.
        """
        n_candidates = len(candidate_words)
        energies = np.zeros(n_candidates, dtype=np.int64)

        if not self._built or self._readout_matrix is None:
            return energies

        # Look up readout vectors for candidates
        # Clamp candidate IDs to valid range
        safe_candidates = np.clip(candidate_words, 0, self.vocab_size - 1)
        candidate_vectors = self._readout_matrix[safe_candidates]  # (n, D) uint8

        # Compute batch similarity
        similarities = self.qfhrr.similarity_batch(
            context_encoding, candidate_vectors
        )  # (n,) int32

        # Maximum possible similarity (self-similarity with identical vectors)
        max_sim = self.qfhrr.MAX_SIM_PER_DIM * self.dimension

        # Convert similarity to energy:
        # E = (max_sim - sim) * vsa_scale / max_sim
        # This makes: max similarity → energy = 0, min similarity → energy = vsa_scale
        # All integer: use (max_sim - sim) * vsa_scale // max_sim
        for i in range(n_candidates):
            sim = int(similarities[i])
            energy = ((max_sim - sim) * vsa_scale) // max_sim
            energies[i] = energy

        return energies

    @property
    def readout_matrix(self) -> Optional[np.ndarray]:
        """Access the readout matrix R, shape (V, D), dtype uint8."""
        return self._readout_matrix

    @property
    def built(self) -> bool:
        """Whether the encoder has been built."""
        return self._built
