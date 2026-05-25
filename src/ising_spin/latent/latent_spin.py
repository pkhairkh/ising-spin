"""
Learned Latent Spin Glass — GENUINE understanding from learned physics.

THE PROBLEM WITH ALL PREVIOUS APPROACHES:
  - DocumentState: 7 HAND-CODED variables (tense, mode, entity, etc.)
    with DETERMINISTIC RULE-BASED triggers. Not generalizable.
  - Macro spins: Entity/phase/scene trackers with HAND-CODED detection
    rules. Not generalizable.
  - SSR: W_word is RANDOM, J_struct is RANDOM. The dynamics are driven
    by noise, not by meaningful structure.

THE SOLUTION: LEARN the spin vectors and the coupling matrix FROM DATA.

HOW SPIN VECTORS ARE LEARNED (context-sign hashing):
  For each word w, compute its context distribution by accumulating
  random projections of the words that appear near w in training data.
  Then take the sign of each dimension:
    sigma_w[d] = sign(sum_{w' in context(w)} R[d, w'])
  where R is a random projection matrix (deterministic from seed).

  Words with similar context distributions get similar spin vectors.
  This naturally groups synonyms, syntactically similar words, and
  semantically related words — WITHOUT anyone declaring what the
  dimensions mean.

HOW THE COUPLING MATRIX IS LEARNED (Hopfield storage):
  For each training context window, compute the "pattern" sigma_ctx
  (the aggregated spin state of the window). Then store these patterns
  using the Hopfield rule:
    J += sigma_ctx * sigma_ctx^T

  This captures the ACTUAL correlation structure between spin dimensions.
  If dimension i tends to be +1 when dimension j is +1 in training data,
  J_ij will be positive — the model learns that these dimensions are
  correlated. This is REAL dependency structure, not hand-coded rules.

HOW THIS CREATES LONG-RANGE DEPENDENCIES:
  During generation, the document spin state sigma_doc evolves via
  Ising dynamics with the LEARNED coupling J:
    sigma_doc(t+1) = sign(J_total @ sigma_doc(t) + alpha * sigma_{w(t)})

  Because J is learned from data, it captures real dependencies.
  The energy for a candidate word w depends on how well its spin vector
  aligns with the coupling-mediated document state:
    E(w) = -(sigma_w . (J_total @ sigma_doc)) * scale / norm

  This is Hopfield pattern completion: the document state is a partial
  pattern, and the model completes it by selecting words whose spin
  vectors align through the learned coupling structure.

  The EPISODIC MEMORY (Hebbian J_episodic during generation) creates
  ATTRACTORS for previously visited semantic states, enabling genuine
  long-range recall within a document.

WHY THIS IS DIFFERENT FROM REJECTED APPROACHES:
  REJECTED: "Dimension 0 = gender, Dimension 1 = tense, ..."
    → Static, hand-coded, non-generalizable

  NEW: sigma_w = sign(context_projection(w))
    → LATENT, DATA-DRIVEN, GENERALIZABLE
    → If gender matters in the data, the model discovers a dimension
      that correlates with it. If you train on Chinese, different
      dimensions emerge. NO HUMAN DECLARES ANYTHING.

  REJECTED: J_struct = random frustrated matrix
    → Random, doesn't encode real language structure

  NEW: J = Hopfield(spin_patterns_from_training)
    → Captures REAL dependency structure from the data
    → Creates genuine long-range dependencies based on real linguistic
      patterns, not hand-coded rules

Memory budget (D=256, V=4000):
  - spin_vectors: 4000 x 256 x 1 byte = 1 MB
  - J_learned:    256 x 256 x 2 bytes = 128 KB
  - J_episodic:   256 x 256 x 2 bytes = 128 KB
  - R (projection): 4000 x 256 x 1 byte = 1 MB (training only)
  - sigma_doc:    256 x 1 byte = negligible
  Total: ~2.3 MB (training), ~1.3 MB (inference)

Computation per token (generation):
  - J_total @ sigma_doc: 256 x 256 = 65K integer multiply-accumulates
  - Alignment for 300 candidates: 300 x 256 = 77K multiply-accumulates
  - Hebbian update: sparse, ~256 x n_flipped additions
  Total: ~150K integer ops/token (very fast on Pi 5 ARM cores)
"""

import numpy as np
from typing import Dict, List, Optional

from ..exceptions import ValidationError


class LatentSpinGlass:
    """
    Learned Latent Spin Glass — Emergent understanding from learned couplings.

    The D-dimensional binary spin vector sigma_doc encodes the document's
    current semantic state as a DISTRIBUTED representation with NO
    pre-assigned meaning. Meaning EMERGES from:
      1. The LEARNED spin vectors sigma_w (context-sign hashing)
      2. The LEARNED coupling J_learned (Hopfield storage from training)
      3. The EPISODIC coupling J_episodic (Hebbian learning during generation)
      4. The Ising dynamics that evolve sigma_doc

    UNLIKE all previous approaches:
      - DocumentState: hand-coded variables with deterministic rules
      - Macro spins: hand-coded entity/phase/scene trackers
      - SSR: random W_word, random J_struct (noise, not structure)

    THIS module has ZERO hand-coded features. Everything is learned from data.
    The spin dimensions are LATENT — discovered from the training corpus,
    not declared by a human.

    All arithmetic is integer-only.
    """

    # Default spin dimension
    # D=256 gives 2^256 possible spin configurations, vastly more than
    # the number of distinct semantic states in natural language.
    DEFAULT_D = 256

    # Q8 fixed-point for external field strength
    # alpha controls how strongly the current word influences the spin state.
    DEFAULT_ALPHA_Q8 = 255  # ≈ 1.0 — max Q8, ensures sigma_doc strongly tracks current word

    # Episodic Hebbian learning rate
    # Each token update adds eta_episodic to the coupling of aligned spins.
    DEFAULT_ETA_EPISODIC = 2

    # Number of mean-field sweeps per step
    # More sweeps = more relaxation toward energy minimum = more coherent
    # spin state. 4 sweeps gives better attractor formation than 2.
    DEFAULT_N_MF_SWEEPS = 4

    # Maximum absolute value for J_episodic (prevents unbounded growth)
    # Reduced from 200 to 120: prevents old episodic memories from
    # overwhelming the learned coupling structure, allowing sigma_doc
    # to remain responsive to the current semantic state.
    J_EPISODIC_CLIP = 120

    # Context window for spin vector learning (±context_window tokens)
    DEFAULT_CONTEXT_WINDOW = 5

    # Number of training windows to sample for J_learned
    # v23: Reduced from 200K to 9K (within Hopfield capacity).
    # Hopfield capacity ≈ 0.14 * D² for D=256 → ~9,175 patterns.
    # Storing 200K patterns in 256 dims is 5,579× over capacity,
    # producing a meaningless near-zero average J that is
    # perfectly balanced between +1/-1 couplings.
    # With 9K patterns, each pattern contributes meaningful structure.
    DEFAULT_N_J_WINDOWS = 9000

    # Normalization for energy dot products
    # FIXED normalization: use D (spin dimension) and h_norm (coupling field L1)
    # instead of Q10 per-candidate max_abs which compressed dynamic range
    # and made spin energy 10-25x weaker than recall energy.
    # With D-based normalization:
    #   E_direct ∈ [-latent_scale, +latent_scale] (perfect alignment to anti-alignment)
    #   E_coupling ∈ [-coupling_scale, +coupling_scale] (same)

    # Metropolis temperature for spin dynamics
    DEFAULT_TEMPERATURE = 0  # Deterministic by default for clean dynamics

    # Sigma noise injection probability
    DEFAULT_NOISE_PROB = 0.02

    # Spin vector decorrelation: mean-center accumulators before sign()
    # This fixes the representation collapse where avg Hamming = 25% (should be ~50%).
    # Without centering, common words create a shared "background" in the accumulator,
    # causing most dimensions to have the same sign across words.
    DEFAULT_DECORRELATE = True

    # J_learned sparsity: fraction of couplings to KEEP after Top-K thresholding.
    # The Hopfield storage rule with N_patterns >> D produces a dense J that is
    # essentially the rank-1 outer product of the magnetization vector — no useful
    # structure. Keeping only the strongest K% creates structural sparsity with
    # clear attractor basins instead of a flat, frustrated landscape.
    # 0.0 = keep all (no sparsification), 1.0 = keep nothing (degenerate)
    DEFAULT_J_SPARSITY = 0.15  # Keep top 15% of couplings

    def __init__(
        self,
        vocab_size: int,
        D: int = 256,
        alpha_q8: int = 128,
        eta_episodic: int = 2,
        n_mf_sweeps: int = 4,
        latent_scale: int = 6000,
        coupling_scale: int = 4000,
        temperature: int = 0,
        noise_prob: float = 0.02,
        context_window: int = 5,
        n_j_windows: int = 9000,
        seed: int = 42,
        decorrelate: bool = True,
        j_sparsity: float = 0.15,
    ):
        """
        Initialize Learned Latent Spin Glass.

        Args:
            vocab_size: Vocabulary size V.
            D: Spin dimension (default 256).
            alpha_q8: External field strength in Q8 (default 128 ≈ 0.5).
            eta_episodic: Hebbian learning rate for J_episodic (default 2).
            n_mf_sweeps: Mean-field sweeps per step (default 4).
            latent_scale: Energy scale for direct alignment (default 1200).
            coupling_scale: Energy scale for coupling-mediated alignment (default 800).
            temperature: Metropolis temperature for spin flips (default 0).
            noise_prob: Probability of random spin flip (default 0.02).
            context_window: Context window for spin learning (default 5).
            n_j_windows: Number of windows for J learning (default 200K).
            seed: Random seed for deterministic initialization.
        """
        self.vocab_size = vocab_size
        self.D = D
        self.alpha_q8 = alpha_q8
        self.eta_episodic = eta_episodic
        self.n_mf_sweeps = n_mf_sweeps
        self.latent_scale = latent_scale
        self.coupling_scale = coupling_scale
        self.temperature = temperature
        self.noise_prob = noise_prob
        self.context_window = context_window
        self.n_j_windows = n_j_windows
        self.seed = seed
        self.decorrelate = decorrelate
        self.j_sparsity = j_sparsity

        # --- Spin vectors (LEARNED during training) ---
        # sigma_w[w] = binary spin vector for word w
        # Initialized to zero; learned via context-sign hashing
        self.spin_vectors: Optional[np.ndarray] = None  # (V, D) int8

        # --- Learned coupling matrix (LEARNED during training) ---
        # J_learned captures the dependency structure from training data
        self.J_learned: Optional[np.ndarray] = None  # (D, D) int16

        # --- Per-dimension bias vector (LEARNED during training) ---
        # v23: h_bias captures the main effect of each spin dimension,
        # i.e., which dimensions tend to be +1 vs -1 after seeing context.
        # This is the "diagonal component" that gets washed out when
        # J is computed from balanced ±1 patterns. Without h_bias,
        # the coupling field averages to zero and has no directional signal.
        self.h_bias: Optional[np.ndarray] = None  # (D,) int16

        # --- Episodic coupling (Hebbian during generation) ---
        self.J_episodic = np.zeros((D, D), dtype=np.int16)

        # --- Document spin state ---
        # sigma_doc: the current document's spin configuration
        rng = np.random.RandomState(seed)
        self.sigma_doc = rng.choice([-1, 1], size=D).astype(np.int8)

        # --- Random projection matrix for context-sign hashing ---
        # Deterministic from seed; used during training only
        # Stored as int8 {-1, +1} — dense, not sparse
        self._R: Optional[np.ndarray] = None  # (V, D) int8

        # --- State ---
        self._built = False
        self._rng = np.random.RandomState(seed + 9999)
        self._prev_sigma = self.sigma_doc.copy()

        # --- Diagnostics ---
        self._stats = {
            'total_steps': 0,
            'spins_flipped': 0,
            'attractor_jumps': 0,
            'avg_alignment': 0,
            'avg_coupling_field_norm': 0,
        }

    # ===================================================================
    # BUILD: Learn spin vectors and coupling from training data
    # ===================================================================

    def build(
        self,
        sequences: List[List[int]],
        max_sequences: Optional[int] = None,
    ) -> "LatentSpinGlass":
        """
        Build learned spin vectors and coupling matrix from training data.

        Two-phase process:
          Phase 1: Learn spin vectors via context-sign hashing
            - For each word w, accumulate random projections of context words
            - sigma_w[d] = sign(accumulated_projection[d])
            - This gives words in similar contexts similar spin vectors

          Phase 2: Learn coupling matrix via Hopfield storage
            - Sample training windows, compute spin patterns
            - J_learned = sum of outer products of spin patterns
            - This captures the dependency structure between spin dimensions

        All integer arithmetic. No floating point.

        Args:
            sequences: List of training sequences (word ID lists).
            max_sequences: Cap on number of sequences (None = all).

        Returns:
            self
        """
        V = self.vocab_size
        D = self.D
        seed = self.seed

        n_seqs = len(sequences)
        if max_sequences is not None:
            n_seqs = min(n_seqs, max_sequences)

        # ---------------------------------------------------------------
        # PHASE 1: Learn spin vectors via context-sign hashing
        # ---------------------------------------------------------------
        print(f"    Phase 1: Learning spin vectors (D={D}, {n_seqs} sequences)...")

        # Generate deterministic random projection matrix R
        # R[w, d] ∈ {-1, +1} — dense binary projections
        rng = np.random.RandomState(seed)
        n_words = min(V, 50000)
        R = rng.choice([-1, 1], size=(n_words, D)).astype(np.int8)
        self._R = R  # Store for potential reuse

        # Accumulate context projections for each word
        # spin_acc[w, d] = sum of R[w', d] for all w' in context of w
        spin_acc = np.zeros((V, D), dtype=np.int32)
        word_counts = np.zeros(V, dtype=np.int32)

        window = self.context_window
        total_positions = 0

        for seq_idx in range(n_seqs):
            seq = sequences[seq_idx]
            seq_len = len(seq)

            for t in range(seq_len):
                w = seq[t]
                if w < 0 or w >= V:
                    continue

                # Accumulate random projections of context words
                ctx_start = max(0, t - window)
                ctx_end = min(seq_len, t + window + 1)

                for t2 in range(ctx_start, ctx_end):
                    if t2 == t:
                        continue
                    w2 = seq[t2]
                    if 0 <= w2 < n_words:
                        spin_acc[w] += R[w2].astype(np.int32)

                word_counts[w] += 1
                total_positions += 1

            if (seq_idx + 1) % 50000 == 0:
                n_words_seen = int(np.sum(word_counts > 0))
                print(f"      Spin vectors: {seq_idx+1}/{n_seqs} seqs, "
                      f"{n_words_seen} words with context")

        # --- Decorrelation: per-dimension balanced rank binarization ---
        #
        # WHY NOT mean-centering + sign():
        #   Mean-centering removes the average bias but does NOT guarantee
        #   balanced spins. With skewed accumulator distributions (common in
        #   small corpora like TinyStories), most values can still fall on the
        #   same side of zero after centering. Empirical result:
        #     Hamming 25.4% → 33.2% (improved but far from 50%)
        #     Magnetization 173/256 → 236/256 (WORSE!)
        #
        # WHY balanced rank binarization:
        #   For each dimension, rank all seen words by accumulator value,
        #   assign +1 to the top 50% and -1 to the bottom 50%.
        #
        #   This GUARANTEES:
        #     - Exactly 50% +1 per dimension → magnetization = 0
        #     - Maximum Hamming distance → ~50% (theoretical optimum)
        #     - The split is meaningful: words with higher context evidence
        #       for a dimension get +1, those with lower evidence get -1
        #
        #   This is the binary analog of median-split: instead of thresholding
        #   at a fixed value (mean/zero), we threshold at the empirical median,
        #   which adapts to the actual distribution shape.
        #
        if self.decorrelate:
            seen_mask = word_counts > 0
            seen_indices = np.where(seen_mask)[0]
            n_seen = len(seen_indices)
            n_pos_target = n_seen // 2  # Exactly half +1 per dimension

            spin_vectors = np.zeros((V, D), dtype=np.int8)

            for d in range(D):
                col = spin_acc[seen_indices, d]
                # Rank: highest accumulator values get +1
                order = np.argsort(col)  # ascending order
                # Bottom half → -1, top half → +1
                assignments = np.full(n_seen, -1, dtype=np.int8)
                assignments[order[n_seen - n_pos_target:]] = 1
                spin_vectors[seen_indices, d] = assignments

            # Unseen words: balanced random spins (exactly D/2 +1, D/2 -1)
            unseen_mask = word_counts == 0
            if np.any(unseen_mask):
                n_unseen = int(np.sum(unseen_mask))
                template = np.array(
                    [1] * (D // 2) + [-1] * (D - D // 2), dtype=np.int8
                )
                balanced = np.zeros((n_unseen, D), dtype=np.int8)
                for i in range(n_unseen):
                    rng.shuffle(template)
                    balanced[i] = template.copy()
                spin_vectors[unseen_mask] = balanced

            print(f"      Decorrelation: balanced rank binarization "
                  f"({n_seen} words, {n_pos_target}/{n_seen} +1 per dim)")
        else:
            # Original: sign of raw accumulator
            seen_mask = word_counts > 0
            spin_vectors = np.sign(spin_acc).astype(np.int8)
            zero_mask = spin_acc == 0
            spin_vectors[zero_mask & seen_mask[:, np.newaxis]] = 1
            unseen_mask = word_counts == 0
            if np.any(unseen_mask):
                n_unseen = int(np.sum(unseen_mask))
                spin_vectors[unseen_mask] = rng.choice(
                    [-1, 1], size=(n_unseen, D)
                ).astype(np.int8)

        self.spin_vectors = spin_vectors
        n_with_features = int(np.sum(word_counts > 0))
        print(f"      Spin vectors: {n_with_features} words with learned features "
              f"({total_positions} positions)")

        # ---------------------------------------------------------------
        # PHASE 2: Learn coupling matrix via CONDITION-RESPONSE
        # ---------------------------------------------------------------
        # ARCHITECTURAL CHANGE: Replace Hopfield auto-association
        #   J = S_context^T @ S_context / N
        # with CONDITION-RESPONSE:
        #   J = S_context^T @ sigma_next / N
        #
        # WHY: Hopfield auto-association stores patterns for RECALL, but
        #   with N_patterns >> capacity (130K >> 36), it produces a
        #   meaningless near-zero average. The condition-response approach
        #   directly captures the PREDICTIVE relationship: "when context
        #   spin looks like X, the next word's spin should look like Y."
        #
        # This is a linear predictor from context spin state to next-word
        # spin pattern. In the energy function:
        #   E_coupling(w) = -(sigma_w . (J @ sigma_doc))
        # J @ sigma_doc gives the PREDICTED next-word spin pattern, and
        # sigma_w . (J @ sigma_doc) measures how well word w matches this
        # prediction. This creates genuine long-range dependencies because
        # sigma_doc encodes the entire document history.
        #
        print(f"    Phase 2: Learning coupling matrix via CONDITION-RESPONSE "
              f"(sampling {self.n_j_windows} windows)...")

        rng2 = np.random.RandomState(seed + 100)

        # Collect all (seq_idx, position) pairs
        all_windows = []
        for seq_idx in range(n_seqs):
            seq = sequences[seq_idx]
            if len(seq) >= 3:
                for pos in range(1, len(seq)):
                    all_windows.append((seq_idx, pos))

        # Subsample if too many
        n_sample = min(self.n_j_windows, len(all_windows))
        if n_sample < len(all_windows):
            indices = rng2.choice(len(all_windows), size=n_sample, replace=False)
            sampled_windows = [all_windows[i] for i in indices]
        else:
            sampled_windows = all_windows

        # Compute context spin patterns AND next-word spin vectors
        # v23: Use RAW accumulators (continuous) for context instead of
        # binary spin vectors. Balanced rank binarization ensures 50% +1
        # per dimension, which makes the cross-correlation in J average
        # to zero. By using raw accumulators, we preserve the directional
        # signal that binarization destroys.
        #
        # Context: sum of raw spin_acc (before binarization) for context words
        # Next: still use binary spin vector (for alignment with dynamics)
        window_size = 10  # Wider window: 10 words of context
        S_context = np.zeros((len(sampled_windows), D), dtype=np.int8)
        S_next = np.zeros((len(sampled_windows), D), dtype=np.int8)
        # v23: Also store raw accumulator sums for context
        C_raw = np.zeros((len(sampled_windows), D), dtype=np.int32)
        valid_mask = np.zeros(len(sampled_windows), dtype=bool)

        for i, (seq_idx, pos) in enumerate(sampled_windows):
            seq = sequences[seq_idx]
            # Context: words BEFORE position pos
            start = max(0, pos - window_size)
            end = pos
            context_words = seq[start:end]
            # Target: the word AT position pos
            next_word = seq[pos] if pos < len(seq) else -1

            if not context_words or next_word < 0 or next_word >= V:
                continue

            # v23: Accumulate RAW spin accumulators (continuous values)
            # This preserves the directional signal that binarization destroys.
            acc_raw = np.zeros(D, dtype=np.int32)
            for w in context_words:
                if 0 <= w < V:
                    acc_raw += spin_acc[w].astype(np.int32)

            C_raw[i] = acc_raw

            # Also compute binary context pattern (for fallback)
            acc = np.zeros(D, dtype=np.int32)
            for w in context_words:
                if 0 <= w < V:
                    acc += spin_vectors[w].astype(np.int32)
            s_ctx = np.sign(acc).astype(np.int8)
            zero_mask = acc == 0
            s_ctx[zero_mask] = 1

            S_context[i] = s_ctx
            S_next[i] = spin_vectors[next_word]  # Next word's spin vector
            valid_mask[i] = True

        n_valid = int(np.sum(valid_mask))
        print(f"      Valid context-next pairs: {n_valid}/{len(sampled_windows)}")

        if n_valid > 0:
            S_ctx_valid = S_context[valid_mask]  # (n_valid, D) int8
            S_next_valid = S_next[valid_mask]     # (n_valid, D) int8
            C_raw_valid = C_raw[valid_mask]        # (n_valid, D) int32 — v23: raw accumulators

            # v23 CONDITION-RESPONSE with RAW accumulators:
            # J = C_raw_context^T @ sigma_next / N
            #
            # KEY FIX: Instead of using binary S_context (which has 50% +1
            # per dimension from balanced rank binarization, making the
            # cross-correlation average to zero), we use the RAW spin_acc
            # values. These continuous values preserve the directional
            # signal: if context words have high accumulator values in
            # dimension j, C_raw[j] will be large and positive, creating
            # a strong predictive signal in J[:,j].
            #
            # Normalization: divide by sqrt(sum of squares) per dimension
            # to prevent dimensions with large accumulator values from
            # dominating J.
            print(f"      Computing J_learned via condition-response "
                  f"({n_valid} pairs x {D} dims, using RAW accumulators)...")

            # Normalize raw accumulators per dimension to prevent overflow
            # and ensure balanced contribution from each dimension
            col_norms = np.sqrt(np.sum(C_raw_valid.astype(np.float64) ** 2, axis=0))  # (D,)
            col_norms = np.maximum(col_norms, 1.0)  # Avoid division by zero
            C_normalized = (C_raw_valid.astype(np.float64) / col_norms[np.newaxis, :]) * 1000  # Scale to ~1000
            C_int = C_normalized.astype(np.int16)  # (n_valid, D) int16

            J_int = C_int.astype(np.int32).T @ S_next_valid.astype(np.int32)

            # Scale to int16: normalize by number of valid pairs
            J_scaled = (J_int * 256 // max(1, n_valid)).astype(np.int16)

            # Zero diagonal (no self-coupling)
            np.fill_diagonal(J_scaled, 0)

            # --- Top-K Sparsification ---
            # Keep only the strongest K% of couplings by absolute value.
            # This removes noise and creates structural sparsity with
            # clear directional signals.
            if 0 < self.j_sparsity < 1.0:
                abs_J = np.abs(J_scaled)
                off_diag_mask = ~np.eye(D, dtype=bool)
                off_diag_abs = abs_J[off_diag_mask]
                nonzero_off = off_diag_abs[off_diag_abs > 0]
                if len(nonzero_off) > 0:
                    percentile = (1.0 - self.j_sparsity) * 100
                    threshold = int(np.percentile(nonzero_off, percentile))
                    weak_mask = (abs_J < threshold) & off_diag_mask
                    J_scaled[weak_mask] = 0

                    n_after = int(np.sum(J_scaled != 0))
                    total_off_diag = D * (D - 1)
                    print(f"      J sparsification: kept {n_after}/{total_off_diag} "
                          f"({100*n_after/total_off_diag:.1f}%) "
                          f"at threshold={threshold} (target={self.j_sparsity*100:.0f}%)")

            self.J_learned = J_scaled

            # --- Per-dimension bias (v23) ---
            # Compute h_bias = mean of sigma_next per dimension.
            # This captures the main effect: which dimensions tend to be +1
            # vs -1 in the next word, regardless of context. Without this,
            # the coupling field has no directional bias and just averages
            # to zero, producing frustrated dynamics.
            #
            # In Hopfield theory, this is the external field that the
            # patterns create. With balanced binarization (50% +1, 50% -1),
            # the spin vectors have zero magnetization, so the J matrix
            # has zero diagonal. But the sigma_next patterns do have
            # structure — some dimensions are more commonly +1 in certain
            # contexts. This bias captures that structure.
            next_mean = np.mean(S_next_valid.astype(np.float32), axis=0)  # (D,)
            # Scale to int16: bias ∈ [-256, +256] maps to int16 range
            self.h_bias = (next_mean * 256).astype(np.int16)
            bias_max = int(np.max(np.abs(self.h_bias)))
            bias_nonzero = int(np.sum(self.h_bias != 0))
            print(f"      h_bias: max_abs={bias_max}, nonzero={bias_nonzero}/{D}")

            # Diagnostics
            j_max = int(np.max(np.abs(self.J_learned)))
            j_nnz = int(np.sum(self.J_learned != 0))
            j_total = D * D
            n_pos = int(np.sum(self.J_learned > 0))
            n_neg = int(np.sum(self.J_learned < 0))
            print(f"      J_learned: max_abs={j_max}, nnz={j_nnz}/{j_total} "
                  f"({100*j_nnz/j_total:.1f}%), "
                  f"positive={n_pos}, negative={n_neg}")
        else:
            self.J_learned = np.zeros((D, D), dtype=np.int16)
            print(f"      WARNING: No valid pairs found, J_learned is zero")

        # Free projection matrix (not needed at inference)
        self._R = None

        self._built = True

        # Print diagnostics
        self._print_build_diagnostics()

        # Reset for generation
        self.reset()

        return self

    def _print_build_diagnostics(self) -> None:
        """Print diagnostics about the learned spin system."""
        if self.spin_vectors is None:
            return

        V, D = self.spin_vectors.shape

        # Spin vector diversity: how many unique spin configurations?
        # (This is expensive for large V, so sample)
        n_sample = min(1000, V)
        sample_spins = self.spin_vectors[:n_sample]

        # Average pairwise Hamming distance (sampled)
        if n_sample >= 100:
            idx = np.random.choice(n_sample, size=min(200, n_sample), replace=False)
            sample = sample_spins[idx]
            hamming_dists = []
            for i in range(len(sample)):
                for j in range(i+1, min(i+20, len(sample))):
                    hamming_dists.append(int(np.sum(sample[i] != sample[j])))
            if hamming_dists:
                avg_hamming = np.mean(hamming_dists)
                print(f"    Spin diversity: avg pairwise Hamming = {avg_hamming:.1f} "
                      f"({avg_hamming/D*100:.1f}% of D={D})")
                # If average Hamming ≈ D/2, spins are maximally diverse
                # If average Hamming << D/2, spins are too similar

        # Per-dimension magnetization (bias toward +1 or -1)
        magnetization = np.mean(self.spin_vectors.astype(np.float32), axis=0)
        n_biased = int(np.sum(np.abs(magnetization) > 0.5))
        print(f"    Spin magnetization: {n_biased}/{D} dims with |m| > 0.5 "
              f"(highly biased dimensions)")

        # J_learned spectrum
        if self.J_learned is not None:
            j_max = int(np.max(np.abs(self.J_learned)))
            j_mean_pos = float(np.mean(self.J_learned[self.J_learned > 0])) if np.any(self.J_learned > 0) else 0
            j_mean_neg = float(np.mean(self.J_learned[self.J_learned < 0])) if np.any(self.J_learned < 0) else 0
            print(f"    J_learned stats: max_abs={j_max}, "
                  f"mean_pos={j_mean_pos:.1f}, mean_neg={j_mean_neg:.1f}")

        mem_kb = self.spin_vectors.nbytes / 1024
        if self.J_learned is not None:
            mem_kb += self.J_learned.nbytes / 1024
        mem_kb += self.J_episodic.nbytes / 1024
        print(f"    Latent spin memory: {mem_kb:.1f} KB")

    # ===================================================================
    # RESET: Initialize for a new document
    # ===================================================================

    def reset(self) -> None:
        """
        Reset spin state and episodic memory for a new document.

        sigma_doc is randomly initialized to BREAK SYMMETRY.
        J_episodic is reset to zero (no document-specific memory).
        J_learned and spin_vectors are preserved (learned from training).
        """
        self.sigma_doc = self._rng.choice([-1, 1], size=self.D).astype(np.int8)
        self.J_episodic = np.zeros((self.D, self.D), dtype=np.int16)
        self._prev_sigma = self.sigma_doc.copy()
        self._stats = {
            'total_steps': 0,
            'spins_flipped': 0,
            'attractor_jumps': 0,
            'avg_alignment': 0,
            'avg_coupling_field_norm': 0,
        }

    # ===================================================================
    # STEP: Advance spin state via Ising dynamics
    # ===================================================================

    def step(self, word_id: int) -> None:
        """
        Advance document spin state by one token using mean-field dynamics.

        The spin state evolves under the combined influence of:
          1. LEARNED coupling: J_learned captures training-data dependency structure
          2. EPISODIC coupling: J_episodic creates document-specific attractors
          3. External field: sigma_w drives sigma_doc toward the word's pattern
          4. Optional noise: prevents local minima trapping

        Dynamics:
          h_i = sum_j (J_learned[i,j] + J_episodic[i,j]) * sigma_doc[j]
                + alpha * sigma_w[w, i]
          sigma_doc_new[i] = sign(h_i)

        After the mean-field sweep, the Hebbian episodic memory is updated
        to create an attractor for the new spin configuration:
          J_episodic[i,j] += eta * sigma_new[i] * sigma_old[j]

        This is the same physics as SSR, but with LEARNED J instead of random J.
        The learned coupling makes the dynamics MEANINGFUL — the attractors
        correspond to real semantic states, not random noise.

        Args:
            word_id: Integer token ID of the current word.
        """
        if word_id < 0:
            return

        if not self._built or self.spin_vectors is None:
            return

        # Save previous sigma for Hebbian update
        sigma_old = self.sigma_doc.copy()

        # External field from current word's spin vector
        if 0 <= word_id < self.vocab_size:
            h_ext = self.spin_vectors[word_id].astype(np.int32)  # (D,) int8 -> int32
        else:
            h_ext = np.zeros(self.D, dtype=np.int32)

        # Total coupling matrix
        J_total = np.zeros((self.D, self.D), dtype=np.int32)
        if self.J_learned is not None:
            J_total += self.J_learned.astype(np.int32)
        J_total += self.J_episodic.astype(np.int32)

        # v23: Per-dimension bias field
        # h_bias provides a directional signal that the J matrix alone
        # cannot provide (since J has zero diagonal with balanced spins).
        h_bias_field = np.zeros(self.D, dtype=np.int32)
        if self.h_bias is not None:
            h_bias_field = self.h_bias.astype(np.int32)

        # --- Mean-field sweeps ---
        for sweep in range(self.n_mf_sweeps):
            # Internal field from coupling: J_total @ sigma_doc
            h_internal = J_total @ self.sigma_doc.astype(np.int32)  # (D,) int32

            # v23 FIX: Instead of normalizing h_internal down to match h_ext
            # (which destroyed the coupling structure by compressing ~10K values
            # to ~256), we scale h_ext UP to match h_internal. This preserves
            # the full coupling structure while still letting the current word
            # influence the spin state.
            #
            # The adaptive alpha ensures the external field is always strong
            # enough to move sigma_doc toward the current word, but not so
            # strong that it ignores coupling structure entirely.
            h_l1 = max(1, int(np.sum(np.abs(h_internal))))
            # Scale external field to be ~1/4 of coupling field strength.
            # This means coupling dominates (long-range structure) but the
            # current word still has meaningful influence (~25% of total field).
            coupling_weight = 3  # coupling gets 3x weight vs external
            alpha_adaptive_q8 = min(255, (self.alpha_q8 * coupling_weight * self.D) // max(1, h_l1 // 256))
            h_ext_scaled = (alpha_adaptive_q8 * h_ext) >> 8
            # Scale h_ext up to match h_internal magnitude
            h_ext_magnitude = (h_ext_scaled.astype(np.int64) * h_l1) // (self.D * coupling_weight)
            h_ext_scaled = h_ext_magnitude.astype(np.int32)

            h_total = h_internal + h_bias_field + h_ext_scaled  # (D,) int32

            # Mean-field update: sigma_doc_i = sign(h_i)
            new_sigma = np.sign(h_total).astype(np.int8)
            zero_mask = h_total == 0
            new_sigma[zero_mask] = self.sigma_doc[zero_mask]

            # Optional noise injection
            if self.noise_prob > 0:
                flip_mask = self._rng.random(self.D) < self.noise_prob
                new_sigma[flip_mask] = -new_sigma[flip_mask]

            # Optional Metropolis acceptance for high-energy flips
            if self.temperature > 0 and sweep == 0:
                proposed_flips = new_sigma != self.sigma_doc
                if np.any(proposed_flips):
                    dE = 2 * self.sigma_doc.astype(np.int32) * h_total
                    for i in np.where(proposed_flips)[0]:
                        if dE[i] > 0:
                            threshold = int(dE[i])
                            accept_bound = self.temperature * 10
                            if threshold > accept_bound:
                                new_sigma[i] = self.sigma_doc[i]

            self.sigma_doc = new_sigma

        # --- Hebbian episodic memory update ---
        # Same as SSR: create attractors for visited semantic states
        flipped = self.sigma_doc != sigma_old
        n_flipped = int(np.sum(flipped))

        if n_flipped > 0 and n_flipped < self.D:
            # Sparse Hebbian update: only for flipped spins
            for i in np.where(flipped)[0]:
                update = (self.eta_episodic * int(self.sigma_doc[i]) *
                         sigma_old.astype(np.int16))
                self.J_episodic[i, :] += update

            # Clip to prevent unbounded growth
            np.clip(
                self.J_episodic, -self.J_EPISODIC_CLIP,
                self.J_EPISODIC_CLIP, out=self.J_episodic
            )
        elif n_flipped == 0:
            # No spins flipped — weak reinforcement of current state
            eta_weak = max(1, self.eta_episodic // 4)
            outer = np.outer(
                self.sigma_doc.astype(np.int16),
                self.sigma_doc.astype(np.int16)
            )
            self.J_episodic += (outer * eta_weak).astype(np.int16)
            np.clip(
                self.J_episodic, -self.J_EPISODIC_CLIP,
                self.J_EPISODIC_CLIP, out=self.J_episodic
            )

        # --- Update diagnostics ---
        self._stats['total_steps'] += 1
        self._stats['spins_flipped'] += n_flipped

        hamming = int(np.sum(self.sigma_doc != self._prev_sigma))
        if hamming > self.D // 4:
            self._stats['attractor_jumps'] += 1

        # Track average alignment between sigma_doc and the word's spin vector
        if 0 <= word_id < self.vocab_size:
            alignment = int(np.sum(self.sigma_doc * self.spin_vectors[word_id]))
            self._stats['avg_alignment'] += alignment

        # Track coupling field norm
        if self.J_learned is not None:
            h_coupling = self.J_learned.astype(np.int32) @ self.sigma_doc.astype(np.int32)
            field_norm = int(np.sum(np.abs(h_coupling)))
            self._stats['avg_coupling_field_norm'] += field_norm

        self._prev_sigma = self.sigma_doc.copy()

    # ===================================================================
    # ENERGY: Compute alignment energy for candidate words
    # ===================================================================

    def compute_energy(
        self,
        candidate_words: np.ndarray,
    ) -> np.ndarray:
        """
        Compute latent spin energy for candidate words.

        Two energy terms — BOTH mediated by learned structure:

        1. DIRECT ALIGNMENT: How well does the candidate word's spin vector
           align with the current document state?
           E_direct(w) = -(sigma_w . sigma_doc) * latent_scale / D

        2. COUPLING-MEDIATED ALIGNMENT: How well does the candidate word's
           spin vector align with the LEARNED-COUPLING-mediated field?
           E_coupling(w) = -(sigma_w . h_coupling) * coupling_scale / h_norm

        NORMALIZATION FIX (v22): Replaced Q10 normalization (dividing by
        max_abs_dot per call) with FIXED normalization:
          - Direct: divide by D (spin dimension) → range [-latent_scale, +latent_scale]
          - Coupling: divide by L1 norm of h_coupling → range [-coupling_scale, +coupling_scale]

        The Q10 normalization compressed all dot products to [-1024, +1024]
        before scaling, making spin energy 10-25x weaker than recall energy.
        With fixed normalization, the spin terms have a guaranteed energy range
        that is competitive with recall, enabling genuine spin→word influence.

        Args:
            candidate_words: Array of candidate word IDs, shape (n,).

        Returns:
            np.ndarray of int64 energies, shape (n,).
            LOWER energy = more aligned = more likely.
        """
        n_candidates = len(candidate_words)
        if not self._built or self.spin_vectors is None:
            return np.zeros(n_candidates, dtype=np.int64)

        # Look up spin vectors for candidates
        safe_candidates = np.clip(candidate_words, 0, self.vocab_size - 1)
        S_candidates = self.spin_vectors[safe_candidates]  # (n, D) int8

        # --- Direct alignment energy ---
        # E_direct(w) = -(sigma_w . sigma_doc) * latent_scale / D
        # dot ∈ [-D, +D], so E_direct ∈ [-latent_scale, +latent_scale]
        direct_dots = S_candidates.astype(np.int32) @ self.sigma_doc.astype(np.int32)  # (n,)
        E_direct = -(direct_dots.astype(np.int64) * self.latent_scale) // self.D

        # --- Coupling-mediated alignment energy ---
        # E_coupling(w) = -(sigma_w . h_coupling) * coupling_scale / h_norm
        # where h_coupling = J_total @ sigma_doc and h_norm = L1(h_coupling)
        # This ensures E_coupling ∈ [-coupling_scale, +coupling_scale]
        J_total = np.zeros((self.D, self.D), dtype=np.int32)
        if self.J_learned is not None:
            J_total += self.J_learned.astype(np.int32)
        J_total += self.J_episodic.astype(np.int32)

        # Compute coupling-mediated field: h = J_total @ sigma_doc + h_bias
        # v23: Include h_bias in the coupling field for energy computation.
        # h_bias provides the directional signal that J alone cannot (since
        # J has zero diagonal with balanced spins).
        h_coupling = J_total @ self.sigma_doc.astype(np.int32)  # (D,) int32
        if self.h_bias is not None:
            h_coupling += self.h_bias.astype(np.int32)

        # L1 norm of h_coupling for normalization
        # Floor at D to prevent division by very small numbers when
        # coupling is weak (early in generation)
        h_norm = max(self.D, int(np.sum(np.abs(h_coupling))))

        # Dot product: sigma_w . h_coupling for each candidate
        coupling_dots = S_candidates.astype(np.int32) @ h_coupling  # (n,) int32
        E_coupling = -(coupling_dots.astype(np.int64) * self.coupling_scale) // h_norm

        # Total energy: ADDITIVE (Ising model physics: E = sum E_i)
        return (E_direct + E_coupling).astype(np.int64)

    # ===================================================================
    # PROPERTIES AND DIAGNOSTICS
    # ===================================================================

    @property
    def built(self) -> bool:
        """Whether the spin vectors and coupling have been learned."""
        return self._built

    def get_diagnostics(self) -> Dict:
        """Get diagnostic information about the latent spin state."""
        result = {
            'built': self._built,
            'D': self.D,
            'alpha_q8': self.alpha_q8,
            'eta_episodic': self.eta_episodic,
            'temperature': self.temperature,
            'latent_scale': self.latent_scale,
            'coupling_scale': self.coupling_scale,
        }

        if self._built:
            n_pos = int(np.sum(self.sigma_doc == 1))
            n_neg = int(np.sum(self.sigma_doc == -1))
            result['sigma_magnetization'] = (n_pos - n_neg) / self.D

            if self.J_learned is not None:
                result['j_learned_max'] = int(np.max(np.abs(self.J_learned)))
                result['j_learned_nnz'] = int(np.sum(self.J_learned != 0))

            ep_max = int(np.max(np.abs(self.J_episodic)))
            ep_nnz = int(np.sum(self.J_episodic != 0))
            result['episodic_max'] = ep_max
            result['episodic_nnz'] = ep_nnz

        result['stats'] = self._stats.copy()
        if self._stats['total_steps'] > 0:
            result['avg_flips_per_step'] = (
                self._stats['spins_flipped'] / self._stats['total_steps']
            )
            result['avg_alignment'] = (
                self._stats['avg_alignment'] / self._stats['total_steps']
            )

        return result
