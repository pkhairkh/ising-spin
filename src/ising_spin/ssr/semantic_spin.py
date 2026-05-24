"""
Semantic Spin Resonance (SSR) — Core physics engine.

Implements a 256-dimensional binary Ising spin glass with:
  1. J_struct: Fixed structural coupling matrix (generic frustrated landscape)
  2. J_episodic: Adaptive episodic coupling (Hebbian learning during generation)
  3. W_word: Word-to-spin external field mapping (random sparse ternary)
  4. W_readout: Spin-to-word readout matrix (pre-aggregated from training)

The SPIN DYNAMICS create genuine long-range dependency:
  sigma(t+1) = sign(J_total @ sigma(t) + alpha * W_word[w(t)])

where J_total = J_struct + J_episodic.

The EPISODIC MEMORY is the key innovation:
  J_episodic += eta * outer(sigma_new, sigma_old)

This creates an attractor for the current semantic state. When the text
should return to a previous topic, J_episodic helps sigma find its way
back to the earlier attractor — GENUINE long-range recall.

Memory budget (D=256, V=4000):
  - J_struct:    256 x 256 x 1 byte  = 64 KB
  - J_episodic:  256 x 256 x 2 bytes = 128 KB
  - W_word:      4000 x 256 x 1 byte = 1 MB
  - W_readout:   4000 x 256 x 2 bytes = 2 MB
  - sigma:       256 x 1 byte        = negligible
  Total: ~3.2 MB (trivial on Pi 5)

Computation per token:
  - Mean-field sweep: 256 x 256 = 65K integer multiply-accumulates
  - Hebbian update: 256 x 256 = 65K integer additions (sparse, only for flipped spins)
  - Energy: 256 integer multiply-accumulates per candidate
  Total: ~200K integer ops/token (very fast on Pi 5 ARM cores)
"""

import numpy as np
from typing import Dict, List, Optional

from ..exceptions import ValidationError


class SemanticSpinResonance:
    """
    Semantic Spin Resonance — Emergent understanding via frustrated spin dynamics.

    The 256-dimensional binary spin vector sigma encodes the document's
    current semantic state as a DISTRIBUTED representation with NO
    pre-assigned meaning. Meaning EMERGES from:
      1. The frustrated coupling J_struct (competing constraints)
      2. The adaptive coupling J_episodic (document-specific memory)
      3. The external field from word selection (feedback)

    Unlike the ESN (linear, fixed random projection) or the macro-spin
    modules (hand-coded entity/phase/scene rules), SSR creates genuine
    understanding through:
      - NONLINEAR dynamics (sign function creates energy barriers)
      - ADAPTIVE couplings (J_episodic learns during generation)
      - FRUSTRATION (J_struct creates competing constraints)
      - ATTRACTOR DYNAMICS (spin configurations are metastable states)

    This is PURE SPIN GLASS PHYSICS. No attention, no state-space model,
    no hand-coded rules. Just Ising spins with learned couplings.

    All arithmetic is integer-only.
    """

    # Spin dimension — the "width" of the semantic representation
    # D=256 gives 2^256 possible spin configurations, vastly more than
    # the number of distinct semantic states in natural language.
    # This is the REPRESENTATIONAL CAPACITY that enables genuine understanding.
    DEFAULT_D = 256

    # Q8 fixed-point for external field strength
    # alpha controls how strongly the current word influences the spin state.
    # alpha_q8 = 255 means alpha ≈ 1.0 (strong influence, sigma tracks current word).
    # With normalized coupling field, alpha=1.0 gives balanced dynamics
    # where the current word's pattern competes equally with coupling structure.
    DEFAULT_ALPHA_Q8 = 255

    # Episodic Hebbian learning rate
    # eta_episodic controls how strongly the episodic memory forms.
    # Each token update adds eta_episodic to the coupling of spins
    # that are aligned (both +1 or both -1).
    DEFAULT_ETA_EPISODIC = 2

    # Number of mean-field sweeps per step
    # More sweeps = more relaxation = more coherent spin state
    # But also more computation. 2-3 sweeps is a good balance.
    DEFAULT_N_MF_SWEEPS = 2

    # Maximum absolute value for J_episodic (prevents unbounded growth)
    # After J_episodic entries reach this value, they are clipped.
    # This prevents early episodic memories from dominating forever.
    J_EPISODIC_CLIP = 200

    # Sparsity of J_struct (fraction of nonzero entries in lower triangle)
    # Sparse J_struct creates LOCAL frustration (each spin has ~25 neighbors)
    # Dense J_struct creates GLOBAL frustration (each spin has D-1 neighbors)
    # For D=256 with 15% sparsity: ~256*255*0.15/2 ≈ 4900 nonzero entries
    # For D=64 with 15% sparsity: ~64*63*0.15/2 ≈ 302 nonzero entries
    # This is enough to create frustration without overwhelming dynamics.
    J_STRUCT_SPARSITY = 0.15  # 15% nonzero entries

    # Q8 normalization for readout count normalization
    COUNT_NORM_Q = 256

    # Q10 normalization for energy dot products
    DOT_NORM_Q = 1024
    DOT_FLOOR = 100

    # Metropolis temperature for spin dynamics
    # At temperature T, a spin flip that increases energy by dE is accepted
    # with probability ~ exp(-dE / T). We implement this as an integer
    # threshold: flip if random_int(0, T) >= dE.
    # T=0: fully deterministic (greedy), T=inf: fully random
    DEFAULT_TEMPERATURE = 50

    # Sigma noise injection probability
    # With this probability, a random spin is flipped regardless of the field.
    # This prevents the spin state from getting stuck in a local minimum
    # and enables exploration of the energy landscape.
    DEFAULT_NOISE_PROB = 0.02  # 2% chance per spin per step

    def __init__(
        self,
        vocab_size: int,
        D: int = 256,
        alpha_q8: int = 128,
        eta_episodic: int = 2,
        n_mf_sweeps: int = 2,
        ssr_scale: int = 2000,
        temperature: int = 50,
        noise_prob: float = 0.02,
        seed: int = 42,
    ):
        """
        Initialize Semantic Spin Resonance.

        Args:
            vocab_size: Vocabulary size V.
            D: Spin dimension (default 256).
            alpha_q8: External field strength in Q8 (default 128 ≈ 0.5).
            eta_episodic: Hebbian learning rate for J_episodic (default 2).
            n_mf_sweeps: Mean-field sweeps per step (default 2).
            ssr_scale: Energy scale for word selection (default 1200).
            temperature: Metropolis temperature for spin flips (default 50).
            noise_prob: Probability of random spin flip per spin per step (default 0.02).
            seed: Random seed for deterministic initialization.
        """
        self.vocab_size = vocab_size
        self.D = D
        self.alpha_q8 = alpha_q8
        self.eta_episodic = eta_episodic
        self.n_mf_sweeps = n_mf_sweeps
        self.ssr_scale = ssr_scale
        self.temperature = temperature
        self.noise_prob = noise_prob
        self.seed = seed

        rng = np.random.RandomState(seed)

        # --- Word-to-spin external field mapping ---
        # W_word[w] maps word w to a D-dimensional external field.
        # Sparse ternary {-1, 0, +1} with ~33% each.
        # This is FIXED (like ESN input weights) — the semantics come
        # from the dynamics, not from the embedding.
        n_word_rows = min(vocab_size, 50000)
        self.W_word = rng.choice(
            [-1, 0, 1],
            size=(n_word_rows, D),
            p=[0.33, 0.34, 0.33],
        ).astype(np.int8)

        # --- Structural coupling matrix ---
        # J_struct creates the FRUSTRATED LANDSCAPE.
        # We generate a SYMMETRIC sparse matrix with balanced ±1 couplings.
        # Method: fill the lower triangle, mirror to upper triangle.
        # This preserves exact ±1 values (no averaging that destroys diversity).
        # The mix of positive and negative couplings creates FRUSTRATION:
        # spin i wants to align with spin j (J_ij > 0) but anti-align
        # with spin k (J_ik < 0), creating competing constraints.
        j_struct = np.zeros((D, D), dtype=np.int8)
        # Generate couplings only for the lower triangle + diagonal
        for i in range(D):
            for j in range(i):  # j < i → lower triangle
                if rng.random() < self.J_STRUCT_SPARSITY:
                    val = rng.choice([-1, 1])
                    j_struct[i, j] = val
                    j_struct[j, i] = val  # Mirror to upper triangle
        # Zero diagonal (no self-coupling)
        np.fill_diagonal(j_struct, 0)
        self.J_struct = j_struct

        # --- Spin state ---
        # sigma ∈ {-1, +1}^D — the semantic state vector
        # RANDOM initialization to break symmetry. Starting all-+1 would
        # cause the system to get stuck in the all-+1 attractor (no diversity).
        self.sigma = rng.choice([-1, 1], size=D).astype(np.int8)

        # --- Episodic coupling matrix ---
        # J_episodic starts at zero for each new document.
        # It is updated via Hebbian learning during generation:
        #   J_episodic[i,j] += eta * sigma_new[i] * sigma_old[j]
        # This creates ATTRACTORS for previously visited semantic states.
        self.J_episodic = np.zeros((D, D), dtype=np.int16)

        # --- Readout matrix (built during training) ---
        # W_readout[w] = Q * mean(sigma before word w) over training data.
        # During inference, E_ssr(w) = -(sigma . W_readout[w]) * ssr_scale / norm
        self.W_readout: Optional[np.ndarray] = None
        self._word_counts: Optional[np.ndarray] = None
        self._built = False

        # --- Random number generator for Metropolis dynamics ---
        self._rng = np.random.RandomState(seed + 9999)

        # --- Diagnostics ---
        self._stats = {
            'total_steps': 0,
            'spins_flipped': 0,
            'episodic_energy': 0,
            'attractor_jumps': 0,
            'sigma_hamming_distance': 0,
        }
        self._prev_sigma = self.sigma.copy()

    def reset(self) -> None:
        """
        Reset spin state and episodic memory for a new document.

        Sigma is randomly initialized to BREAK SYMMETRY (critical for
        generating diverse spin configurations). The frustrated dynamics
        will then shape sigma into a coherent state driven by the text.
        J_episodic is reset to zero (no document-specific memory).
        The structural coupling J_struct and word field W_word are preserved.
        """
        # Random initialization: each spin is independently ±1 with equal probability
        # This is CRITICAL — starting from all-+1 causes the system to get stuck
        # in a trivial ferromagnetic state with no useful information.
        self.sigma = self._rng.choice([-1, 1], size=self.D).astype(np.int8)
        self.J_episodic = np.zeros((self.D, self.D), dtype=np.int16)
        self._prev_sigma = self.sigma.copy()
        self._stats = {
            'total_steps': 0,
            'spins_flipped': 0,
            'episodic_energy': 0,
            'attractor_jumps': 0,
            'sigma_hamming_distance': 0,
        }

    def step(self, word_id: int) -> None:
        """
        Advance spin state by one token using mean-field dynamics.

        The spin state evolves under the combined influence of:
          1. Structural coupling: J_struct creates frustration
          2. Episodic coupling: J_episodic creates document-specific memory
          3. External field: W_word[w] drives sigma toward the word's pattern
          4. Noise: occasional random flips prevent local minima trapping

        After the mean-field sweep, the Hebbian episodic memory is updated
        to create an attractor for the new spin configuration.

        Dynamics:
          h_i = sum_j (J_struct[i,j] + J_episodic[i,j]) * sigma[j]
                + alpha * W_word[w, i]
          sigma_new[i] = sign(h_i)  with Metropolis noise

        Episodic update:
          J_episodic[i,j] += eta * sigma_new[i] * sigma_old[j]
          (only for spins that CHANGED — sparse update)

        Args:
            word_id: Integer token ID of the current word.
        """
        if word_id < 0:
            return

        # Save previous sigma for Hebbian update and diagnostics
        sigma_old = self.sigma.copy()

        # External field from current word
        if 0 <= word_id < self.W_word.shape[0]:
            h_ext = self.W_word[word_id].astype(np.int32)  # (D,) int8 -> int32
        else:
            h_ext = np.zeros(self.D, dtype=np.int32)

        # --- Mean-field sweeps ---
        # Each sweep updates all spins based on the current field.
        # Multiple sweeps allow the system to relax toward an energy minimum.
        J_total = self.J_struct.astype(np.int32) + self.J_episodic.astype(np.int32)

        for sweep in range(self.n_mf_sweeps):
            # Internal field from coupling: J_total @ sigma
            h_internal = J_total @ self.sigma.astype(np.int32)  # (D,) int32

            # v22 FIX: Normalize coupling field so it's comparable to the
            # external field (same fix as latent_spin.py). Without this,
            # the coupling field (~hundreds) completely dominates the
            # external field (~1), making sigma unresponsive to new words.
            h_norm = max(self.D, int(np.sum(np.abs(h_internal))))
            h_internal_normalized = (h_internal.astype(np.int64) * self.D) // h_norm

            # Total field: normalized internal + external (scaled by alpha)
            h_total = h_internal_normalized.astype(np.int32) + ((self.alpha_q8 * h_ext) >> 8)  # (D,) int32

            # Mean-field update: sigma_i = sign(h_i)
            # If h_i > 0: sigma_i = +1; if h_i < 0: sigma_i = -1; if h_i == 0: keep current
            new_sigma = np.sign(h_total).astype(np.int8)
            # Where h == 0, keep current sigma (rare but possible)
            zero_mask = h_total == 0
            new_sigma[zero_mask] = self.sigma[zero_mask]

            # Metropolis noise: with small probability, flip a random spin
            # This prevents the system from getting trapped in local minima
            if self.temperature > 0 and self.noise_prob > 0:
                # Random flips with probability noise_prob
                flip_mask = self._rng.random(self.D) < self.noise_prob
                new_sigma[flip_mask] = -new_sigma[flip_mask]

                # Additional Metropolis acceptance for high-energy flips
                # Compute energy change for each spin that wants to flip
                if sweep == 0:  # Only on first sweep (for efficiency)
                    # dE_i = 2 * sigma_i * h_i (energy change if spin i flips)
                    # If dE > 0, the flip increases energy -> reject with some probability
                    proposed_flips = new_sigma != self.sigma
                    if np.any(proposed_flips):
                        dE = 2 * self.sigma.astype(np.int32) * h_total
                        # Accept if dE <= 0 (energy decreases) or random < exp(-dE/T)
                        # Integer approximation: accept if random_int(0, T) >= dE
                        for i in np.where(proposed_flips)[0]:
                            if dE[i] > 0:
                                threshold = int(dE[i])
                                # Higher temperature -> more likely to accept
                                accept_bound = self.temperature * 10
                                if threshold > accept_bound:
                                    new_sigma[i] = self.sigma[i]  # Reject flip
                                    # (energy increase too large)

            self.sigma = new_sigma

        # --- Hebbian episodic memory update ---
        # This is the KEY INNOVATION. After each word, we update the
        # episodic coupling to create an ATTRACTOR for the current spin state.
        #
        # The Hebbian rule: J_episodic[i,j] += eta * sigma_new[i] * sigma_old[j]
        # This means: when spins i and j are aligned (both +1 or both -1),
        # their coupling increases, making them more likely to stay aligned.
        # When they're anti-aligned, the coupling decreases.
        #
        # SPARSE UPDATE: Only update for spins that CHANGED between sigma_old
        # and sigma_new. This makes the update O(D * n_flipped) instead of O(D^2),
        # and more importantly, it means the episodic memory focuses on the
        # TRANSITIONS between semantic states, not the states themselves.
        flipped = self.sigma != sigma_old
        n_flipped = int(np.sum(flipped))

        if n_flipped > 0 and n_flipped < self.D:
            # Sparse Hebbian update: only update rows corresponding to flipped spins
            # and columns corresponding to the OLD sigma values
            for i in np.where(flipped)[0]:
                # Update row i of J_episodic
                # J_episodic[i, :] += eta * sigma_new[i] * sigma_old[:]
                update = (self.eta_episodic * int(self.sigma[i]) *
                         sigma_old.astype(np.int16))
                self.J_episodic[i, :] += update

            # Clip to prevent unbounded growth
            np.clip(
                self.J_episodic, -self.J_EPISODIC_CLIP,
                self.J_EPISODIC_CLIP, out=self.J_episodic
            )
        elif n_flipped == 0:
            # No spins flipped — the current state is stable.
            # Still apply a WEAK Hebbian update to reinforce the attractor.
            # This is like "crystallizing" the current state.
            # Use a reduced eta for reinforcement (1/4 of normal)
            eta_weak = max(1, self.eta_episodic // 4)
            # Full outer product but with reduced strength
            # sigma * sigma^T gives +1 for aligned, -1 for anti-aligned
            outer = np.outer(
                self.sigma.astype(np.int16),
                self.sigma.astype(np.int16)
            )
            self.J_episodic += (outer * eta_weak).astype(np.int16)
            np.clip(
                self.J_episodic, -self.J_EPISODIC_CLIP,
                self.J_EPISODIC_CLIP, out=self.J_episodic
            )

        # --- Update diagnostics ---
        self._stats['total_steps'] += 1
        self._stats['spins_flipped'] += n_flipped

        # Track Hamming distance between consecutive states
        hamming = int(np.sum(self.sigma != self._prev_sigma))
        self._stats['sigma_hamming_distance'] += hamming

        # Detect "attractor jumps" — large Hamming distance indicates
        # a phase transition in the spin state
        if hamming > self.D // 4:  # More than 25% of spins flipped
            self._stats['attractor_jumps'] += 1

        # Episodic energy: trace of J_episodic @ sigma @ sigma^T / D
        # This measures how strongly the episodic memory is "pulling" on the current state
        if n_flipped > 0:
            ep_e = int(np.sum(self.J_episodic * np.outer(
                self.sigma.astype(np.int16), self.sigma.astype(np.int16)
            )))
            self._stats['episodic_energy'] += abs(ep_e) // self.D

        self._prev_sigma = self.sigma.copy()

    def compute_energy(
        self,
        candidate_words: np.ndarray,
    ) -> np.ndarray:
        """
        Compute SSR energy for candidate words.

        E_ssr(w) = -(sigma . W_readout[w]) * ssr_scale / R_L1[w]

        where R_L1[w] = sum(|W_readout[w,d]|) is the L1 norm of the readout
        vector, used as per-word normalization. This ensures:
          E_ssr ∈ [-ssr_scale, +ssr_scale] for each word
        regardless of the readout vector magnitude.

        NORMALIZATION FIX (v22): Replaced Q10 max_abs normalization with
        per-word L1 normalization. The Q10 approach compressed all candidates
        to the same range regardless of their readout magnitude, destroying
        the discriminative power. L1 normalization preserves the relative
        alignment signal while keeping the energy scale fixed.

        Words whose readout vector ALIGNS with the current spin state
        get lower energy (more likely under Boltzmann sampling).

        Args:
            candidate_words: Array of candidate word IDs, shape (n,).

        Returns:
            np.ndarray of int64 energies, shape (n,).
            LOWER energy = more aligned with spin state = more likely.
        """
        n_candidates = len(candidate_words)
        if not self._built or self.W_readout is None:
            return np.zeros(n_candidates, dtype=np.int64)

        # Look up readout vectors for candidates
        safe_candidates = np.clip(candidate_words, 0, self.vocab_size - 1)
        R_candidates = self.W_readout[safe_candidates]  # (n, D) int16

        # Dot product: sigma . W_readout[w] for each candidate
        # sigma is {-1, +1} int8, W_readout is int16
        dots = R_candidates.astype(np.int32) @ self.sigma.astype(np.int32)  # (n,)

        # Per-word L1 normalization: E = -(dot * ssr_scale) / R_L1
        # R_L1[w] = sum(|W_readout[w,d]|) = maximum possible dot for word w
        # This gives E ∈ [-ssr_scale, +ssr_scale] for each word
        R_L1 = np.sum(np.abs(R_candidates), axis=1).astype(np.int64)  # (n,)
        R_L1 = np.maximum(R_L1, 1)  # Avoid division by zero for unseen words

        energies = -(dots.astype(np.int64) * self.ssr_scale) // R_L1

        return energies.astype(np.int64)

    # ===================================================================
    # BUILD: Learn readout matrix from training data
    # ===================================================================

    def build(
        self,
        sequences: List[List[int]],
        max_sequences: Optional[int] = None,
    ) -> "SemanticSpinResonance":
        """
        Build the readout matrix W_readout from training sequences.

        For each training document:
          1. Initialize sigma = all +1
          2. For each word w_t:
             a. Record sigma BEFORE feeding word w_t
             b. Run SSR step (mean-field dynamics + Hebbian update)
          3. Accumulate sigma_before[w] for each word w

        After processing all documents:
          W_readout[w] = Q8 * mean(sigma_before_w) / max(1, count[w])

        This pre-aggregation captures what the spin state looks like
        BEFORE each word. During generation, words whose readout vectors
        align with the current spin state get lower energy.

        CRITICAL DIFFERENCE from ESN pre-aggregation:
        - ESN: Linear state (int16), exponential decay
        - SSR: Binary state ({-1,+1}), frustrated dynamics with
               structural coupling AND Hebbian episodic memory
        The SSR readout captures the ATTRACTOR STRUCTURE of the spin
        dynamics, not just the linear autocorrelation.

        Args:
            sequences: List of training sequences (word ID lists).
            max_sequences: Cap on number of sequences (None = all).

        Returns:
            self
        """
        V = self.vocab_size
        D = self.D

        n_seqs = len(sequences)
        if max_sequences is not None:
            n_seqs = min(n_seqs, max_sequences)

        # Accumulate readout in int32
        R_sum = np.zeros((V, D), dtype=np.int32)
        word_counts = np.zeros(V, dtype=np.int32)

        print(f"    Building SSR readout ({n_seqs} sequences, D={D})...")

        # Track global J_episodic across all training sequences
        # (This is like learning a "general" episodic memory from training)
        J_train_episodic = np.zeros((D, D), dtype=np.int16)

        for seq_idx in range(n_seqs):
            seq = sequences[seq_idx]

            # Reset for new document
            self.sigma = self._rng.choice([-1, 1], size=D).astype(np.int8)
            # Reset episodic to the TRAINING accumulated one
            # (This allows the model to build up episodic structure
            # across similar documents in the training set)
            self.J_episodic = J_train_episodic.copy()

            for pos in range(len(seq)):
                word_id = seq[pos]

                # Record spin state BEFORE this word (context state)
                if pos > 0 and 0 <= word_id < V:
                    R_sum[word_id] += self.sigma.astype(np.int32)
                    word_counts[word_id] += 1

                # Advance spin state with this word
                self.step(word_id)

            # Periodically update the training episodic memory
            # Average the current J_episodic back into J_train_episodic
            # This slowly accumulates the "average" episodic structure
            if (seq_idx + 1) % 1000 == 0:
                # Blend: 90% old + 10% new (slow adaptation)
                J_train_episodic = (
                    (J_train_episodic.astype(np.int32) * 9 +
                     self.J_episodic.astype(np.int32)) // 10
                ).astype(np.int16)

            if (seq_idx + 1) % 50000 == 0:
                n_words = int(np.sum(word_counts > 0))
                print(f"      SSR readout: {seq_idx+1}/{n_seqs} seqs, "
                      f"{n_words} words with features")

        # Count-normalize: W_readout[w] = R_sum[w] * Q8 / max(1, count[w])
        counts_safe = np.maximum(word_counts, 1)[:, np.newaxis]  # (V, 1)
        normalized = (R_sum * self.COUNT_NORM_Q) // counts_safe  # (V, D) int32
        R_norm = np.clip(normalized, -32768, 32767).astype(np.int16)
        zero_mask = word_counts == 0
        R_norm[zero_mask] = 0

        self.W_readout = R_norm
        self._word_counts = word_counts
        self._built = True

        # Save the training-accumulated episodic memory as a "warm start"
        # for generation. This is optional — J_episodic will still adapt
        # during generation — but it provides a better starting point.
        self.J_warm_start = J_train_episodic.copy()

        n_nonzero = int(np.sum(word_counts > 0))
        mem_kb = self.W_readout.nbytes / 1024
        print(f"    SSR readout: {n_nonzero} words with features, "
              f"memory={mem_kb:.1f} KB")

        # Diagnostics: analyze the spin dynamics during training
        self._print_training_diagnostics()

        # Reset state after building
        self.reset()

        return self

    def _print_training_diagnostics(self) -> None:
        """Print diagnostics about the SSR training process."""
        if self.W_readout is None:
            return

        # Analyze readout structure
        V, D = self.W_readout.shape

        # Per-word readout norm (measures how distinctive each word's
        # spin pattern is)
        norms = np.sqrt(np.sum(self.W_readout.astype(np.float32)**2, axis=1))
        nonzero_norms = norms[norms > 0]

        if len(nonzero_norms) > 0:
            print(f"    SSR readout norms: mean={np.mean(nonzero_norms):.1f}, "
                  f"std={np.std(nonzero_norms):.1f}, "
                  f"min={np.min(nonzero_norms):.1f}, "
                  f"max={np.max(nonzero_norms):.1f}")

        # Analyze J_struct frustration
        n_positive = int(np.sum(self.J_struct > 0))
        n_negative = int(np.sum(self.J_struct < 0))
        n_zero = int(np.sum(self.J_struct == 0))
        frustration_ratio = n_negative / max(1, n_positive + n_negative)
        print(f"    J_struct: {n_positive} positive, {n_negative} negative, "
              f"{n_zero} zero, frustration={frustration_ratio:.3f}")

        # Spin flip statistics
        if self._stats['total_steps'] > 0:
            avg_flips = self._stats['spins_flipped'] / self._stats['total_steps']
            avg_hamming = self._stats['sigma_hamming_distance'] / self._stats['total_steps']
            print(f"    SSR dynamics: avg {avg_flips:.1f} spins flipped/step, "
                  f"avg Hamming distance {avg_hamming:.1f}, "
                  f"{self._stats['attractor_jumps']} attractor jumps")

    @property
    def built(self) -> bool:
        """Whether the readout matrix has been built."""
        return self._built

    def get_diagnostics(self) -> Dict:
        """Get diagnostic information about the current SSR state."""
        result = {
            'built': self._built,
            'D': self.D,
            'alpha_q8': self.alpha_q8,
            'eta_episodic': self.eta_episodic,
            'temperature': self.temperature,
            'n_mf_sweeps': self.n_mf_sweeps,
        }

        if self._built:
            # Current spin state analysis
            n_pos = int(np.sum(self.sigma == 1))
            n_neg = int(np.sum(self.sigma == -1))
            result['sigma_positive'] = n_pos
            result['sigma_negative'] = n_neg
            result['sigma_magnetization'] = (n_pos - n_neg) / self.D

            # Episodic memory analysis
            ep_max = int(np.max(np.abs(self.J_episodic))) if self.J_episodic.size > 0 else 0
            ep_nnz = int(np.sum(self.J_episodic != 0))
            ep_total = self.D * self.D
            result['episodic_max'] = ep_max
            result['episodic_sparsity'] = 1.0 - ep_nnz / ep_total
            result['episodic_nnz'] = ep_nnz

        # Dynamic statistics
        result['stats'] = self._stats.copy()
        if self._stats['total_steps'] > 0:
            result['avg_flips_per_step'] = (
                self._stats['spins_flipped'] / self._stats['total_steps']
            )
            result['avg_hamming'] = (
                self._stats['sigma_hamming_distance'] / self._stats['total_steps']
            )

        return result
