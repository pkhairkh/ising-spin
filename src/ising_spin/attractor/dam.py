"""
Dense Associative Memory (DAM) — F-lookup energy, PCD learning.

THE CORE INSIGHT (Ramsauer et al. 2020, Krotov & Hopfield 2016):
  A Dense Associative Memory with higher-order energy function
  E = -Σ F(J_ij * s_i * s_j)
  is MATHEMATICALLY EQUIVALENT to softmax attention in transformers.

  - F(x) = x  → standard Hopfield, LINEAR capacity ≈ 0.14*D
  - F(x) = x² → polynomial capacity ≈ D^(β-1) for β-th order
  - F(x) = exp(x) → EXPONENTIAL capacity (modern Hopfield)

  The F-lookup table implements this nonlinearity in integer arithmetic.

SPARSE BINARY STATES:
  States s ∈ {0,1}^D with k active bits (k << D).
  Energy computation only visits k*(k-1)/2 active pairs — O(k²) not O(D²).
  For D=512, k=10: 45 pairs vs 130K — 3000x speedup!

PCD LEARNING (Predictive Contrastive Divergence):
  ΔJ_ij = η * (c_ij^data - c_ij^model)
  where:
    c_ij^data = s_i^data * s_j^data  (correlations with target clamped)
    c_ij^model = s_i^model * s_j^model (correlations after attractor relaxation)
  This sculpts the energy landscape so good predictions are deep attractors.

UV-COMPLETE REGULARIZATION:
  Couplings J must be "renormalizable" — coarse-graining J from level l
  must produce a well-defined effective theory at level l+1.
  This is enforced by regularizing the coupling spectrum during training:
    - Coupling distribution must have finite variance
    - No "irrelevant" couplings that diverge under RG flow
    - UV fixed point: coupling spectrum is stable under repeated coarse-graining

INTEGER-ONLY:
  - J: int16 coupling matrix
  - F_lookup: int32 energy lookup table
  - All dot products: integer multiply-accumulate
  - kWTA: integer sort + threshold

Memory (D=512):
  - J: 512 × 512 × 2 bytes = 512 KB
  - F_lookup: ~2000 entries × 4 bytes = 8 KB
  - state: 512 × 1 byte = negligible
  Total: ~520 KB per layer
"""

import numpy as np
from typing import Optional, Tuple


class DAMLayer:
    """
    Dense Associative Memory layer with F-lookup energy and PCD learning.

    The energy function uses a nonlinear F that gives exponential storage
    capacity. The F-lookup table converts integer coupling*state products
    into energy contributions, all in integer arithmetic.

    PCD learning sculpts the energy landscape so that context-appropriate
    predictions are deep attractors and spurious states are shallow.
    """

    # F-lookup range: J_ij * s_i * s_j ∈ [-J_MAX, +J_MAX]
    J_MAX = 1000  # Maximum absolute coupling value (int16 range is 32767)

    # F function type: controls the energy nonlinearity
    # 'quadratic': F(x) = x² for x > 0, 0 for x ≤ 0 (good capacity, integer-friendly)
    # 'cubic': F(x) = x³ (higher capacity)
    # 'exp_approx': F(x) ≈ exp(x/T) piecewise (exponential capacity)
    F_QUADRATIC = 0
    F_CUBIC = 1
    F_EXP_APPROX = 2

    def __init__(
        self,
        D: int = 512,
        k: int = 10,
        energy_beta: int = 2,
        scale: int = 1600,
        learning_rate: int = 1,
        n_dream_steps: int = 3,
        j_clip: int = 500,
        uv_regularize: bool = True,
        uv_lambda: int = 5,
        seed: int = 42,
    ):
        """
        Args:
            D: State dimension.
            k: Number of active bits (sparsity).
            energy_beta: Polynomial degree for F function (2=quadratic, 3=cubic).
            scale: Energy scale for DAM attractor dynamics.
            learning_rate: PCD learning rate (integer).
            n_dream_steps: Number of attractor relaxation steps in PCD.
            j_clip: Maximum absolute coupling value (prevents unbounded growth).
            uv_regularize: Whether to apply UV-complete regularization.
            uv_lambda: UV regularization strength.
            seed: Random seed.
        """
        self.D = D
        self.k = k
        self.energy_beta = energy_beta
        self.scale = scale
        self.learning_rate = learning_rate
        self.n_dream_steps = n_dream_steps
        self.j_clip = j_clip
        self.uv_regularize = uv_regularize
        self.uv_lambda = uv_lambda
        self.seed = seed

        # Coupling matrix J: (D, D) int16 — learned via PCD
        self.J = np.zeros((D, D), dtype=np.int16)

        # External field h: (D,) int16 — learned bias
        self.h = np.zeros(D, dtype=np.int16)

        # F-lookup table: maps J_ij * s_i * s_j to energy contribution
        # Indexed by value in range [-J_MAX, +J_MAX]
        self.F_lookup: Optional[np.ndarray] = None

        # Current state: sparse binary vector
        self.state = np.zeros(D, dtype=np.uint8)

        # RNG
        self._rng = np.random.RandomState(seed)

        # Build F-lookup table
        self._build_F_lookup()

        # Diagnostics
        self._stats = {
            'total_steps': 0,
            'total_pcd_updates': 0,
            'avg_data_correlation': 0,
            'avg_model_correlation': 0,
        }

    def _build_F_lookup(self) -> None:
        """
        Build the F-lookup table for integer energy computation.

        F is the nonlinear energy function that gives DAM its exponential
        storage capacity. For integer arithmetic, we precompute F(x) for
        all possible values of x = J_ij * s_i * s_j.

        For sparse binary states s ∈ {0,1}:
          x = J_ij * s_i * s_j ∈ {0, J_ij}
        So the lookup only needs entries for [0, J_MAX].
        We also store negative values for completeness.

        F functions:
          quadratic: F(x) = x² for x > 0, 0 for x ≤ 0
          cubic: F(x) = x³ for x > 0, 0 for x ≤ 0
          exp_approx: F(x) ≈ scale * (1 + x/T) for x > 0
        """
        J_MAX = self.J_MAX
        # Lookup range: [-J_MAX, +J_MAX], offset by J_MAX
        lookup_size = 2 * J_MAX + 1
        F = np.zeros(lookup_size, dtype=np.int64)

        for x in range(-J_MAX, J_MAX + 1):
            idx = x + J_MAX
            if self.energy_beta == 2:
                # Quadratic: F(x) = max(0, x)²
                # For x ≤ 0: no energy contribution (no attraction)
                # For x > 0: quadratic attraction (strong attractors)
                F[idx] = max(0, x) * max(0, x)
            elif self.energy_beta == 3:
                # Cubic: F(x) = sign(x) * |x|³
                F[idx] = (x * x * x) // 1000  # Scale down to prevent overflow
            else:
                # Linear (standard Hopfield): F(x) = x
                F[idx] = x

        self.F_lookup = F
        self._F_offset = J_MAX  # Index offset for negative values

    def compute_energy(self, state: np.ndarray) -> int:
        """
        Compute DAM energy for a sparse binary state.

        E = -Σ_{i<j} F_lookup[J_ij * s_i * s_j] - Σ_i h_i * s_i

        For sparse states, only active pairs contribute:
        O(k²) instead of O(D²).

        Args:
            state: Binary vector (D,) uint8.

        Returns:
            Integer energy value.
        """
        active = np.where(state > 0)[0]
        k = len(active)
        if k < 2:
            return 0

        # Compute energy from active pairs only
        energy = 0
        for ii in range(k):
            i = active[ii]
            # External field contribution
            energy -= int(self.h[i]) * int(state[i])
            for jj in range(ii + 1, k):
                j = active[jj]
                # Coupling contribution: F(J_ij * s_i * s_j)
                x = int(self.J[i, j]) * int(state[i]) * int(state[j])
                # Clamp to lookup range
                x = max(-self.J_MAX, min(self.J_MAX, x))
                energy -= int(self.F_lookup[x + self._F_offset])

        return energy

    def compute_energy_batch(
        self,
        states: np.ndarray,
    ) -> np.ndarray:
        """
        Compute DAM energy for multiple states.

        More efficient than calling compute_energy in a loop
        because we can vectorize the field computation.

        Args:
            states: Binary matrix (N, D) uint8.

        Returns:
            Energy array (N,) int64.
        """
        N = states.shape[0]

        # External field: -h · s for each state
        field_energy = -(states.astype(np.int32) @ self.h.astype(np.int32))

        # Coupling energy: for each state, -Σ F(J_ij * s_i * s_j)
        # For sparse states, this is more efficient with active-bit iteration
        coupling_energy = np.zeros(N, dtype=np.int64)

        for n in range(N):
            active = np.where(states[n] > 0)[0]
            k = len(active)
            if k < 2:
                continue

            for ii in range(k):
                i = active[ii]
                for jj in range(ii + 1, k):
                    j = active[jj]
                    x = int(self.J[i, j]) * int(states[n, i]) * int(states[n, j])
                    x = max(-self.J_MAX, min(self.J_MAX, x))
                    coupling_energy[n] -= int(self.F_lookup[x + self._F_offset])

        return field_energy.astype(np.int64) + coupling_energy

    def compute_field(self, state: np.ndarray) -> np.ndarray:
        """
        Compute the local field for each unit: h_i = Σ_j J_ij * s_j + h_i.

        This is the input to the kWTA decision. Units with the highest
        field values are the ones that should be active.

        For sparse states, only active units contribute to the field:
        h_i = Σ_{j active} J_ij + h_i

        Args:
            state: Binary vector (D,) uint8.

        Returns:
            Field vector (D,) int32.
        """
        # J @ state (only active columns contribute)
        active = np.where(state > 0)[0]
        if len(active) == 0:
            return self.h.astype(np.int32).copy()

        field = self.J[:, active].astype(np.int32) @ state[active].astype(np.int32)
        field += self.h.astype(np.int32)

        return field

    def step(
        self,
        context_field: np.ndarray,
        n_sweeps: int = 3,
    ) -> np.ndarray:
        """
        Run attractor dynamics for n_sweeps mean-field iterations.

        At each sweep:
          1. Compute total field: h_total = J @ state + h + context_field
          2. kWTA: keep top k units of h_total as the new state

        The context_field comes from:
          - SDR encoding of context words (bottom-up)
          - Higher hierarchical layers (top-down)
          - Episodic memory retrieval

        Args:
            context_field: External field (D,) int32.
            n_sweeps: Number of mean-field sweeps.

        Returns:
            New state (D,) uint8 with exactly k active bits.
        """
        state = self.state.copy()

        for _ in range(n_sweeps):
            # Internal field from couplings
            internal_field = self.compute_field(state)

            # Total field
            total_field = internal_field + context_field

            # kWTA: select top k units
            top_k = np.argpartition(total_field, -self.k)[-self.k:]
            state = np.zeros(self.D, dtype=np.uint8)
            state[top_k] = 1

        self.state = state
        return state

    def store_hebbian(
        self,
        context_sdr: np.ndarray,
        target_sdr: np.ndarray,
        eta: int = 1,
    ) -> None:
        """
        Store an association using Hebbian learning.

        J_ij += eta * target_i * context_j (only for active pairs)

        This is the one-shot learning rule. It's fast but can accumulate
        interference. PCD learning (below) refines these couplings.

        For sparse SDRs, only k_target * k_context elements are updated.

        Args:
            context_sdr: Context SDR (D,) uint8.
            target_sdr: Target SDR (D,) uint8.
            eta: Learning rate (integer).
        """
        target_active = np.where(target_sdr > 0)[0]
        context_active = np.where(context_sdr > 0)[0]

        # Hebbian update: J[target_i, context_j] += eta
        for i in target_active:
            for j in context_active:
                if i != j:
                    new_val = int(self.J[i, j]) + eta
                    new_val = max(-self.j_clip, min(self.j_clip, new_val))
                    self.J[i, j] = np.int16(new_val)
                    self.J[j, i] = np.int16(new_val)  # Symmetric

        # Update bias: h[target_i] += eta
        for i in target_active:
            new_val = int(self.h[i]) + eta
            new_val = max(-self.j_clip, min(self.j_clip, new_val))
            self.h[i] = np.int16(new_val)

    def pcd_update(
        self,
        context_sdr: np.ndarray,
        target_sdr: np.ndarray,
        eta: Optional[int] = None,
    ) -> None:
        """
        PCD (Predictive Contrastive Divergence) learning update.

        Phase 1 (DATA): Correlations with target clamped
          c_ij^data = target_i * context_j

        Phase 2 (MODEL): Run attractor dynamics, compute correlations
          c_ij^model = model_i * context_j

        Update: ΔJ_ij = η * (c_ij^data - c_ij^model)

        This sculpts the energy landscape so the correct target is
        a deeper attractor than the model's current prediction.

        UV-COMPLETE REGULARIZATION:
          After each update, we regularize the coupling matrix to ensure
          it's "renormalizable" — its spectrum should be well-behaved
          under coarse-graining. This prevents irrelevant couplings from
          accumulating and destabilizing the hierarchy.

        Args:
            context_sdr: Context SDR (D,) uint8.
            target_sdr: Target SDR (D,) uint8.
            eta: Learning rate (default: self.learning_rate).
        """
        if eta is None:
            eta = self.learning_rate

        # --- DATA PHASE ---
        # Correlations: target_i * context_j for active pairs
        target_active = np.where(target_sdr > 0)[0]
        context_active = np.where(context_sdr > 0)[0]

        # --- MODEL PHASE ---
        # Run attractor dynamics from context
        context_field = np.zeros(self.D, dtype=np.int32)
        context_field[context_active] = self.scale  # Strong context drive

        old_state = self.state.copy()
        model_state = self.step(context_field, n_sweeps=self.n_dream_steps)
        model_active = np.where(model_state > 0)[0]

        # --- COUPLING UPDATE ---
        # ΔJ_ij = η * (data_corr - model_corr)
        # data_corr[i,j] = target_i * context_j
        # model_corr[i,j] = model_i * context_j

        # Compute updates for all affected pairs
        # Active in data but not model: strengthen coupling
        for i in target_active:
            for j in context_active:
                if i != j:
                    delta = eta  # data_corr = 1
                    if model_sdr_active(i, model_state):
                        delta -= eta  # model_corr = 1
                    new_val = int(self.J[i, j]) + delta
                    new_val = max(-self.j_clip, min(self.j_clip, new_val))
                    self.J[i, j] = np.int16(new_val)
                    self.J[j, i] = np.int16(new_val)

        # Active in model but not data: weaken coupling
        for i in model_active:
            if target_sdr[i] == 0:  # Not in data
                for j in context_active:
                    if i != j:
                        delta = -eta  # model_corr = 1, data_corr = 0
                        new_val = int(self.J[i, j]) + delta
                        new_val = max(-self.j_clip, min(self.j_clip, new_val))
                        self.J[i, j] = np.int16(new_val)
                        self.J[j, i] = np.int16(new_val)

        # Bias update
        for i in target_active:
            new_val = int(self.h[i]) + eta
            new_val = max(-self.j_clip, min(self.j_clip, new_val))
            self.h[i] = np.int16(new_val)
        for i in model_active:
            if target_sdr[i] == 0:
                new_val = int(self.h[i]) - eta
                new_val = max(-self.j_clip, min(self.j_clip, new_val))
                self.h[i] = np.int16(new_val)

        # --- UV-COMPLETE REGULARIZATION ---
        if self.uv_regularize:
            self._uv_regularize()

        # Restore state
        self.state = old_state

        # Diagnostics
        self._stats['total_pcd_updates'] += 1

    def _uv_regularize(self) -> None:
        """
        UV-complete regularization: ensure couplings are renormalizable.

        In Wilsonian RG, a theory is UV-complete if its couplings
        remain well-defined under repeated coarse-graining. This means:

        1. The coupling distribution has finite variance
        2. No "dangerous irrelevant" couplings that grow under RG flow
        3. The coupling spectrum is compatible with a UV fixed point

        Implementation:
          - L2-like regularization: shrink weak couplings toward zero
          - This mimics the RG flow: irrelevant operators (weak couplings)
            decay under coarse-graining. By explicitly shrinking them,
            we ensure the effective theory at the next level is clean.
          - Strong couplings (relevant operators) are preserved — they
            carry the meaningful structure.

        The regularization strength (uv_lambda) controls how aggressively
        irrelevant couplings are suppressed. Higher lambda = more
        aggressive = cleaner coarse-graining but potentially less capacity.
        """
        if self.uv_lambda <= 0:
            return

        # Decay weak couplings toward zero (irrelevant operators decay under RG)
        # Strong couplings (|J| > threshold) are preserved (relevant operators)
        abs_J = np.abs(self.J)
        threshold = self.uv_lambda

        # Shrink couplings below threshold
        weak_mask = (abs_J > 0) & (abs_J <= threshold)
        # Reduce by 1 (minimum decay)
        shrink = np.where(weak_mask, np.int16(1), np.int16(0))
        self.J -= shrink * np.sign(self.J).astype(np.int16)

    def store_batch_hebbian(
        self,
        context_sdrs: np.ndarray,
        target_sdrs: np.ndarray,
        eta: int = 1,
    ) -> None:
        """
        Batch Hebbian storage for efficient training.

        Instead of storing one pattern at a time, we accumulate all
        the updates and apply them at once. This is equivalent to
        the Hopfield outer-product rule:

        J = Σ_n η * target_n ⊗ context_n

        Then clip to [-j_clip, +j_clip].

        Args:
            context_sdrs: Context SDRs (N, D) uint8.
            target_sdrs: Target SDRs (N, D) uint8.
            eta: Learning rate.
        """
        N = context_sdrs.shape[0]

        # Batch outer product: J += η * target^T @ context
        # (D, N) @ (N, D) = (D, D)
        J_update = (
            target_sdrs.astype(np.int32).T @ context_sdrs.astype(np.int32)
        ) * eta

        # Zero diagonal (no self-coupling)
        np.fill_diagonal(J_update, 0)

        # Symmetrize
        J_update = (J_update + J_update.T) // 2

        # Apply update with clipping
        J_new = self.J.astype(np.int32) + J_update
        np.clip(J_new, -self.j_clip, self.j_clip, out=J_new)
        self.J = J_new.astype(np.int16)

        # Bias update
        h_update = np.sum(target_sdrs.astype(np.int32), axis=0) * eta
        h_new = self.h.astype(np.int32) + h_update
        np.clip(h_new, -self.j_clip, self.j_clip, out=h_new)
        self.h = h_new.astype(np.int16)

    def reset(self) -> None:
        """Reset state for a new document."""
        # Random initial state (k active bits)
        self.state = np.zeros(self.D, dtype=np.uint8)
        active = self._rng.choice(self.D, size=self.k, replace=False)
        self.state[active] = 1

    def get_diagnostics(self) -> dict:
        """Return layer diagnostics."""
        J = self.J
        return {
            'D': self.D,
            'k': self.k,
            'J_max': int(np.max(np.abs(J))),
            'J_mean_abs': float(np.mean(np.abs(J[J != 0]))) if np.any(J != 0) else 0,
            'J_nnz': int(np.sum(J != 0)),
            'J_total': self.D * self.D,
            'J_sparsity': float(np.sum(J == 0)) / (self.D * self.D),
            'h_max': int(np.max(np.abs(self.h))),
            'h_nnz': int(np.sum(self.h != 0)),
            'scale': self.scale,
            'memory_kb': (J.nbytes + self.h.nbytes) / 1024,
        }


def model_sdr_active(idx: int, state: np.ndarray) -> bool:
    """Check if unit idx is active in state."""
    return bool(state[idx] > 0)
