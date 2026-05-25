"""
Hierarchical DAM with Wilsonian RG Flow — UV-Complete Architecture.

WILSONIAN RENORMALIZATION GROUP (RG):
  In quantum field theory, the RG describes how couplings change under
  scale transformations. A theory is "UV-complete" if it remains
  well-defined at all energy scales (from UV/high-energy/short-distance
  to IR/low-energy/long-distance).

  Applied to our hierarchical spin glass:
    - L0 (lexical, D=512) → UV (high energy, short range)
    - L3 (discourse, D=64) → IR (low energy, long range)
    - RG flow: couplings transform as we move between scales

COARSE-GRAINING (Block-Spin Transformation):
  The mapping from level l to level l+1 is a block-spin transformation:
    1. Group D_l / D_{l+1} spins at level l into one block
    2. Block spin = majority vote (or weighted sum) of block spins
    3. Effective coupling at l+1 is derived from l's couplings

  This is EXACTLY the physics of renormalization: the effective theory
  at a coarser scale is derived by "integrating out" short-range degrees
  of freedom.

RG BETA FUNCTIONS:
  The beta function β(g) describes how a coupling g changes under RG flow:
    g_{l+1} = g_l - β(g_l) * Δl

  For our discrete hierarchy:
    g_{l+1} = β(g_l)

  Where β is computed from the block-spin transformation. This ensures
  the couplings at each level are consistent with the level below.

UV FIXED POINT:
  A UV fixed point g* satisfies β(g*) = 0 — the couplings don't change
  under further refinement. At the UV fixed point, the theory is
  scale-invariant. For our model, this means:
    - The coupling spectrum at L0 is such that coarse-graining produces
      a well-defined effective theory at L1
    - And recursively for all levels

  In practice: we regularize L0's couplings to be "renormalizable" —
  they must produce stable effective couplings at L1.

TOP-DOWN FEEDBACK (Relevant Operators):
  In RG, relevant operators grow under the flow from UV to IR. This means
  that the IR (coarse) theory has "memory" of the UV (fine) theory's
  relevant operators. In our model:

  Top-down feedback: L3 → L2 → L1 → L0
  The coarse levels provide context that constrains the fine levels.
  This is analogous to "relevant operators flowing back" from IR to UV.

  Implementation: each level l receives an external field from level l+1:
    h_topdown[l] = W_down[l] @ state[l+1] * topdown_scale

INTER-LAYER COUPLING:
  - W_up[l]: bottom-up projection (L_l → L_{l+1})
    Block-spin: D_l → D_{l+1} via majority/sum
  - W_down[l]: top-down projection (L_{l+1} → L_l)
    Upsampling: D_{l+1} → D_l via interpolation

ALL INTEGER ARITHMETIC. Runs on Pi 5.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .dam import DAMLayer
from .sdr import SDREncoder


class HierarchicalDAM:
    """
    Hierarchical Dense Associative Memory with Wilsonian RG flow.

    Four layers with increasing abstraction and decreasing dimension:
      L0: D=512, k=10 (2% sparse) — lexical patterns (individual words)
      L1: D=256, k=8 (3% sparse) — syntactic patterns (POS sequences)
      L2: D=128, k=5 (4% sparse) — semantic patterns (topic/concept)
      L3: D=64,  k=3 (5% sparse) — discourse patterns (narrative structure)

    Inter-layer coupling:
      - Bottom-up: block-spin coarse-graining (RG flow from UV to IR)
      - Top-down: relevant operator feedback (IR to UV)
      - RG beta functions: ensure coupling consistency across scales

    UV completeness:
      - Couplings at each level are regularized to be renormalizable
      - The coarse-graining operator produces well-defined effective theories
      - UV fixed point ensures stability at the finest scale
    """

    # Default layer configurations: (D, k, scale)
    DEFAULT_LAYERS = [
        (512, 10, 1600),  # L0: lexical
        (256, 8, 1200),   # L1: syntactic
        (128, 5, 800),    # L2: semantic
        (64, 3, 400),     # L3: discourse
    ]

    def __init__(
        self,
        layers_config: Optional[List[Tuple[int, int, int]]] = None,
        learning_rate: int = 1,
        n_dream_steps: int = 3,
        j_clip: int = 500,
        uv_regularize: bool = True,
        uv_lambda: int = 5,
        topdown_scale: int = 200,
        rg_beta_strength: int = 100,
        seed: int = 42,
    ):
        """
        Args:
            layers_config: List of (D, k, scale) for each layer.
            learning_rate: PCD learning rate.
            n_dream_steps: PCD dream steps.
            j_clip: Coupling clip value.
            uv_regularize: Enable UV-complete regularization.
            uv_lambda: UV regularization strength.
            topdown_scale: Scale for top-down feedback field.
            rg_beta_strength: Strength of RG beta function coupling.
            seed: Random seed.
        """
        if layers_config is None:
            layers_config = self.DEFAULT_LAYERS

        self.layers_config = layers_config
        self.n_layers = len(layers_config)
        self.learning_rate = learning_rate
        self.n_dream_steps = n_dream_steps
        self.j_clip = j_clip
        self.uv_regularize = uv_regularize
        self.uv_lambda = uv_lambda
        self.topdown_scale = topdown_scale
        self.rg_beta_strength = rg_beta_strength
        self.seed = seed

        # Create DAM layers
        self.layers: List[DAMLayer] = []
        for l, (D, k, scale) in enumerate(layers_config):
            layer = DAMLayer(
                D=D, k=k, scale=scale,
                learning_rate=learning_rate,
                n_dream_steps=n_dream_steps,
                j_clip=j_clip,
                uv_regularize=uv_regularize,
                uv_lambda=uv_lambda,
                seed=seed + l * 1000,
            )
            self.layers.append(layer)

        # Inter-layer coupling matrices (built during training)
        # W_up[l]: (D_{l+1}, D_l) — block-spin projection
        # W_down[l]: (D_l, D_{l+1}) — top-down feedback
        self.W_up: List[Optional[np.ndarray]] = [None] * (self.n_layers - 1)
        self.W_down: List[Optional[np.ndarray]] = [None] * (self.n_layers - 1)

        # RG beta functions: computed during training
        # beta_g[l] maps coupling strength at level l to level l+1
        self.rg_beta: List[Optional[float]] = [None] * (self.n_layers - 1)

        self._built = False
        self._rng = np.random.RandomState(seed)

    def build(self, sdr_encoder: SDREncoder) -> "HierarchicalDAM":
        """
        Build inter-layer coupling matrices and initialize layers.

        The block-spin transformation (W_up) maps fine-grained states
        to coarse-grained states. Each coarse dimension is a weighted
        sum of a block of fine dimensions.

        The top-down projection (W_down) maps coarse states back to
        fine states, providing context from higher levels.

        RG beta functions are initialized from the block-spin
        transformation and refined during training.

        Args:
            sdr_encoder: The SDR encoder (for dimension reference).

        Returns:
            self
        """
        for l in range(self.n_layers - 1):
            D_fine = self.layers_config[l][0]   # D at level l
            D_coarse = self.layers_config[l + 1][0]  # D at level l+1

            # Block-spin projection: D_fine → D_coarse
            # Each coarse dimension sums over D_fine/D_coarse fine dimensions
            block_size = D_fine // D_coarse
            assert D_fine % D_coarse == 0, \
                f"D_fine ({D_fine}) must be divisible by D_coarse ({D_coarse})"

            W_up = np.zeros((D_coarse, D_fine), dtype=np.int16)
            for c in range(D_coarse):
                start = c * block_size
                end = start + block_size
                W_up[c, start:end] = 1  # Uniform block spin

            self.W_up[l] = W_up

            # Top-down projection: D_coarse → D_fine
            # Each fine dimension receives from its block's coarse dimension
            W_down = np.zeros((D_fine, D_coarse), dtype=np.int16)
            for c in range(D_coarse):
                start = c * block_size
                end = start + block_size
                W_down[start:end, c] = 1  # Broadcast back to block

            self.W_down[l] = W_down

            # Initialize RG beta function
            # β(g) = g * (D_coarse / D_fine) — linearized RG flow
            # Stronger couplings at fine scale produce weaker effective
            # couplings at coarse scale (decimation reduces coupling)
            self.rg_beta[l] = D_coarse / D_fine

        self._built = True

        # Print diagnostics
        self._print_diagnostics()

        return self

    def coarse_grain(self, state: np.ndarray, level: int) -> np.ndarray:
        """
        Block-spin coarse-graining: project state from level l to level l+1.

        This is the Wilsonian RG step: "integrate out" short-range degrees
        of freedom to produce an effective theory at coarser scale.

        Implementation: weighted sum of fine-grained state, then kWTA
        to maintain sparsity at the coarse level.

        Args:
            state: Fine-grained state (D_l,) uint8.
            level: Level index l (coarse-grain from l to l+1).

        Returns:
            Coarse-grained state (D_{l+1},) uint8.
        """
        if level >= self.n_layers - 1:
            return state  # Can't coarse-grain the top level

        W = self.W_up[level]
        if W is None:
            return np.zeros(self.layers_config[level + 1][0], dtype=np.uint8)

        # Block-spin: weighted sum
        coarse_acc = W.astype(np.int32) @ state.astype(np.int32)

        # kWTA: keep top k at the coarse level
        D_coarse = self.layers_config[level + 1][0]
        k_coarse = self.layers_config[level + 1][1]
        top_k = np.argpartition(coarse_acc, -k_coarse)[-k_coarse:]

        coarse_state = np.zeros(D_coarse, dtype=np.uint8)
        coarse_state[top_k] = 1

        return coarse_state

    def compute_topdown_field(self, state: np.ndarray, level: int) -> np.ndarray:
        """
        Compute top-down feedback field from level l+1 to level l.

        This is the "relevant operator" flow from IR to UV. The coarse
        level provides context that constrains the fine level's dynamics.

        Implementation: W_down @ state * topdown_scale

        Args:
            state: Coarse-grained state (D_{l+1},) uint8.
            level: Level index l (feedback from l+1 to l).

        Returns:
            Field vector (D_l,) int32.
        """
        if level >= self.n_layers - 1:
            return np.zeros(self.layers_config[level][0], dtype=np.int32)

        W = self.W_down[level]
        if W is None:
            return np.zeros(self.layers_config[level][0], dtype=np.int32)

        # Top-down projection with scale
        field = W.astype(np.int32) @ state.astype(np.int32) * self.topdown_scale

        return field

    def step_all(
        self,
        l0_context_field: np.ndarray,
        n_sweeps: int = 3,
    ) -> List[np.ndarray]:
        """
        Run hierarchical attractor dynamics across all layers.

        Order of operations (per sweep):
          1. Bottom-up: L0 → L1 → L2 → L3 (coarse-graining)
          2. Each layer runs its own attractor dynamics
          3. Top-down: L3 → L2 → L1 → L0 (feedback)
          4. Final L0 step with all context

        This implements the RG flow: information flows from UV (fine)
        to IR (coarse), and relevant operators flow back.

        Args:
            l0_context_field: Context field for L0 (D0,) int32.
            n_sweeps: Number of hierarchical sweeps.

        Returns:
            List of states for each layer [(D0,), (D1,), (D2,), (D3,)].
        """
        states = [layer.state.copy() for layer in self.layers]

        for sweep in range(n_sweeps):
            # === BOTTOM-UP PASS: coarse-grain and update ===
            for l in range(self.n_layers - 1):
                # Coarse-grain current state
                coarse_state = self.coarse_grain(states[l], l)

                # Update coarse state via attractor dynamics
                # Context field = bottom-up projection from fine level
                bu_field = self.W_up[l].astype(np.int32) @ states[l].astype(np.int32) * 100
                states[l + 1] = self.layers[l + 1].step(bu_field, n_sweeps=1)

            # === TOP-DOWN PASS: feedback from coarse to fine ===
            for l in range(self.n_layers - 2, -1, -1):
                # Compute top-down field from level l+1 to level l
                td_field = self.compute_topdown_field(states[l + 1], l)

                # Run with combined context
                if l == 0:
                    total_field = l0_context_field + td_field
                else:
                    # Also include bottom-up projection from level l-1 to l
                    # This uses W_up[l-1] to project fine state to coarse field
                    W = self.W_up[l - 1]
                    bu_field = W.astype(np.int32) @ states[l - 1].astype(np.int32) * 100
                    total_field = bu_field + td_field

                states[l] = self.layers[l].step(total_field, n_sweeps=1)

        # Final L0 step with full context
        total_field = l0_context_field.copy()
        if self.n_layers > 1:
            total_field += self.compute_topdown_field(states[1], 0)
        states[0] = self.layers[0].step(total_field, n_sweeps=1)

        # Store states
        for l in range(self.n_layers):
            self.layers[l].state = states[l]

        return states

    def compute_energy(
        self,
        candidate_sdr: np.ndarray,
        context_sdr: np.ndarray,
    ) -> int:
        """
        Compute total hierarchical energy for a candidate word SDR.

        The energy includes:
          - L0 energy: direct alignment with context + coupling structure
          - Higher-level energies: consistency across scales
          - Inter-layer coupling energy: RG flow consistency

        Lower energy = more likely prediction.

        Args:
            candidate_sdr: Candidate word SDR (D0,) uint8.
            context_sdr: Context SDR (D0,) uint8.

        Returns:
            Integer energy value.
        """
        total_energy = 0

        # L0 energy: alignment between candidate and context through coupling
        # E_L0 = -Σ F(J_ij * candidate_i * context_j)
        active_cand = np.where(candidate_sdr > 0)[0]
        active_ctx = np.where(context_sdr > 0)[0]

        for i in active_cand:
            # Field from context
            total_energy -= int(self.layers[0].h[i])
            for j in active_ctx:
                if i != j:
                    x = int(self.layers[0].J[i, j])
                    x = max(-self.layers[0].J_MAX, min(self.layers[0].J_MAX, x))
                    total_energy -= int(self.layers[0].F_lookup[x + self.layers[0]._F_offset])

        # Higher-level energies: how well does the candidate fit at each scale?
        state_l = candidate_sdr.copy()
        for l in range(self.n_layers - 1):
            # Coarse-grain candidate
            coarse_candidate = self.coarse_grain(state_l, l)

            # Compute coarse context
            if l == 0:
                coarse_context = self.coarse_grain(context_sdr, l)
            else:
                coarse_context = self.layers[l + 1].state  # Already computed

            # Energy at this level
            level_energy = self.layers[l + 1].compute_energy(coarse_candidate)
            total_energy += level_energy * self.layers_config[l + 1][2] // self.layers_config[0][2]

            state_l = coarse_candidate

        return total_energy

    def compute_word_energies(
        self,
        context_sdr: np.ndarray,
        candidate_words: np.ndarray,
        sdr_encoder: SDREncoder,
        scale: int = 1600,
    ) -> np.ndarray:
        """
        Compute energy for each candidate word given context.

        This is the main interface for the generator: given a context
        SDR and a list of candidate words, compute the energy of each
        candidate. Lower energy = more likely.

        Uses efficient batch computation where possible.

        Args:
            context_sdr: Context SDR (D0,) uint8.
            candidate_words: Array of candidate word IDs.
            sdr_encoder: SDR encoder for word→SDR mapping.
            scale: Energy scale multiplier.

        Returns:
            Energy array (len(candidate_words),) int64.
        """
        n_cand = len(candidate_words)
        energies = np.zeros(n_cand, dtype=np.int64)

        # L0 coupling field from context
        context_active = np.where(context_sdr > 0)[0]
        D0 = self.layers_config[0][0]

        # Pre-compute context field: J[:, active_ctx] @ ones(k_ctx) + h
        if len(context_active) > 0:
            context_field = (
                self.layers[0].J[:, context_active].astype(np.int32) @
                context_sdr[context_active].astype(np.int32)
            )
        else:
            context_field = np.zeros(D0, dtype=np.int32)
        context_field += self.layers[0].h.astype(np.int32)

        # Top-down field from higher layers
        if self.n_layers > 1 and self._built:
            td_field = self.compute_topdown_field(self.layers[1].state, 0)
            context_field += td_field

        # For each candidate word, compute overlap with context field
        # Energy ∝ -overlap(candidate_active_bits, context_field)
        for i, w in enumerate(candidate_words):
            w = int(w)
            if w < 0 or w >= sdr_encoder.vocab_size:
                energies[i] = scale * 10  # High energy for OOV
                continue

            candidate_active = sdr_encoder.word_active_bits[w]

            # Field alignment: sum of context_field at candidate's active positions
            alignment = int(np.sum(context_field[candidate_active]))

            # Energy = -alignment * scale / D (normalized)
            # Negative alignment → high energy (anti-aligned with context)
            # Positive alignment → low energy (aligned with context)
            energies[i] = -alignment * scale // D0

        return energies

    def train_batch_hebbian(
        self,
        context_sdrs: np.ndarray,
        target_sdrs: np.ndarray,
        eta: int = 1,
    ) -> None:
        """
        Batch Hebbian training across all layers.

        For each training pair (context, target):
          1. L0: store association context → target
          2. Coarse-grain to higher levels
          3. Store associations at each level

        The RG flow ensures consistency: the coarse-grained
        associations are derived from the fine-grained ones,
        maintaining the UV-completeness of the hierarchy.

        Args:
            context_sdrs: Context SDRs (N, D0) uint8.
            target_sdrs: Target SDRs (N, D0) uint8.
            eta: Learning rate.
        """
        N = context_sdrs.shape[0]

        # L0: direct Hebbian storage
        self.layers[0].store_batch_hebbian(context_sdrs, target_sdrs, eta)

        # Higher levels: coarse-grain and store
        prev_context = context_sdrs
        prev_target = target_sdrs

        for l in range(1, self.n_layers):
            D_coarse = self.layers_config[l][0]
            k_coarse = self.layers_config[l][1]

            # Coarse-grain context and target
            W = self.W_up[l - 1]
            if W is None:
                break

            # Block-spin projection
            coarse_ctx_acc = W.astype(np.int32) @ prev_context.astype(np.int32).T  # (D_coarse, N)
            coarse_tgt_acc = W.astype(np.int32) @ prev_target.astype(np.int32).T  # (D_coarse, N)

            # kWTA each column
            coarse_context = np.zeros((N, D_coarse), dtype=np.uint8)
            coarse_target = np.zeros((N, D_coarse), dtype=np.uint8)
            for n in range(N):
                top_k_ctx = np.argpartition(coarse_ctx_acc[:, n], -k_coarse)[-k_coarse:]
                coarse_context[n, top_k_ctx] = 1
                top_k_tgt = np.argpartition(coarse_tgt_acc[:, n], -k_coarse)[-k_coarse:]
                coarse_target[n, top_k_tgt] = 1

            # Store at this level
            self.layers[l].store_batch_hebbian(coarse_context, coarse_target, eta)

            # Update RG beta function
            # β = <J_coarse> / <J_fine> — ratio of effective couplings
            J_fine_max = max(1, int(np.max(np.abs(self.layers[l - 1].J))))
            J_coarse_max = max(1, int(np.max(np.abs(self.layers[l].J))))
            self.rg_beta[l - 1] = J_coarse_max / J_fine_max

            prev_context = coarse_context
            prev_target = coarse_target

        # UV regularization at all levels
        if self.uv_regularize:
            for layer in self.layers:
                layer._uv_regularize()

    def reset(self) -> None:
        """Reset all layer states for a new document."""
        for layer in self.layers:
            layer.reset()

    def get_diagnostics(self) -> Dict:
        """Return diagnostics for all layers."""
        diag = {'n_layers': self.n_layers, 'built': self._built}
        for l, layer in enumerate(self.layers):
            diag[f'L{l}'] = layer.get_diagnostics()
        if self._built:
            for l in range(self.n_layers - 1):
                if self.rg_beta[l] is not None:
                    diag[f'rg_beta_{l}'] = self.rg_beta[l]
        total_mem = sum(layer.J.nbytes + layer.h.nbytes for layer in self.layers)
        diag['total_memory_kb'] = total_mem / 1024
        return diag

    def _print_diagnostics(self) -> None:
        """Print hierarchy diagnostics."""
        print(f"    Hierarchical DAM: {self.n_layers} layers")
        for l, (D, k, scale) in enumerate(self.layers_config):
            sparsity = k / D * 100
            print(f"      L{l}: D={D}, k={k} ({sparsity:.1f}% sparse), scale={scale}")
        if self._built:
            for l in range(self.n_layers - 1):
                if self.W_up[l] is not None:
                    print(f"      RG flow L{l}→L{l+1}: block_size={self.layers_config[l][0]//self.layers_config[l+1][0]}, "
                          f"β={self.rg_beta[l]:.3f}")
        total_mem = sum(layer.J.nbytes + layer.h.nbytes for layer in self.layers)
        print(f"      Total memory: {total_mem/1024:.1f} KB")
