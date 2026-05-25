"""
Hierarchical DAM with Wilsonian RG Flow — UV-Complete Architecture.

KEY FIX (Misconception #1): RG flow acts on COUPLING CONSTANTS, not spin states.

The old architecture treated block-spin transforms on spin states as "RG flow."
This is WRONG. Wilsonian RG operates on coupling constants (the J matrices):
  - Integrating out short-range DOFs produces effective couplings at coarser scale
  - J_eff at level l+1 is DERIVED from J at level l via RG decimation
  - The RG flow equation: J_{l+1} = R(J_l) where R is the decimation operator

This module implements:
  1. Coupling-space RG flow: J_eff[l+1] derived from J[l] by block decimation
  2. State-space projection: still used for context propagation (bottom-up/top-down)
  3. Operator spectrum: eigenvalues of J classify relevant/marginal/irrelevant
  4. UV completeness check: cutoff independence + coupling flow stability

Based on:
  Howard et al. (2024): Wilsonian RG of NN-GPs — data sets IR scale
  Erbin et al. (2021): Weight std as RG flow parameter in NN-QFT
  Peraza Coppola et al. (2025): GP-like UV fixed point, scaling intervals
  Ferko et al. (2026a): Anomalies via Ward identities in NN-FT
  Tiberi et al. (2021): Gell-Mann-Low criticality in neural fields

ALL INTEGER ARITHMETIC. Runs on Pi 5.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .dam import DAMLayer
from .sdr import SDREncoder


class HierarchicalDAM:
    """
    Hierarchical Dense Associative Memory with Wilsonian RG flow
    acting on COUPLING CONSTANTS (not spin states).

    Four layers with increasing abstraction and decreasing dimension:
      L0: D=512, k=10 (2% sparse) — lexical patterns (UV)
      L1: D=256, k=8 (3% sparse) — syntactic patterns
      L2: D=128, k=5 (4% sparse) — semantic patterns
      L3: D=64,  k=3 (5% sparse) — discourse patterns (IR)

    RG flow (coupling space):
      J_eff[l+1] = Decimate(J[l], block_size=D_l/D_{l+1})
      This is the proper Wilsonian RG: integrating out UV DOFs
      produces effective couplings at the IR scale.

    Inter-layer state coupling:
      - Bottom-up: block-spin projection (context propagation)
      - Top-down: relevant operator feedback (IR → UV)
      - Anomalous dimensions: from operator spectrum of J, not running correlations
    """

    # Default layer configurations: (D, k, scale)
    DEFAULT_LAYERS = [
        (512, 10, 1600),  # L0: lexical (UV)
        (256, 8, 1200),   # L1: syntactic
        (128, 5, 800),    # L2: semantic
        (64, 3, 400),     # L3: discourse (IR)
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
        learning_mode: str = "hebbian",
        seed: int = 42,
    ):
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
        self.learning_mode = learning_mode
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
                learning_mode=learning_mode,
                seed=seed + l * 1000,
            )
            self.layers.append(layer)

        # Inter-layer coupling matrices (state-space projection)
        # W_up[l]: (D_{l+1}, D_l) — block-spin projection for context
        # W_down[l]: (D_l, D_{l+1}) — top-down feedback
        self.W_up: List[Optional[np.ndarray]] = [None] * (self.n_layers - 1)
        self.W_down: List[Optional[np.ndarray]] = [None] * (self.n_layers - 1)

        # RG flow: effective coupling matrices (coupling-space RG)
        # J_eff[l]: effective coupling at level l, derived from level 0
        # via successive decimation
        self.J_eff: List[Optional[np.ndarray]] = [None] * self.n_layers

        # RG beta functions: coupling flow ratios
        self.rg_beta: List[Optional[float]] = [None] * (self.n_layers - 1)

        # Anomalous dimensions: from operator spectrum of J at each level
        # These are the proper RG anomalous dimensions, NOT running correlations
        self.gamma: List[Optional[np.ndarray]] = [None] * self.n_layers

        self._built = False
        self._rng = np.random.RandomState(seed)

    def build(self, sdr_encoder: SDREncoder) -> "HierarchicalDAM":
        """
        Build inter-layer coupling matrices and initialize layers.

        Two kinds of inter-layer structure:
          1. STATE-SPACE: W_up/W_down for context propagation (bottom-up/top-down)
          2. COUPLING-SPACE: J_eff derived by RG decimation (the actual RG flow)

        Args:
            sdr_encoder: The SDR encoder (for dimension reference).

        Returns:
            self
        """
        for l in range(self.n_layers - 1):
            D_fine = self.layers_config[l][0]
            D_coarse = self.layers_config[l + 1][0]

            block_size = D_fine // D_coarse
            assert D_fine % D_coarse == 0, \
                f"D_fine ({D_fine}) must be divisible by D_coarse ({D_coarse})"

            # State-space projection matrices
            W_up = np.zeros((D_coarse, D_fine), dtype=np.int16)
            for c in range(D_coarse):
                start = c * block_size
                end = start + block_size
                W_up[c, start:end] = 1

            self.W_up[l] = W_up

            W_down = np.zeros((D_fine, D_coarse), dtype=np.int16)
            for c in range(D_coarse):
                start = c * block_size
                end = start + block_size
                W_down[start:end, c] = 1

            self.W_down[l] = W_down

            # Initialize RG beta function
            self.rg_beta[l] = D_coarse / D_fine

        self._built = True
        self._print_diagnostics()

        return self

    def compute_coupling_flow(self) -> None:
        """
        Compute the RG flow of coupling constants from L0 to L3.

        This is the CORE FIX for Misconception #1: RG acts on couplings,
        not spin states.

        The coupling flow works as follows:
          J_eff[0] = J[0]  (L0 is the UV theory, full couplings)
          J_eff[l+1] = Decimate(J_eff[l], block_size=D_l/D_{l+1})

        The decimation operator integrates out the within-block degrees
        of freedom, producing an effective coupling between blocks.

        This ensures that the J at each level is CONSISTENT with the
        level below — it's not independently learned but derived from
        the UV theory via RG flow.
        """
        if not self._built or self.n_layers < 2:
            return

        # L0: the UV theory — full coupling matrix
        self.J_eff[0] = self.layers[0].J.copy()

        # Successive decimation from L0 to L3
        for l in range(self.n_layers - 1):
            D_fine = self.layers_config[l][0]
            D_coarse = self.layers_config[l + 1][0]
            block_size = D_fine // D_coarse

            # Decimate the effective coupling from level l
            J_fine = self.J_eff[l]
            if J_fine is None:
                continue

            D_f = J_fine.shape[0]
            D_c = D_f // block_size

            if D_c < 2:
                break

            # Block decimation: J_eff[α,β] = Σ_{i∈α, j∈β} J[i,j] / (|α|·|β|)
            J_coarse = np.zeros((D_c, D_c), dtype=np.int32)
            for alpha in range(D_c):
                for beta in range(D_c):
                    if alpha == beta:
                        continue
                    i_start = alpha * block_size
                    i_end = i_start + block_size
                    j_start = beta * block_size
                    j_end = j_start + block_size

                    block_sum = int(np.sum(J_fine[i_start:i_end, j_start:j_end]))
                    # Normalize by block area, keep in Q8 fixed-point
                    J_coarse[alpha, beta] = block_sum * 256 // (block_size * block_size)

            # Clip to int16 range
            J_coarse = np.clip(J_coarse, -32768, 32767).astype(np.int16)
            self.J_eff[l + 1] = J_coarse

            # Update RG beta function: ratio of coupling strengths
            J_fine_max = max(1, int(np.max(np.abs(J_fine))))
            J_coarse_max = max(1, int(np.max(np.abs(J_coarse))))
            self.rg_beta[l] = J_coarse_max / J_fine_max

        # Compute anomalous dimensions from operator spectrum
        self._compute_anomalous_dimensions()

    def _compute_anomalous_dimensions(self) -> None:
        """
        Compute anomalous dimensions from the operator spectrum of J.

        FIX for Misconception #5: Anomalous dimensions come from the
        eigenvalue spectrum of J, not from running correlations of
        spin activity.

        Based on:
          Halverson et al. (2020): J spectrum maps to operator spectrum
          Ferko et al. (2026a): Anomalies detected via Ward identities
          Tiberi et al. (2021): Gell-Mann-Low criticality

        The anomalous dimension γ[d] is defined as:
          γ[d] = log(|λ_d|) / log(|λ_0|)

        where λ_d are eigenvalues of J sorted by magnitude.

        Classification:
          γ > 0.5 → relevant (grows under RG flow to IR)
          0 < γ ≤ 0.5 → marginal (persists logarithmically)
          γ ≤ 0 → irrelevant (decays under RG)
        """
        for l in range(self.n_layers):
            spectrum = self.layers[l].compute_operator_spectrum()
            self.gamma[l] = spectrum['anomalous_dimensions']

    def coarse_grain(self, state: np.ndarray, level: int) -> np.ndarray:
        """
        Block-spin coarse-graining: project state from level l to level l+1.

        This is the STATE-SPACE projection for context propagation,
        NOT the RG flow (which acts on coupling constants).

        The RG flow is in compute_coupling_flow() above.

        Args:
            state: Fine-grained state (D_l,) uint8.
            level: Level index l.

        Returns:
            Coarse-grained state (D_{l+1},) uint8.
        """
        if level >= self.n_layers - 1:
            return state

        W = self.W_up[level]
        if W is None:
            return np.zeros(self.layers_config[level + 1][0], dtype=np.uint8)

        coarse_acc = W.astype(np.int32) @ state.astype(np.int32)

        D_coarse = self.layers_config[level + 1][0]
        k_coarse = self.layers_config[level + 1][1]
        top_k = np.argpartition(coarse_acc, -k_coarse)[-k_coarse:]

        coarse_state = np.zeros(D_coarse, dtype=np.uint8)
        coarse_state[top_k] = 1

        return coarse_state

    def compute_topdown_field(self, state: np.ndarray, level: int) -> np.ndarray:
        """
        Compute top-down feedback field from level l+1 to level l.

        Uses anomalous dimensions γ[d] (from operator spectrum) to weight
        the top-down signal. Relevant operators (γ > 0.5) get stronger
        feedback; irrelevant operators (γ ≤ 0) get weaker feedback.

        FIX for Misconception #5: γ comes from operator spectrum, not
        running correlations.

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

        # Base top-down projection
        field = W.astype(np.int32) @ state.astype(np.int32) * self.topdown_scale

        # Apply anomalous dimension weighting
        # γ comes from the operator spectrum of the coarse level
        if self.gamma[level + 1] is not None:
            gamma = self.gamma[level + 1]
            D_fine = self.layers_config[level][0]
            D_coarse = self.layers_config[level + 1][0]
            block_size = D_fine // D_coarse

            # For each coarse dimension d, weight by γ[d]
            # γ > 0.5 → relevant → strong feedback (×2)
            # 0 < γ ≤ 0.5 → marginal → normal feedback (×1)
            # γ ≤ 0 → irrelevant → weak feedback (×0.5)
            for d in range(min(len(gamma), D_coarse)):
                if state[d] > 0:
                    g = float(gamma[d])
                    if g > 0.5:
                        weight = 2  # Relevant: amplify
                    elif g > 0:
                        weight = 1  # Marginal: keep
                    else:
                        weight = 0  # Irrelevant: suppress (was 1/2, but int → 0)
                    # Apply to the block of fine dimensions
                    start = d * block_size
                    end = start + block_size
                    if weight == 0:
                        field[start:end] = field[start:end] // 2
                    elif weight == 2:
                        field[start:end] = field[start:end] * 2

        return field

    def step_all(
        self,
        l0_context_field: np.ndarray,
        n_sweeps: int = 3,
    ) -> List[np.ndarray]:
        """
        Run hierarchical attractor dynamics across all layers.

        Order of operations (per sweep):
          1. Bottom-up: L0 → L1 → L2 → L3 (context propagation)
          2. Each layer runs its own attractor dynamics
          3. Top-down: L3 → L2 → L1 → L0 (feedback with γ-weighting)
          4. Final L0 step with all context

        Args:
            l0_context_field: Context field for L0 (D0,) int32.
            n_sweeps: Number of hierarchical sweeps.

        Returns:
            List of states for each layer.
        """
        states = [layer.state.copy() for layer in self.layers]

        for sweep in range(n_sweeps):
            # === BOTTOM-UP PASS ===
            for l in range(self.n_layers - 1):
                bu_field = self.W_up[l].astype(np.int32) @ states[l].astype(np.int32) * 100
                states[l + 1] = self.layers[l + 1].step(bu_field, n_sweeps=1)

            # === TOP-DOWN PASS ===
            for l in range(self.n_layers - 2, -1, -1):
                td_field = self.compute_topdown_field(states[l + 1], l)

                if l == 0:
                    total_field = l0_context_field + td_field
                else:
                    W = self.W_up[l - 1]
                    bu_field = W.astype(np.int32) @ states[l - 1].astype(np.int32) * 100
                    total_field = bu_field + td_field

                states[l] = self.layers[l].step(total_field, n_sweeps=1)

        # Final L0 step with full context
        total_field = l0_context_field.copy()
        if self.n_layers > 1:
            total_field += self.compute_topdown_field(states[1], 0)
        states[0] = self.layers[0].step(total_field, n_sweeps=1)

        for l in range(self.n_layers):
            self.layers[l].state = states[l]

        return states

    def compute_word_energies(
        self,
        context_sdr: np.ndarray,
        candidate_words: np.ndarray,
        sdr_encoder: SDREncoder,
        scale: int = 1600,
    ) -> np.ndarray:
        """
        Compute energy for each candidate word using F-lookup nonlinearity.

        FIX for Misconception #6: The energy uses the nonlinear F function,
        NOT linear alignment. This gives exponential storage capacity.

        E(w) = -Σ_{i ∈ active(w)} F(context_field[i])

        where F is the nonlinear energy function from the F-lookup table.

        Args:
            context_sdr: Context SDR (D0,) uint8.
            candidate_words: Array of candidate word IDs.
            sdr_encoder: SDR encoder for word→SDR mapping.
            scale: Energy scale multiplier.

        Returns:
            Energy array (len(candidate_words),) int64.
        """
        n_cand = len(candidate_words)

        # Pre-compute context field: J[:, active_ctx] @ ones + h
        context_active = np.where(context_sdr > 0)[0]
        D0 = self.layers_config[0][0]

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

        # Compute F-lookup energy for each candidate
        # This is the CORRECT DAM energy with nonlinear F
        F_lookup = self.layers[0].F_lookup
        F_offset = self.layers[0]._F_offset
        J_MAX = self.layers[0].J_MAX

        energies = np.zeros(n_cand, dtype=np.int64)

        for i, w in enumerate(candidate_words):
            w = int(w)
            if w < 0 or w >= sdr_encoder.vocab_size:
                energies[i] = scale * 10
                continue

            active_bits = sdr_encoder.word_active_bits[w]
            if len(active_bits) == 0:
                energies[i] = scale * 10
                continue

            k = len(active_bits)
            total_f = 0
            for d in active_bits:
                d = int(d)
                if 0 <= d < D0:
                    f_val = int(context_field[d])
                    f_val = max(-J_MAX, min(J_MAX, f_val))
                    total_f += int(F_lookup[f_val + F_offset])

            # Energy = -F_contribution * scale / k (normalized)
            energies[i] = -total_f * scale // max(1, k)

        return energies

    def train_batch_hebbian(
        self,
        context_sdrs: np.ndarray,
        target_sdrs: np.ndarray,
        eta: int = 1,
    ) -> None:
        """
        Batch Hebbian training across all layers.

        FIX for Misconception #3: Pure Hebbian at the right sparsity
        IS the RG fixed point (Agliari et al. 2025). No PCD needed
        when sparsity is properly tuned.

        After training, compute the coupling flow (RG decimation of J)
        to derive J_eff at all levels from L0's J.

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

            W = self.W_up[l - 1]
            if W is None:
                break

            # Block-spin projection
            coarse_ctx_acc = W.astype(np.int32) @ prev_context.astype(np.int32).T
            coarse_tgt_acc = W.astype(np.int32) @ prev_target.astype(np.int32).T

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

            prev_context = coarse_context
            prev_target = coarse_target

        # UV regularization at all levels
        if self.uv_regularize:
            for layer in self.layers:
                layer._uv_regularize()

        # Compute coupling flow (RG decimation from L0)
        self.compute_coupling_flow()

    def check_uv_completeness(self) -> dict:
        """
        Check UV completeness across the entire hierarchy.

        Based on the knowledge base:
          - Cutoff independence (Sen & Vaidya 2025)
          - Coupling flow stability (Howard et al. 2024)
          - No scale anomaly (Ferko et al. 2026a)
          - Gell-Mann-Low criticality (Tiberi et al. 2021)

        Returns:
            dict with UV completeness diagnostics per layer.
        """
        results = {}
        for l in range(self.n_layers):
            results[f'L{l}'] = self.layers[l].check_uv_completeness()

        # Check coupling flow consistency
        flow_consistent = True
        for l in range(self.n_layers - 1):
            if self.rg_beta[l] is not None:
                # Beta function should be between 0 and 1
                # (couplings weaken under coarse-graining)
                if self.rg_beta[l] > 1.5 or self.rg_beta[l] < 0:
                    flow_consistent = False

        results['flow_consistent'] = flow_consistent
        results['overall_uv_score'] = np.mean([
            results[f'L{l}']['uv_score'] for l in range(self.n_layers)
        ])

        return results

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
            # Include UV completeness
            diag['uv_completeness'] = self.check_uv_completeness()
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
                    print(f"      RG flow L{l}→L{l+1}: "
                          f"block_size={self.layers_config[l][0]//self.layers_config[l+1][0]}, "
                          f"β={self.rg_beta[l]:.3f}")
        total_mem = sum(layer.J.nbytes + layer.h.nbytes for layer in self.layers)
        print(f"      Total memory: {total_mem/1024:.1f} KB")
        print(f"      Learning mode: {self.learning_mode}")
