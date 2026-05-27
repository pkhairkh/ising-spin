"""
Hierarchical DAM with Wilsonian RG Flow — UV-Complete Architecture.

DEEP FIX (Misconception #1): RG flow acts on COUPLING CONSTANTS, not spin states.
  AND the RG-derived J_eff REPLACES the independently-learned J at higher levels.

The old architecture had two problems:
  1. Block-spin transforms on spin states were called "RG flow" — WRONG.
     Wilsonian RG operates on coupling constants (the J matrices).
  2. J at each level was independently learned via Hebbian — WRONG.
     J at higher levels should be DERIVED from J[0] via RG decimation.

This module implements the CORRECT architecture:
  1. Only L0 is trained (Hebbian, the RG fixed point)
  2. J_eff[l+1] = Decimate(J_eff[l]) for l = 0, 1, 2, ...
  3. J_eff[l] REPLACES layers[l].J — no independent learning at higher levels
  4. The hierarchy is a WILSONIAN RG TOWER: each level is the effective
     theory at a coarser scale, derived from the level below

This ensures RG CONSISTENCY: the coupling at every level is derivable from
L0 by successive decimation. This is what Wilsonian RG means.

Inter-layer state coupling still uses block-spin projection (bottom-up)
and top-down feedback (IR -> UV), but these are for CONTEXT PROPAGATION,
not for the RG flow (which is in coupling space).

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
from .three_band import ThreeBandState


class HierarchicalDAM:
    """
    Hierarchical Dense Associative Memory with Wilsonian RG flow
    acting on COUPLING CONSTANTS (not spin states).

    Four layers with increasing abstraction and decreasing dimension:
      L0: D=512, k=10 (2% sparse) — lexical patterns (UV)
      L1: D=256, k=8  (3% sparse) — syntactic patterns
      L2: D=128, k=5  (4% sparse) — semantic patterns
      L3: D=64,  k=3  (5% sparse) — discourse patterns (IR)

    RG flow (coupling space):
      J_eff[l+1] = Decimate(J[l], block_size=D_l/D_{l+1})
      This is the proper Wilsonian RG: integrating out UV DOFs
      produces effective couplings at the IR scale.

    DEEP FIX: After training L0 via Hebbian, the J at higher levels
    is REPLACED by J_eff derived from L0. This ensures RG consistency:
    every level's coupling is derivable from L0 by successive decimation.

    Inter-layer state coupling:
      - Bottom-up: block-spin projection (context propagation)
      - Top-down: relevant operator feedback (IR -> UV)
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
        j_clip: int = 500,
        uv_regularize: bool = True,
        uv_lambda: int = 5,
        topdown_scale: int = 200,
        f_type: int = 2,  # F_EXP_APPROX by default
        exp_temperature: int = 100,
        seed: int = 42,
    ):
        if layers_config is None:
            layers_config = self.DEFAULT_LAYERS

        self.layers_config = layers_config
        self.n_layers = len(layers_config)
        self.j_clip = j_clip
        self.uv_regularize = uv_regularize
        self.uv_lambda = uv_lambda
        self.topdown_scale = topdown_scale
        self.f_type = f_type
        self.exp_temperature = exp_temperature
        self.seed = seed

        # Create DAM layers
        self.layers: List[DAMLayer] = []
        for l, (D, k, scale) in enumerate(layers_config):
            layer = DAMLayer(
                D=D, k=k, scale=scale,
                j_clip=j_clip,
                uv_regularize=uv_regularize,
                uv_lambda=uv_lambda,
                f_type=f_type,
                exp_temperature=exp_temperature,
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
        # via successive decimation. After compute_coupling_flow(),
        # J_eff[l] REPLACES layers[l].J.
        self.J_eff: List[Optional[np.ndarray]] = [None] * self.n_layers

        # RG beta functions: coupling flow ratios
        self.rg_beta: List[Optional[float]] = [None] * (self.n_layers - 1)

        # Anomalous dimensions: from operator spectrum of J at each level
        self.gamma: List[Optional[np.ndarray]] = [None] * self.n_layers

        # Track whether J_eff has been applied to layers
        self._rg_applied = False

        # v62: Three-band magnetization state (spin glass internal memory)
        # Initialized with D from L0 (the only level that matters for fields)
        self.three_band: Optional[ThreeBandState] = None

        self._built = False
        self._rng = np.random.RandomState(seed)

    def build(self, sdr_encoder: SDREncoder) -> "HierarchicalDAM":
        """
        Build inter-layer coupling matrices and initialize layers.

        Two kinds of inter-layer structure:
          1. STATE-SPACE: W_up/W_down for context propagation
          2. COUPLING-SPACE: J_eff derived by RG decimation (applied after training)

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

        # v62: Initialize three-band magnetization state
        D0 = self.layers_config[0][0]
        self.three_band = ThreeBandState(D=D0)

        self._print_diagnostics()

        return self

    def train_l0_hebbian(
        self,
        context_sdrs: np.ndarray,
        target_sdrs: np.ndarray,
        eta: int = 1,
        defer_rg: bool = False,
    ) -> None:
        """
        Train ONLY L0 via batch Hebbian. Higher levels get J from RG flow.

        DEEP FIX for Misconception #1: J at higher levels is NOT independently
        learned. It is DERIVED from L0's J via Wilsonian RG decimation.

        This ensures RG CONSISTENCY: the coupling at every level is derivable
        from L0 by successive decimation. This is what Wilsonian RG means.

        After training L0:
          1. Apply UV regularization to L0
          2. Compute coupling flow: J_eff[l+1] = Decimate(J_eff[l])
          3. REPLACE layers[l].J with J_eff[l] for all l > 0
          4. Compute anomalous dimensions from the new operator spectra

        Args:
            context_sdrs: Context SDRs (N, D0) uint8.
            target_sdrs: Target SDRs (N, D0) uint8.
            eta: Learning rate.
            defer_rg: If True, skip RG flow computation (do it once at the end).
                      This avoids recomputing RG flow after every batch, which
                      is unnecessary since only L0's J changes during training.
                      Call finalize_rg_flow() once after all batches are done.
        """
        # Train L0
        self.layers[0].store_batch_hebbian(context_sdrs, target_sdrs, eta)

        # UV regularization on L0 only
        if self.uv_regularize:
            self.layers[0]._uv_regularize()

        # Compute coupling flow and apply to all higher levels
        # DEFERRED: skip during batch training, do once at the end
        if not defer_rg:
            self.compute_coupling_flow()
            self._apply_coupling_flow()

    def finalize_rg_flow(self) -> None:
        """
        Compute and apply RG flow ONCE after all Hebbian batches are done.

        This is called instead of running RG flow after every batch.
        Since only L0's J changes during training, the RG flow to higher
        levels only needs to be computed once at the end.
        """
        self.compute_coupling_flow()
        self._apply_coupling_flow()

    def _apply_coupling_flow(self) -> None:
        """
        Apply the RG-derived J_eff to all layers.

        This is the KEY step: replace independently-learned J at higher
        levels with the RG-derived effective coupling from L0.

        After this:
          - layers[0].J remains the Hebbian-trained L0 coupling (UV theory)
          - layers[l].J = J_eff[l] for l > 0 (IR theory derived from UV)
          - The hierarchy is now a PROPER Wilsonian RG tower
        """
        for l in range(self.n_layers):
            if self.J_eff[l] is not None:
                self.layers[l].apply_coupling_flow(self.J_eff[l])

        self._rg_applied = True

    def compute_coupling_flow(self) -> None:
        """
        Compute the RG flow of coupling constants from L0 to L(n-1).

        v34 CRITICAL FIX: Previous versions had TWO bugs:
          1. Used self.layers[l].J (which is ZERO for l>0 since _apply_coupling_flow
             hasn't been called yet) instead of J_eff[l] for decimation.
             This is why L2-L4 were always zero.
          2. Divided by block_size² (mean coupling), which dilutes couplings
             by 4x per level. After 4 levels: 256x dilution = dead layers.
             Changed to sum/block_size (Kadanoff block-spin RG: each block-spin
             represents block_size spins, so coupling per block-spin pair =
             total / block_size, not total / block_size²).
          3. Added RG rescaling: after decimation, normalize J_eff to j_clip
             so that attractor dynamics remain strong at all levels. This is
             the standard Wilsonian rescaling step (relevant operators must
             be rescaled to stay in the perturbative regime).

        The coupling flow works as follows:
          J_eff[0] = J[0]  (L0 is the UV theory, full couplings)
          J_eff[l+1] = Rescale(Decimate(J_eff[l], block_size))
        """
        if not self._built or self.n_layers < 2:
            return

        # L0: the UV theory — full coupling matrix
        self.J_eff[0] = self.layers[0].J.copy()

        # Successive decimation from L0 to L(n-1)
        for l in range(self.n_layers - 1):
            D_fine = self.layers_config[l][0]
            D_coarse = self.layers_config[l + 1][0]
            block_size = D_fine // D_coarse

            # Decimate the effective coupling from level l
            J_fine = self.J_eff[l]
            if J_fine is None:
                continue

            # v34 FIX: Decimate J_eff[l] directly, NOT layers[l].J.
            # layers[l].J is zero for l>0 because _apply_coupling_flow()
            # hasn't been called yet. This was the root cause of dead L2-L4.
            J_coarse = self._decimate_J(J_fine, block_size)

            if J_coarse is None:
                break

            # v34 FIX: RG rescaling — normalize J to j_clip so attractor
            # dynamics remain strong at all levels. This is the standard
            # Wilsonian rescaling step: relevant operators must be rescaled
            # to stay in the perturbative regime.
            J_coarse_max = int(np.max(np.abs(J_coarse)))
            if J_coarse_max > 0:
                # Preserve coupling pattern but scale to j_clip range.
                # This is like multiplying by the RG rescaling factor.
                scale_factor = self.j_clip / J_coarse_max
                J_coarse = (J_coarse.astype(np.float64) * scale_factor)
                J_coarse = np.clip(J_coarse, -32768, 32767)
                J_coarse = np.round(J_coarse).astype(np.int16)

            self.J_eff[l + 1] = J_coarse

            # Update RG beta function: ratio of coupling strengths
            J_fine_max = max(1, int(np.max(np.abs(J_fine))))
            J_coarse_max_after = max(1, int(np.max(np.abs(J_coarse))))
            self.rg_beta[l] = J_coarse_max_after / J_fine_max

        # Compute anomalous dimensions from operator spectrum
        self._compute_anomalous_dimensions()

    @staticmethod
    def _decimate_J(J_fine: np.ndarray, block_size: int) -> Optional[np.ndarray]:
        """
        Decimate a coupling matrix via Kadanoff block-spin RG.

        v34: Fixed two bugs from the old DAMLayer.compute_coupling_flow():
          1. Now operates on the GIVEN matrix (J_eff[l]), not layers[l].J
          2. Uses sum/block_size instead of sum/block_size² (mean)

        The Kadanoff block-spin RG prescription:
          - Group fine-grained spins into blocks of size block_size
          - Define block-spin S_a = (1/block_size) * sum_{i in a} s_i
          - The effective coupling between block-spins is:
            J_eff[a,b] = sum_{i in a, j in b} J[i,j] / block_size

        Why / block_size (not / block_size²)?
          Each block-spin S_a represents block_size fine-grained spins.
          The coupling per block-spin pair is the TOTAL coupling between
          blocks, divided by one factor of block_size (for the spin
          normalization). With /block_size² (mean), the coupling dilutes
          by 4x per level — after 4 levels: 256x dilution = dead layers.

        Args:
            J_fine: Fine-grained coupling matrix (D, D) int16.
            block_size: Number of fine-grained spins per block.

        Returns:
            Decimated coupling matrix (D//block_size, D//block_size) int16,
            or None if D_coarse < 2.
        """
        D = J_fine.shape[0]
        D_coarse = D // block_size

        if D_coarse < 2:
            return None

        J_eff = np.zeros((D_coarse, D_coarse), dtype=np.int32)

        for alpha in range(D_coarse):
            for beta_idx in range(D_coarse):
                if alpha == beta_idx:
                    continue
                i_start = alpha * block_size
                i_end = i_start + block_size
                j_start = beta_idx * block_size
                j_end = j_start + block_size

                # Sum of all couplings between blocks
                block_sum = int(np.sum(J_fine[i_start:i_end, j_start:j_end]))
                # Kadanoff: divide by block_size (NOT block_size²)
                J_eff[alpha, beta_idx] = block_sum // block_size

        # Clip to int16 range
        J_eff = np.clip(J_eff, -32768, 32767).astype(np.int16)

        return J_eff

    def _compute_anomalous_dimensions(self) -> None:
        """
        Compute anomalous dimensions from the operator spectrum of J.

        DEEP FIX for Misconception #5: Anomalous dimensions come from the
        eigenvalue spectrum of J, not from running correlations of spin activity.

        Based on:
          Halverson et al. (2020): J spectrum maps to operator spectrum
          Ferko et al. (2026a): Anomalies detected via Ward identities
          Tiberi et al. (2021): Gell-Mann-Low criticality

        The anomalous dimension gamma[d] is defined as:
          gamma[d] = log(|lambda_d|) / log(|lambda_0|)

        where lambda_d are eigenvalues of J sorted by magnitude.

        Classification:
          gamma > 0.5  -> relevant (grows under RG flow to IR)
          0 < gamma <= 0.5 -> marginal (persists logarithmically)
          gamma <= 0   -> irrelevant (decays under RG)
        """
        for l in range(self.n_layers):
            spectrum = self.layers[l].compute_operator_spectrum(force_recompute=True)
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

        Uses anomalous dimensions gamma[d] (from operator spectrum) to weight
        the top-down signal. Relevant operators (gamma > 0.5) get stronger
        feedback; irrelevant operators (gamma <= 0) get weaker feedback.

        DEEP FIX for Misconception #5: gamma comes from operator spectrum, not
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
        if self.gamma[level + 1] is not None:
            gamma = self.gamma[level + 1]
            D_fine = self.layers_config[level][0]
            D_coarse = self.layers_config[level + 1][0]
            block_size = D_fine // D_coarse

            for d in range(min(len(gamma), D_coarse)):
                if state[d] > 0:
                    g = float(gamma[d])
                    if g > 0.5:
                        weight = 2  # Relevant: amplify
                    elif g > 0:
                        weight = 1  # Marginal: keep
                    else:
                        weight = 0  # Irrelevant: suppress
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
          1. Bottom-up: L0 -> L1 -> L2 -> L3 (context propagation)
          2. Each layer runs its own attractor dynamics
          3. Top-down: L3 -> L2 -> L1 -> L0 (feedback with gamma-weighting)
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

        # v62: Update three-band magnetization with the new L0 state.
        # This is the KEY step that provides persistent memory:
        # after the DAM settles into its attractor, we record which
        # spins are active into the EMA at three time scales.
        if self.three_band is not None:
            self.three_band.update(states[0])

        return states

    # v39: Energy normalization constant.
    # LOG2_NORM=512 — 8x more energy resolution than v37/v38's 4096.
    #
    # v37/v38 REGRESSION ROOT CAUSE: LOG2_NORM=4096 with integer division
    # collapses distinct energy levels. With typical total_f 20000-40000:
    #   LOG2_NORM=4096: energy range -4 to -9 → only ~5 distinct levels!
    #   LOG2_NORM=512:  energy range -39 to -78 → ~39 distinct levels.
    # The Boltzmann sampler CANNOT discriminate with only 5 levels.
    # This was the root cause of v35 PPL=461 → v37 PPL=5587.
    #
    # v35 used LOG2_NORM=512 (with k division = effective 5120) and got PPL=461.
    # v39 returns to LOG2_NORM=512 WITHOUT k division, giving effective 512
    # (10x MORE resolution than v35) with dE ~ O(200-300) and beta ~ 0.01.
    LOG2_NORM = 512

    # v37: h field REMOVED from word energy computation.
    # The h field (Hebbian bias) encodes word frequency. It creates
    # unconditional bias that makes common words always win regardless
    # of context. This was the primary driver of repetitive output.
    #
    # v36 tried scaling h to 5%, but even 5% was too much — it reinforced
    # the episodic frequency bias. v37 removes h entirely from word
    # selection energy.
    #
    # For attractor dynamics (step_all, kWTA), h is used at full strength.
    # This scaling only applies to word selection energy.
    H_SCALE_NUM = 0
    H_SCALE_DEN = 1  # Unused when H_SCALE_NUM=0

    # v67: Three-band spin state weights — TAU-NORMALIZED with correct scale.
    #
    # DESIGN GOAL: Spin energy should be ~10-20% of DAM energy range.
    # DAM energy range: ~40 distinct levels ([-80, -40] typically).
    # Target spin energy range: ~8-16 distinct levels → meaningful
    # discrimination without overwhelming the n-gram/DAM signals.
    #
    # The magnetization vectors m_z/m_x/m_y accumulate over many steps
    # (up to tau=50 for Z band). After τ-normalization (dividing by τ),
    # fields represent "average activity" (0 to 1) projected through J.
    # This puts them on the same scale as the context field — exactly like
    # a transformer's KV cache contribution.
    #
    # v67 CHANGES from v66:
    #   - Z weight increased from 2 to 5 (topic is the primary coherence
    #     signal — should be dominant)
    #   - X weight reduced from 1 to 1 (narrative is fast, already strong
    #     due to low τ=5; doesn't need boost)
    #   - Y weight increased from 1 to 3 (v67 Y band uses min(s, m_z)*4
    #     instead of AND, so it now has meaningful signal)
    #   - All denominators are 1 (simplifies the deferred division)
    #
    # These weights produce spin energy ~[-20, +20] range after the
    # deferred division, giving ~40 distinct levels — comparable to DAM.
    Z_WEIGHT_NUM = 5   # 5x for topic field (AFTER tau-normalization)
    Z_WEIGHT_DEN = 1
    X_WEIGHT_NUM = 1   # 1x for narrative field (AFTER tau-normalization)
    X_WEIGHT_DEN = 1
    Y_WEIGHT_NUM = 3   # 3x for syntactic field (AFTER tau-normalization)
    Y_WEIGHT_DEN = 1

    def compute_word_energies(
        self,
        context_sdr: np.ndarray,
        candidate_words: np.ndarray,
        sdr_encoder: SDREncoder,
        scale: int = 1600,
    ) -> np.ndarray:
        """
        Compute energy for each candidate word using NORMALIZED log2-F.

        v39 FIX: LOG2_NORM reduced from 4096 to 512.
          v37/v38 used LOG2_NORM=4096, which with integer division gave only
          ~5 distinct energy levels for the typical total_f range. This made
          the Boltzmann sampler unable to discriminate between candidates,
          causing PPL=5587 (v37) vs v35's PPL=461 with LOG2_NORM=512.

          With LOG2_NORM=512 (no k division), we get ~39 distinct levels —
          8x more discriminative power. dE ~ O(200-300), beta ~ 0.01.

        E(w) = -sum_{d in active(w)} log2_F(context_field[d]) / LOG2_NORM

        where log2_F(x) = log2(T + x%T) + (x//T) in 256x fixed-point,
        context_field = J[:, ctx_active] @ ctx_ones + topdown (NO h).

        Args:
            context_sdr: Context SDR (D0,) uint8.
            candidate_words: Array of candidate word IDs.
            sdr_encoder: SDR encoder for word->SDR mapping.
            scale: IGNORED in log2-space (kept for API compatibility).

        Returns:
            Energy array (len(candidate_words),) int64. Lower = more likely.
            Values are normalized so dE ~ O(200-300), suitable for beta ~ 0.01.
        """
        n_cand = len(candidate_words)

        # Pre-compute context field: J[:, active_ctx] @ ones (NO h in v37)
        context_active = np.where(context_sdr > 0)[0]
        D0 = self.layers_config[0][0]

        if len(context_active) > 0:
            context_field = (
                self.layers[0].J[:, context_active].astype(np.int32) @
                context_sdr[context_active].astype(np.int32)
            )
        else:
            context_field = np.zeros(D0, dtype=np.int32)
        # v37: h field REMOVED from word energy.
        # h encodes word frequency → degenerate frequency dominance.
        # Even at 5% (v36), it reinforced episodic frequency bias.
        # Word selection uses ONLY context-specific coupling signal.
        # Attractor dynamics (kWTA in step_all) still use full h.
        if self.H_SCALE_NUM > 0:
            context_field += (self.layers[0].h.astype(np.int32) * self.H_SCALE_NUM) // self.H_SCALE_DEN

        # v65: Three-band spin fields REMOVED from context_field.
        #
        # FUNDAMENTAL FIX: Spin fields must NOT go through log2_F().
        # The log2_F() function is an exponential-like nonlinearity.
        # Adding spin fields to context_field BEFORE log2_F() causes
        # nonlinear amplification that scrambles the DAM energy landscape.
        #
        # v63 added spin fields before log2_F → PPL 1066 (was 14.65)
        # v64 added tau-normalization → still PPL 144 (should be ~15)
        # The tau-normalization helped but wasn't enough because log2_F
        # amplifies even small additions nonlinearly.
        #
        # v65 FIX: Spin fields are now computed as LINEAR energy corrections
        # AFTER log2_F, in the new compute_spin_word_energies() method.
        # This is exactly how bigram/binding/episodic energies work — they
        # are added as perturbations to the DAM energy, not fed through the
        # exponential function.
        #
        # The transformer analogy: the KV cache output is added to the
        # residual stream, which then goes through layer norm + FFN.
        # It is NOT added before the softmax.

        # Top-down field from higher layers
        if self.n_layers > 1 and self._built:
            td_field = self.compute_topdown_field(self.layers[1].state, 0)
            context_field += td_field

        # Compute log2_F for ALL field values at once (no J_MAX clip!)
        # v37: With h removed and no k division, dE ~ O(20-40) for beta ~ 0.1-0.3
        log2_F_all = self.layers[0]._log2_piecewise_F(context_field.astype(np.int64))

        energies = np.zeros(n_cand, dtype=np.int64)

        # v57: VECTORIZED — batch lookup instead of Python loop
        # Stack all candidate active bits into a (n_cand, max_k) array
        max_k = sdr_encoder.k
        # Pre-build stacked index array
        stacked_idx = np.zeros((n_cand, max_k), dtype=np.intp)
        valid_mask = np.ones((n_cand, max_k), dtype=bool)
        n_valid_bits = np.zeros(n_cand, dtype=np.int32)

        for i, w in enumerate(candidate_words):
            w = int(w)
            if w < 0 or w >= sdr_encoder.vocab_size:
                valid_mask[i, :] = False
                continue
            active_bits = sdr_encoder.word_active_bits[w]
            k = len(active_bits)
            if k == 0:
                valid_mask[i, :] = False
                continue
            n_valid_bits[i] = k
            active_idx = np.asarray(active_bits, dtype=np.intp)
            stacked_idx[i, :k] = active_idx
            if k < max_k:
                valid_mask[i, k:] = False
            # Check bounds
            oob = (active_idx < 0) | (active_idx >= D0)
            if np.any(oob):
                valid_mask[i, :k][oob] = False

        # Batch lookup: log2_F_all[stacked_idx] → (n_cand, max_k)
        all_f = log2_F_all[stacked_idx]
        # Zero out invalid positions
        all_f[~valid_mask] = 0
        # Sum over k dimension
        total_f_arr = np.sum(all_f, axis=1)
        # Normalize
        energies = -(total_f_arr.astype(np.int64) // self.LOG2_NORM)
        # OOV / empty candidates get default high energy
        empty = ~np.any(valid_mask, axis=1)
        energies[empty] = 200

        return energies

    def compute_spin_word_energies(
        self,
        candidate_words: np.ndarray,
        sdr_encoder: SDREncoder,
        weight_num: int = 1,
        weight_den: int = 1,
    ) -> np.ndarray:
        """
        v67: Compute LINEAR spin-field energy corrections for each candidate word.

        v66 CRITICAL FIX: The v65 code divided by τ and LOG2_NORM and weight_den
        in separate integer division steps, causing cascading truncation to zero.
        With field values ~[-40,40] after τ division, k=10 active bits gives
        overlap ~[-400,400], then // 512 = [-1,1], then // 3 = 0. ALL spin
        energy was being truncated to zero by premature integer division!

        v66 fix: Accumulate the NUMERATOR first (J@m * weight_num), then do
        ONE division at the very end by (τ * LOG2_NORM * weight_den). This
        preserves precision by deferring truncation as long as possible.

        v67 FIX: Include cross-band WEIGHT_DEN factors in the deferred division.
        Also updated Z/X/Y weight multipliers for stronger spin signal.

        E_spin(w) = -(weight_num/weight_den) * overlap(spin_field, sdr(w)) / LOG2_NORM

        where:
          spin_field = (Z_num/Z_den) * (J @ m_z) / τ_z
                     + (X_num/X_den) * (J @ m_x) / τ_x
                     + (Y_num/Y_den) * (J @ m_y) / τ_y
          overlap(field, sdr) = Σ_{d in active(w)} field[d]

        Computation order (avoids truncation):
          1. Compute raw fields: J @ m_z, J @ m_x, J @ m_y
          2. Accumulate weighted numerator with cross-band DEN factors:
             num = Z_num * z_raw * τ_x * τ_y * X_den * Y_den
                 + X_num * x_raw * τ_z * τ_y * Z_den * Y_den
                 + Y_num * y_raw * τ_z * τ_x * Z_den * X_den
          3. Compute overlap with SDR (sum over k=10 active bits)
          4. ONE division: -overlap * weight_num / (τ_z*τ_x*τ_y * Z_den*X_den*Y_den * LOG2_NORM * weight_den)

        Args:
            candidate_words: Array of candidate word IDs.
            sdr_encoder: SDR encoder for word->SDR mapping.
            weight_num: Weight numerator (default 1). Scale spin energy.
            weight_den: Weight denominator (default 1). Scale spin energy.

        Returns:
            Energy array (len(candidate_words),) int64. Lower = more likely.
            Zero if three_band has no accumulated state.
        """
        n_cand = len(candidate_words)

        if self.three_band is None or self.three_band._step_count == 0:
            return np.zeros(n_cand, dtype=np.int64)

        J = self.layers[0].J
        D0 = self.layers_config[0][0]
        tb = self.three_band

        # Compute raw fields WITHOUT dividing by τ yet.
        z_field_raw = tb.compute_z_field(J)   # (D,) int32
        x_field_raw = tb.compute_x_field(J)   # (D,) int32
        y_field_raw = tb.compute_y_field(J)   # (D,) int32

        # Accumulate weighted numerator BEFORE division.
        # Each band contributes: NUM/DEL * J@m / τ
        # To put all on common denominator (τ_z * τ_x * τ_y * Z_den * X_den * Y_den):
        #   Z band term: Z_NUM * z_field * τ_x * τ_y * X_den * Y_den
        #   X band term: X_NUM * x_field * τ_z * τ_y * Z_den * Y_den
        #   Y band term: Y_NUM * y_field * τ_z * τ_x * Z_den * X_den
        tz, tx, ty = tb.TAU_Z, tb.TAU_X, tb.TAU_Y
        z_den, x_den, y_den = self.Z_WEIGHT_DEN, self.X_WEIGHT_DEN, self.Y_WEIGHT_DEN

        # Cross-denominator products (for common denominator)
        x_den_y_den = x_den * y_den
        z_den_y_den = z_den * y_den
        z_den_x_den = z_den * x_den

        spin_field_num = (
            z_field_raw.astype(np.int64) * self.Z_WEIGHT_NUM * (tx * ty) * x_den_y_den +
            x_field_raw.astype(np.int64) * self.X_WEIGHT_NUM * (tz * ty) * z_den_y_den +
            y_field_raw.astype(np.int64) * self.Y_WEIGHT_NUM * (tz * tx) * z_den_x_den
        )

        # For each candidate word, compute overlap(spin_field_num, sdr(w))
        energies = np.zeros(n_cand, dtype=np.int64)
        max_k = sdr_encoder.k
        stacked_idx = np.zeros((n_cand, max_k), dtype=np.intp)
        valid_mask = np.ones((n_cand, max_k), dtype=bool)

        for i, w in enumerate(candidate_words):
            w = int(w)
            if w < 0 or w >= sdr_encoder.vocab_size:
                valid_mask[i, :] = False
                continue
            active_bits = sdr_encoder.word_active_bits[w]
            k = len(active_bits)
            if k == 0:
                valid_mask[i, :] = False
                continue
            active_idx = np.asarray(active_bits, dtype=np.intp)
            stacked_idx[i, :k] = active_idx
            if k < max_k:
                valid_mask[i, k:] = False
            oob = (active_idx < 0) | (active_idx >= D0)
            if np.any(oob):
                valid_mask[i, :k][oob] = False

        # Batch lookup: spin_field_num[stacked_idx] → (n_cand, max_k)
        all_spin = spin_field_num[stacked_idx]
        all_spin[~valid_mask] = 0
        total_spin_num = np.sum(all_spin, axis=1).astype(np.int64)

        # ONE division at the end, preserving maximum precision.
        # Full denominator = common_tau * Z_den * X_den * Y_den * LOG2_NORM * weight_den
        # Negative because high overlap → more likely → lower energy
        full_denom = tz * tx * ty * z_den * x_den * y_den * self.LOG2_NORM * weight_den
        if full_denom > 0 and weight_num > 0:
            energies = -(total_spin_num * weight_num) // full_denom
        else:
            energies = -(total_spin_num) // (tz * tx * ty * self.LOG2_NORM)

        # OOV / empty candidates get zero spin energy (no bonus, no penalty)
        empty = ~np.any(valid_mask, axis=1)
        energies[empty] = 0

        return energies

    def train_batch_hebbian(
        self,
        context_sdrs: np.ndarray,
        target_sdrs: np.ndarray,
        eta: int = 1,
    ) -> None:
        """
        Batch Hebbian training: L0 only, then RG flow to all levels.

        DEEP FIX for Misconception #1 + #3:
          - Only L0 is trained via Hebbian (Misconception #3: pure Hebbian
            IS the RG fixed point at right sparsity)
          - Higher levels get J from RG decimation of L0 (Misconception #1:
            RG acts on couplings, not states)

        Args:
            context_sdrs: Context SDRs (N, D0) uint8.
            target_sdrs: Target SDRs (N, D0) uint8.
            eta: Learning rate.
        """
        # Train L0 via batch Hebbian
        self.layers[0].store_batch_hebbian(context_sdrs, target_sdrs, eta)

        # UV regularization on L0
        if self.uv_regularize:
            self.layers[0]._uv_regularize()

        # Compute coupling flow and apply to all higher levels
        self.compute_coupling_flow()
        self._apply_coupling_flow()

    def check_uv_completeness(self) -> dict:
        """
        Check UV completeness across the entire hierarchy.

        DEEP FIX for Misconception #2: UV completeness requires Ward identities
        and cutoff independence, NOT spectral gap monitoring.

        Based on the knowledge base:
          - Cutoff independence (Sen & Vaidya 2025)
          - Ward identities (Ferko et al. 2026a)
          - Coupling flow stability (Howard et al. 2024)
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
                if self.rg_beta[l] > 1.5 or self.rg_beta[l] < 0:
                    flow_consistent = False

        results['flow_consistent'] = flow_consistent
        results['rg_applied'] = self._rg_applied
        results['overall_uv_score'] = np.mean([
            results[f'L{l}']['uv_score'] for l in range(self.n_layers)
        ])

        return results

    def reset(self) -> None:
        """Reset all layer states for a new document.

        v62: Also resets the three-band magnetization state.
        """
        for layer in self.layers:
            layer.reset()
        if self.three_band is not None:
            self.three_band.full_reset()

    def soft_reset(self, decay: int = 2) -> None:
        """v63: Sentence boundary reset — Pauli matrix spin state reset.

        At sentence boundaries, the discourse undergoes a "measurement":
          - Z band (topic/magnetization): Soft decay (3/4), topic persists
          - X band (transitions/narrative): Hard reset, new sentence = new direction
          - Y band (coherence/syntax): Medium decay (1/2), some style persists
          - DAM layer states: Soft decay + re-sparsify

        This implements the PHYSICAL behavior of spin measurement:
        σ_z is measured softly (topic mostly preserved),
        σ_x is measured sharply (transitions collapse),
        σ_y is measured moderately (style partially preserved).

        Args:
            decay: Number of bits to right-shift the DAM state. Default 2.
        """
        # Three-band spin state reset (Pauli matrix measurement)
        if self.three_band is not None:
            self.three_band.sentence_reset()

        # DAM layer state: soft decay + re-sparsify
        for layer in self.layers:
            layer.state = (layer.state >> decay).astype(np.uint8)
            # Re-sparsify: if fewer than k bits are active, top up from field
            n_active = int(np.sum(layer.state > 0))
            if n_active < layer.k:
                # Run one kWTA step from current field to restore sparsity
                field = layer.compute_field(layer.state)
                top_k = np.argpartition(field, -layer.k)[-layer.k:]
                new_state = np.zeros(layer.D, dtype=np.uint8)
                new_state[top_k] = 1
                layer.state = new_state

    def get_diagnostics(self) -> Dict:
        """Return diagnostics for all layers."""
        diag = {
            'n_layers': self.n_layers,
            'built': self._built,
            'rg_applied': self._rg_applied,
        }
        for l, layer in enumerate(self.layers):
            diag[f'L{l}'] = layer.get_diagnostics()
        if self._built:
            for l in range(self.n_layers - 1):
                if self.rg_beta[l] is not None:
                    diag[f'rg_beta_{l}'] = self.rg_beta[l]
            # Include UV completeness (with Ward identity checks)
            diag['uv_completeness'] = self.check_uv_completeness()
        total_mem = sum(layer.J.nbytes + layer.h.nbytes for layer in self.layers)
        diag['total_memory_kb'] = total_mem / 1024
        return diag

    def _print_diagnostics(self) -> None:
        """Print hierarchy diagnostics."""
        f_type_name = {0: 'quadratic', 1: 'cubic', 2: 'exp_approx'}.get(
            self.f_type, 'unknown'
        )
        print(f"    Hierarchical DAM: {self.n_layers} layers")
        print(f"    F function: {f_type_name}, T={self.exp_temperature/100:.2f}")
        for l, (D, k, scale) in enumerate(self.layers_config):
            sparsity = k / D * 100
            print(f"      L{l}: D={D}, k={k} ({sparsity:.1f}% sparse), scale={scale}")
        if self._built:
            for l in range(self.n_layers - 1):
                if self.W_up[l] is not None:
                    print(f"      RG flow L{l}->L{l+1}: "
                          f"block_size={self.layers_config[l][0]//self.layers_config[l+1][0]}, "
                          f"beta={self.rg_beta[l]:.3f}")
        total_mem = sum(layer.J.nbytes + layer.h.nbytes for layer in self.layers)
        print(f"      Total memory: {total_mem/1024:.1f} KB")
        print(f"      Learning: Hebbian (L0 only, RG flow to higher levels)")
