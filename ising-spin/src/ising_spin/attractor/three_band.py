"""
Three-Band Spin Hidden State — Pauli Matrix Decomposition.

The DAM with exponential F has the same representational capacity as
softmax attention (Ramsauer et al. 2020). The problem wasn't capacity —
it was that the state was disconnected from the output computation.

This module provides the MISSING HIDDEN STATE, analogous to the KV cache
in a transformer. It decomposes the DAM spin state using the three Pauli
matrices — the generators of SU(2) — to extract three orthogonal views:

  BAND Z (Magnetization — σ_z = diag(+1,-1)):
    m_z = EMA(s, τ_z=50)
    Which spins are PERSISTENTLY ACTIVE — the TOPIC.
    "We're telling a fairy tale about a girl."
    Slow evolution, persists across sentences.

  BAND X (Transitions — σ_x = off-diagonal):
    m_x = EMA(s XOR s_prev, τ_x=5)
    Which spins are ACTIVELY CHANGING — the NARRATIVE DIRECTION.
    "The story is moving from introduction to conflict."
    The XOR is the discrete analogue of <0|σ_x|1> = 1 — the
    off-diagonal matrix element that measures coherence between
    the |0> and |1> states. In language: which dimensions are
    in transition between active and inactive.

  BAND Y (Coherence — σ_y = i*off-diagonal):
    v67 FIX: Changed from AND to WEIGHTED INTERSECTION.
    Old: m_y = EMA(s AND s_prev, τ_y=15) — dead with sparse SDRs!
      With k=10/D=512, expected AND overlap = k²/D ≈ 0.2 bits →
      m_y was essentially always zero.
    New: m_y = EMA(min(s, m_z) * Y_SIGNAL_SCALE, τ_y=15)
      Uses m_z (topic magnetization) as "recently active" signal,
      weighted by current state. This captures "which topic dimensions
      are active RIGHT NOW" — the syntactic/structural mode.
      Effect: m_y accumulates Z band signal filtered through current state,
      giving a medium-timescale "active topic" representation.

FIELD COMPUTATION (no new parameters!):
  field_z = J @ m_z  (topic field — what to talk about)
  field_x = J @ m_x  (narrative field — where to go next)
  field_y = J @ m_y  (syntactic field — how to structure)

  These three fields use the SAME J matrix but project through
  DIFFERENT magnetization patterns, extracting different physics.

CROSS-BAND PRECESSION (Heisenberg dynamics):
  In quantum mechanics, the spin precesses: dσ/dt = i[H, σ].
  For our Ising Hamiltonian H = -Σ J_ij σ_i σ_j:

    Z feeds X: topic constrains which transitions are likely
    X feeds Y: transitions shape the coherence pattern
    Y feeds Z: coherence reinforces the topic

  This is implemented as integer shift-and-add coupling that
  approximately conserves the total "spin angular momentum."

SENTENCE BOUNDARY BEHAVIOR:
  Z band (topic):  Soft decay (3/4) — topic persists across sentences
  X band (transitions): Hard reset — transitions don't cross sentences
  Y band (coherence):  Medium decay (1/2) — some style persists

ALL INTEGER ARITHMETIC. Runs on Pi 5.
"""

import numpy as np
from typing import Optional


class ThreeBandState:
    """
    Three-band spin hidden state from Pauli matrix decomposition.

    Maintains three EMA vectors derived from the DAM spin state
    using the three Pauli matrix measurements. Each band captures
    a qualitatively different aspect of the discourse state:

      Z (magnetization): What is the text ABOUT?
      X (transitions):   WHERE is the narrative going?
      Y (coherence):     HOW is the text structured?

    Uses the SAME J coupling matrix to compute fields — no new
    learnable parameters required.

    v67 FIX: Y band changed from AND to m_z-weighted current state.
    AND was dead (0.2 bits overlap with k=10/D=512 SDRs).
    New Y uses min(s, m_z) which captures "which topic dimensions
    are currently active" — a much stronger signal.
    """

    # Time constants (in generation steps)
    TAU_Z = 50    # Slow band: topic memory (~50 words / ~2-3 sentences)
    TAU_X = 5     # Fast band: narrative transitions (~5 words)
    TAU_Y = 15    # Medium band: syntactic coherence (~15 words / ~1 sentence)

    # v67: Y band signal scale.
    # With k=10/D=512, min(s, m_z) has ~10 active bits with values ~1-5.
    # Multiply by 4 to give Y band comparable signal strength to X band
    # after τ-normalization (Y has τ=15, X has τ=5, so Y needs 3x more
    # signal to compensate for 3x more decay per step).
    Y_SIGNAL_SCALE = 4

    # Cross-band precession coupling strengths (shift amounts)
    # These implement the Heisenberg equation dσ/dt = i[H, σ]
    # in integer arithmetic as shift-and-add.
    # Z→X coupling: topic constrains narrative direction
    ZX_SHIFT = 4   # m_z >> 4 = 1/16 of Z feeds into X
    # X→Y coupling: transitions shape coherence
    XY_SHIFT = 4   # m_x >> 4 = 1/16 of X feeds into Y
    # Y→Z coupling: coherence reinforces topic
    YZ_SHIFT = 5   # m_y >> 5 = 1/32 of Y feeds back to Z

    def __init__(self, D: int = 512):
        """
        Args:
            D: Dimension of the DAM state space.
        """
        self.D = D

        # Magnetization vectors (int32, range ~[0, τ*2])
        # These are EXPONENTIAL MOVING AVERAGES of different
        # Pauli matrix measurements of the spin state.

        # Z band: EMA of state itself — magnetization (σ_z measurement)
        # m_z[i] = how persistently spin i has been active
        self.m_z = np.zeros(D, dtype=np.int32)

        # X band: EMA of state XOR previous state — transitions (σ_x measurement)
        # m_x[i] = how frequently spin i has been TRANSITIONING
        self.m_x = np.zeros(D, dtype=np.int32)

        # Y band: EMA of min(s, m_z) * Y_SIGNAL_SCALE — active topic (σ_y measurement)
        # v67: Replaced AND with m_z-weighted current state.
        # m_y[i] = how much the current active topic overlaps with current state
        self.m_y = np.zeros(D, dtype=np.int32)

        # Previous state for XOR computation
        self._prev_state = np.zeros(D, dtype=np.uint8)

        # Precomputed thresholds for diagnostics
        self._z_threshold = max(1, self.TAU_Z // 4)
        self._x_threshold = max(1, self.TAU_X // 4)
        self._y_threshold = max(1, self.TAU_Y // 4)

        self._step_count = 0

    def update(self, state: np.ndarray) -> None:
        """
        Update all three spin bands with the new DAM state.

        The update implements the three Pauli matrix measurements:
          Z: m_z = EMA(state, τ_z)                    — magnetization
          X: m_x = EMA(state XOR prev, τ_x)           — transitions
          Y: m_y = EMA(min(s, m_z) * scale, τ_y)     — active topic (v67)

        Then applies cross-band precession (Heisenberg dynamics):
          m_x += m_z >> ZX_SHIFT  (topic informs transitions)
          m_y += m_x >> XY_SHIFT  (transitions shape coherence)
          m_z += m_y >> YZ_SHIFT  (coherence reinforces topic)

        Incremental EMA: m[i] = m[i] - m[i]//τ + signal[i]
        This is equivalent to: m[i] = m[i]*(τ-1)/τ + signal[i]

        Args:
            state: Binary DAM state (D,) uint8, with k active bits.
        """
        s = state.astype(np.int32)
        s_prev = self._prev_state.astype(np.int32)

        # --- σ_z measurement: magnetization (persistent activity) ---
        decay_z = self.m_z // self.TAU_Z
        self.m_z = self.m_z - decay_z + s

        # --- σ_x measurement: transitions (off-diagonal, XOR) ---
        # XOR captures which bits CHANGED — the discrete analogue
        # of the off-diagonal matrix element <0|σ_x|1> = 1
        transitions = s ^ s_prev  # XOR: 1 where bits differ
        decay_x = self.m_x // self.TAU_X
        self.m_x = self.m_x - decay_x + transitions

        # --- σ_y measurement: active topic (v67 FIX) ---
        # OLD: coherence = s & s_prev (AND) — DEAD with sparse SDRs!
        #   Expected overlap with k=10/D=512: 10*10/512 ≈ 0.2 bits/step
        #   → m_y was essentially always zero
        #
        # NEW: coherence = min(s, m_z) * Y_SIGNAL_SCALE
        #   min(s, m_z) captures "which currently-active spins are also
        #   topic spins" — the intersection of "what's active now" and
        #   "what's been persistently active." This gives ~10 non-zero
        #   entries (the current SDR) weighted by their topic magnetization.
        #
        #   With Y_SIGNAL_SCALE=4, the signal is amplified to compensate
        #   for τ_y=15 being 3x larger than τ_x=5.
        #
        #   Physical meaning: σ_y measures PHASE COHERENCE between states.
        #   The intersection of current state and topic magnetization is
        #   the discrete analogue: bits that are both active AND part of
        #   the persistent pattern are "coherent" — they haven't decayed
        #   despite the DAM settling into a new attractor.
        coherence = np.minimum(s, self.m_z) * self.Y_SIGNAL_SCALE
        decay_y = self.m_y // self.TAU_Y
        self.m_y = self.m_y - decay_y + coherence

        # --- Cross-band precession (Heisenberg dynamics) ---
        # dσ/dt = i[H, σ] — the spin precesses around the effective field.
        # In our integer approximation:
        #   Z→X: topic constrains which transitions are likely
        #   X→Y: transitions shape the coherence pattern
        #   Y→Z: coherence reinforces the topic
        #
        # This creates a DYNAMICAL SYSTEM that maintains coherent
        # discourse state, not just independent measurements.
        zx_coupling = self.m_z >> self.ZX_SHIFT  # 1/16 of Z
        xy_coupling = self.m_x >> self.XY_SHIFT  # 1/16 of X
        yz_coupling = self.m_y >> self.YZ_SHIFT  # 1/32 of Y

        # Apply precession (shift-and-add, all integer)
        self.m_x = self.m_x + zx_coupling - xy_coupling
        self.m_y = self.m_y + xy_coupling - yz_coupling
        self.m_z = self.m_z + yz_coupling - zx_coupling

        # Clip to valid range [0, τ*2] to prevent unbounded growth
        # from precession coupling. The *2 allows some overshoot
        # from cross-coupling without letting values diverge.
        np.clip(self.m_z, 0, self.TAU_Z * 2, out=self.m_z)
        np.clip(self.m_x, 0, self.TAU_X * 2, out=self.m_x)
        np.clip(self.m_y, 0, self.TAU_Y * 2, out=self.m_y)

        # Store current state for next XOR computation
        self._prev_state = state.copy()
        self._step_count += 1

    def compute_z_field(self, J: np.ndarray) -> np.ndarray:
        """
        Compute the field from the Z (magnetization/topic) band.

        field_z[i] = Σ_j J[i,j] * m_z[j]  for all j where m_z > 0

        This field provides TOPIC-LEVEL coherence — persistent signal
        about what the text is about. Words whose SDRs overlap with
        the topic magnetization get a strong field, making them
        more likely to be selected.

        The Z band is the STRONGEST coherence signal because topic
        is the primary determinant of word selection at the discourse
        level (analogous to the key-value attention in transformers).

        Args:
            J: Coupling matrix (D, D) int16.

        Returns:
            Field vector (D,) int32.
        """
        active = np.where(self.m_z > 0)[0]
        if len(active) == 0:
            return np.zeros(self.D, dtype=np.int32)

        m_vals = self.m_z[active].astype(np.int32)
        field = J[:, active].astype(np.int32) @ m_vals
        return field

    def compute_x_field(self, J: np.ndarray) -> np.ndarray:
        """
        Compute the field from the X (transitions/narrative) band.

        field_x[i] = Σ_j J[i,j] * m_x[j]  for all j where m_x > 0

        This field provides NARRATIVE DIRECTION — which dimensions
        are actively transitioning, and what tends to follow from
        those transitions. This is a PREDICTIVE signal: it doesn't
        just say what IS active, but what is CHANGING and therefore
        what's about to become active.

        In transformer terms, this is analogous to the QUERY part
        of attention — it encodes "what am I looking for next?"

        Args:
            J: Coupling matrix (D, D) int16.

        Returns:
            Field vector (D,) int32.
        """
        active = np.where(self.m_x > 0)[0]
        if len(active) == 0:
            return np.zeros(self.D, dtype=np.int32)

        m_vals = self.m_x[active].astype(np.int32)
        field = J[:, active].astype(np.int32) @ m_vals
        return field

    def compute_y_field(self, J: np.ndarray) -> np.ndarray:
        """
        Compute the field from the Y (coherence/syntactic) band.

        field_y[i] = Σ_j J[i,j] * m_y[j]  for all j where m_y > 0

        This field provides SYNTACTIC MODE coherence — which
        dimensions are coherently co-active, and what tends to
        be associated with that stable pattern. This maintains
        structural consistency: if we've been in a "DET ADJ NOUN"
        pattern, this field keeps us in that syntactic mode.

        In transformer terms, this is analogous to the VALUE part
        of attention — it encodes "what stable pattern should
        I continue following?"

        v67: Y band now uses m_z-weighted current state instead of AND,
        so m_y has much stronger signal (~10 active bits * Y_SIGNAL_SCALE=4).

        Args:
            J: Coupling matrix (D, D) int16.

        Returns:
            Field vector (D,) int32.
        """
        active = np.where(self.m_y > 0)[0]
        if len(active) == 0:
            return np.zeros(self.D, dtype=np.int32)

        m_vals = self.m_y[active].astype(np.int32)
        field = J[:, active].astype(np.int32) @ m_vals
        return field

    def sentence_reset(self) -> None:
        """
        Reset at sentence boundary: different decay per band.

        Physics: at a sentence boundary, the discourse undergoes a
        "measurement" — the syntactic structure collapses and a new
        one begins, but the topic and style persist.

        Z band (topic):  Soft decay (3/4) — topic persists across sentences
          "We were talking about a fairy tale" → still about that tale
        X band (transitions): Hard reset — transitions don't cross sentences
          "The narrative was heading toward conflict" → new direction needed
        Y band (coherence):  Medium decay (1/2) — some structural style persists
          "We were in formal register" → likely stays formal

        This is the spin-matrix analogue of partial measurement:
        σ_z is measured softly (topic mostly preserved),
        σ_x is measured sharply (transitions collapse),
        σ_y is measured moderately (style partially preserved).
        """
        # X band: hard reset — transitions don't persist across sentences
        self.m_x[:] = 0

        # Y band: medium decay (1/2) — syntactic style partially persists
        self.m_y = self.m_y >> 1

        # Z band: soft decay (3/4) — topic mostly persists
        self.m_z = (self.m_z * 3) // 4

    def full_reset(self) -> None:
        """Hard reset all bands — used at start of new generation."""
        self.m_z[:] = 0
        self.m_x[:] = 0
        self.m_y[:] = 0
        self._prev_state[:] = 0
        self._step_count = 0

    def get_diagnostics(self) -> dict:
        """Return diagnostics for the three-band spin state."""
        z_active = int(np.sum(self.m_z > self._z_threshold))
        x_active = int(np.sum(self.m_x > self._x_threshold))
        y_active = int(np.sum(self.m_y > self._y_threshold))
        z_max = int(np.max(self.m_z))
        x_max = int(np.max(self.m_x))
        y_max = int(np.max(self.m_y))
        z_mean = float(np.mean(self.m_z[self.m_z > 0])) if np.any(self.m_z > 0) else 0.0
        x_mean = float(np.mean(self.m_x[self.m_x > 0])) if np.any(self.m_x > 0) else 0.0
        y_mean = float(np.mean(self.m_y[self.m_y > 0])) if np.any(self.m_y > 0) else 0.0

        # τ-normalized magnetization stats (m/τ gives "average activity" in [0,1])
        z_norm_max = float(z_max) / self.TAU_Z if self.TAU_Z > 0 else 0.0
        x_norm_max = float(x_max) / self.TAU_X if self.TAU_X > 0 else 0.0
        y_norm_max = float(y_max) / self.TAU_Y if self.TAU_Y > 0 else 0.0

        return {
            'step_count': self._step_count,
            # Z band (magnetization/topic)
            'z_active': z_active,
            'z_max': z_max,
            'z_mean_nonzero': z_mean,
            'z_norm_max': z_norm_max,
            'z_threshold': self._z_threshold,
            'tau_z': self.TAU_Z,
            # X band (transitions/narrative)
            'x_active': x_active,
            'x_max': x_max,
            'x_mean_nonzero': x_mean,
            'x_norm_max': x_norm_max,
            'x_threshold': self._x_threshold,
            'tau_x': self.TAU_X,
            # Y band (coherence/syntax)
            'y_active': y_active,
            'y_max': y_max,
            'y_mean_nonzero': y_mean,
            'y_norm_max': y_norm_max,
            'y_threshold': self._y_threshold,
            'tau_y': self.TAU_Y,
        }
