"""
Energy Computer — combines all energy signals for word selection.

Energy hierarchy:
  1. Multi-Scale Recall (PRIMARY): word + POS + topic n-gram energies
  2. Document State (SECONDARY): topic/mode/tense/negation/specificity/argument conditioning
  3. Document State Coupling (v18): pairwise compatibility energy E_coupling
  4. ESN Reservoir (v18): long-range temporal dynamics E_reservoir
  5. VSA/qFHRR (v18): compositional vector symbolic architecture E_vsa
  6. Macro-Spin Layer (v19): entity + phase + scene coupling E_macro
  7. Latent Spin Glass (v21): LEARNED spin vectors + LEARNED coupling (replaces hand-coded)
  8. Hard constraints: POS type penalties, same-word penalty, closed-class double penalty

All additive: E(w) = E_recall + E_state + E_coupling + E_reservoir + E_vsa
                    + E_macro + E_latent + E_hard
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np

from ..utils import validate_array
from ..vocabulary.pos import POSTypeSystem, CLOSED_CLASS, POS2IDX

if TYPE_CHECKING:
    from ..recall import MultiScaleRecall
    from ..state import DocumentState
    from ..reservoir.integer_esn import IntegerESN
    from ..reservoir.multi_timescale import MultiTimescaleReservoir
    from ..vsa.qfhrr import VSAEncoder
    from ..macro import MacroSpinCoupling
    from ..ssr.semantic_spin import SemanticSpinResonance
    from ..latent.latent_spin import LatentSpinGlass


class EnergyComputer:
    """
    Combines all energy signals for word selection.

    Energy hierarchy:
      1. Multi-Scale Recall (PRIMARY): word + POS + topic n-gram energies
      2. Document State (SECONDARY): state-word compatibility conditioning
      3. Document State Coupling (v18): pairwise state-variable compatibility
      4. ESN Reservoir (v18): long-range temporal dynamics (~50 token lookback)
      5. VSA/qFHRR (v18): compositional vector symbolic architecture energy
      6. Macro-Spin Layer (v19): entity/phase/scene long-range coupling
      7. Latent Spin Glass (v21): LEARNED spins + LEARNED coupling (EMERGENT understanding)
      8. Hard constraints: POS type, same-word, closed-class penalties

    All energy terms are ADDITIVE (Ising model physics: E = Σ E_i).
    All arithmetic is integer-only. LOWER energy = more likely.
    """

    def __init__(
        self,
        multiscale_recall: "MultiScaleRecall",
        document_state: "DocumentState",
        pos_system: POSTypeSystem,
        # v17 energy scales
        recall_scale: int = 1600,
        pos_recall_scale: int = 800,
        topic_recall_scale: int = 400,
        state_scale: int = 200,
        same_word_penalty: int = 200,
        closed_class_double_penalty: int = 50000,
        max_closed_class_run: int = 2,
        # Recall interpolation settings (CRITICAL: must match training)
        interpolated: bool = True,
        kn_backoff: bool = True,
        # v18 energy scales
        coupling_scale: int = 200,
        reservoir_scale: int = 800,
        vsa_scale: int = 800,
        # v18 optional modules (None = disabled)
        reservoir: Optional["IntegerESN"] = None,
        vsa_encoder: Optional["VSAEncoder"] = None,
        # v19: Macro-spin layer for long-range coherence
        macro_coupling: Optional["MacroSpinCoupling"] = None,
        macro_scale: int = 800,
        # v19.1: Multi-Timescale Reservoir (EMERGENT long-range coherence)
        mtr: Optional["MultiTimescaleReservoir"] = None,
        # v20: Semantic Spin Resonance (EMERGENT understanding via frustrated dynamics)
        ssr: Optional["SemanticSpinResonance"] = None,
        ssr_scale: int = 1200,
        # v21: Learned Latent Spin Glass (GENUINE understanding from learned physics)
        latent_spin: Optional["LatentSpinGlass"] = None,
    ):
        self.multiscale_recall = multiscale_recall
        self.document_state = document_state
        self.pos_system = pos_system
        self.recall_scale = recall_scale
        self.pos_recall_scale = pos_recall_scale
        self.topic_recall_scale = topic_recall_scale
        self.state_scale = state_scale
        self.same_word_penalty = same_word_penalty
        self.closed_class_double_penalty = closed_class_double_penalty
        self.max_closed_class_run = max_closed_class_run

        # Recall interpolation settings
        self.interpolated = interpolated
        self.kn_backoff = kn_backoff

        # v18 energy scales
        self.coupling_scale = coupling_scale
        self.reservoir_scale = reservoir_scale
        self.vsa_scale = vsa_scale

        # v18 modules
        self.reservoir = reservoir
        self.vsa_encoder = vsa_encoder

        # v19: Macro-spin layer
        self.macro_coupling = macro_coupling
        self.macro_scale = macro_scale

        # v19.1: Multi-Timescale Reservoir
        self.mtr = mtr

        # v20: Semantic Spin Resonance
        self.ssr = ssr
        self.ssr_scale = ssr_scale

        # v21: Learned Latent Spin Glass
        self.latent_spin = latent_spin

        # Pre-compute closed-class POS ID set (avoids rebuilding per call)
        self._closed_class_pos_ids: frozenset[int] = frozenset(
            POS2IDX[c] for c in CLOSED_CLASS
        )

    def compute_energy(
        self,
        context_words: List[int],
        candidate_words: np.ndarray,
        current_type: int = -1,
        prev_word: int = -1,
        closed_class_run: int = 0,
    ) -> np.ndarray:
        """
        Compute total energy for all candidate words.

        E(w) = E_recall(w) + E_state(w) + E_coupling(w) + E_reservoir(w)
             + E_vsa(w) + E_macro(w) + E_latent(w) + E_hard(w)

        All hard constraints use vectorized numpy boolean indexing.
        v18/v19/v21 energy terms are only computed if their respective
        modules are enabled (not None and built).

        Args:
            context_words: List of integer word IDs forming the context.
            candidate_words: 1-D numpy ndarray of integer candidate word IDs.
            current_type: Required POS type ID, or -1 if unconstrained.
            prev_word: Previous word ID, or -1 if no previous word.
            closed_class_run: Current count of consecutive closed-class words.

        Returns:
            energies: int64 array of shape (len(candidate_words),).
            LOWER = more likely.

        Raises:
            TypeError: If input types are invalid.
        """
        # ── Input validation ────────────────────────────────────────────────
        if not isinstance(context_words, list):
            raise TypeError(
                f"context_words must be a list, got {type(context_words).__name__}"
            )
        validate_array(candidate_words, "candidate_words", ndim=1)
        if not isinstance(current_type, int):
            raise TypeError(
                f"current_type must be an int, got {type(current_type).__name__}"
            )
        if not isinstance(prev_word, int):
            raise TypeError(
                f"prev_word must be an int, got {type(prev_word).__name__}"
            )

        n_candidates = len(candidate_words)
        energies = np.zeros(n_candidates, dtype=np.int64)

        # 1. Multi-scale recall energy (PRIMARY)
        # CRITICAL: Forward interpolated and kn_backoff settings
        # These MUST match the training configuration, otherwise POS/topic
        # recall is severely weakened during generation.
        recall_energy = self.multiscale_recall.compute_energy(
            context_words, candidate_words,
            word_scale=self.recall_scale,
            pos_scale=self.pos_recall_scale,
            topic_scale=self.topic_recall_scale,
            longest_only=not self.interpolated,
            interpolated=self.interpolated,
            kn_backoff=self.kn_backoff,
        )
        energies += recall_energy

        # 2. Document state energy (SECONDARY)
        #    Can be disabled by setting state_scale=0 (e.g., when using
        #    learned latent spins as the primary long-range mechanism)
        if self.state_scale > 0:
            state_energy = self.document_state.compute_energy(
                candidate_words, state_scale=self.state_scale
            )
            energies += state_energy

        # 3. Coupling energy (v18) — scalar offset, same for all candidates
        #    This ensures mean-field-inferred state values are consistent.
        #    Adds a constant energy offset based on current state compatibility.
        if self.coupling_scale > 0 and self.document_state._coupling_built:
            coupling_e = self.document_state.compute_coupling_energy(
                coupling_scale=self.coupling_scale
            )
            # Scalar: add same offset to all candidates (shifts baseline)
            energies += coupling_e

        # 4. ESN Reservoir energy (v18) — long-range temporal dynamics
        if (self.reservoir is not None
                and self.reservoir_scale > 0
                and self.reservoir.built):
            reservoir_energy = self.reservoir.compute_energy(
                candidate_words,
                reservoir_scale=self.reservoir_scale,
            )
            energies += reservoir_energy

        # 5. VSA/qFHRR energy (v18) — compositional vector symbolic architecture
        if (self.vsa_encoder is not None
                and self.vsa_scale > 0
                and self.vsa_encoder.built):
            # Build context encoding for VSA
            context_encoding = self.vsa_encoder.compute_context_encoding(
                context_word_ids=context_words,
                context_pos_ids=None,   # Will use defaults if not available
                context_topic_ids=None, # Will use defaults if not available
            )
            vsa_energy = self.vsa_encoder.compute_vsa_energy(
                context_encoding,
                candidate_words,
                vsa_scale=self.vsa_scale,
            )
            energies += vsa_energy

        # 6. Macro-Spin energy (v19) — HAND-CODED long-range bias
        #    Optional supplementary layer with entity/phase/scene rules.
        #    Can be disabled in favor of the emergent MTR approach.
        if (self.macro_coupling is not None
                and self.macro_scale > 0
                and self.macro_coupling.built):
            macro_energy = self.macro_coupling.compute_energy(candidate_words)
            energies += macro_energy

        # 6b. Multi-Timescale Reservoir (v19.1) — EMERGENT long-range coherence
        #    The PHYSICALLY CORRECT approach: multiple reservoirs at different
        #    timescales (fast/medium/slow) with LEARNED readout matrices.
        #    At α=0.997, the slow reservoir maintains 30% of position 0's
        #    information at position 400, giving correlation length ξ >> 400.
        #    No hand-coded rules — the model discovers what matters at each
        #    timescale through the readout weights.
        if self.mtr is not None and self.mtr.built:
            mtr_energy = self.mtr.compute_energy(candidate_words)
            energies += mtr_energy

        # 6c. Semantic Spin Resonance (v20) — EMERGENT UNDERSTANDING
        #    The genuinely novel approach: binary spins with frustrated couplings
        #    and Hebbian episodic memory. The spin state encodes MEANING as a
        #    distributed representation. The episodic coupling creates ATTRACTORS
        #    that enable genuine long-range recall. No hand-coded rules, no
        #    linear dynamics, no static readout. Pure spin glass physics.
        if self.ssr is not None and self.ssr.built:
            ssr_energy = self.ssr.compute_energy(candidate_words)
            energies += ssr_energy

        # 6d. Learned Latent Spin Glass (v21) — GENUINE UNDERSTANDING FROM LEARNED PHYSICS
        #    The CORRECT approach: LEARNED spin vectors (not random, not hand-coded)
        #    + LEARNED coupling matrix (not random, not hand-coded) + Ising dynamics.
        #    The spin dimensions are LATENT — discovered from data, not declared.
        #    The coupling captures REAL dependency structure from training data.
        #    Long-range dependencies EMERGE from the learned coupling structure.
        #    This is Hopfield pattern completion: sigma_doc is a partial pattern,
        #    and the model completes it by selecting words whose spin vectors
        #    align through the learned coupling. Pure spin glass physics.
        if self.latent_spin is not None and self.latent_spin.built:
            latent_energy = self.latent_spin.compute_energy(candidate_words)
            energies += latent_energy

        # 7. Hard constraints (vectorized)

        # Same-word penalty: vectorized comparison
        if prev_word >= 0:
            energies[candidate_words == prev_word] += self.same_word_penalty

        # Closed-class double penalty
        if closed_class_run >= self.max_closed_class_run:
            is_closed_class = np.array([
                bool(self.pos_system.allowed_types.get(int(w), set()) & self._closed_class_pos_ids)
                for w in candidate_words
            ], dtype=bool)
            energies[is_closed_class] += self.closed_class_double_penalty

        # POS type penalty
        if current_type >= 0:
            violates_pos = np.array([
                (allowed := self.pos_system.allowed_types.get(int(w), set()))
                and current_type not in allowed
                for w in candidate_words
            ], dtype=bool)
            energies[violates_pos] += 50000  # Hard constraint (was 5000, too soft for scale=1600 recall)

        return energies
