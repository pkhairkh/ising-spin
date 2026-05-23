import numpy as np
from typing import Dict, List, Optional
from ising_spin.recall import MultiScaleRecall
from ising_spin.state import DocumentState
from ising_spin.vocabulary.pos import POSTypeSystem, CLOSED_CLASS, POS2IDX
from ising_spin.vsa import VSAEncoder
from ising_spin.dense_am import DenseAMEnergy


class EnergyComputer:
    """
    Combines all energy signals for word selection.

    Energy hierarchy (v18.1):
      1. Multi-Scale Recall (PRIMARY): word + POS + topic n-gram energies
      2. Dense AM (v18.1 NEW): Nonlinear pattern matching with random features
      3. VSA Binding (v18.0): Compositional word+POS+topic similarity energy
      4. Document State (SECONDARY): topic/mode/tense/negation/specificity/argument conditioning
      5. Hard constraints: POS type penalties, same-word penalty, closed-class double penalty

    v18.1: Added E_dense_am — Dense Associative Memory with polynomial nonlinearity.
      This replaces the LINEAR recall energy for word-level matching with a
      NONLINEAR energy function that creates sharper basins. The quadratic
      nonlinearity (degree=2) amplifies differences: good matches get MUCH
      lower energy, bad matches get MUCH higher energy.

    v18.0: Added E_vsa_bind — VSA compositional binding energy.
      This captures interactions between word, POS, and topic that the
      additive v17 model cannot distinguish.

    v17.3 CRITICAL FIX: interpolated and kn_backoff are now forwarded to
    MultiScaleRecall. Previously they defaulted to False, meaning the model
    built KN-smoothed indexes but never used them during energy computation.
    This caused ~2x PPL inflation.
    """

    def __init__(
        self,
        multiscale_recall: MultiScaleRecall,
        document_state: DocumentState,
        pos_system: POSTypeSystem,
        vsa_encoder: Optional[VSAEncoder] = None,       # v18.0: VSA encoder
        dense_am: Optional[DenseAMEnergy] = None,        # v18.1: Dense AM
        recall_scale: int = 1600,      # word n-gram scale (primary)
        pos_recall_scale: int = 800,    # POS n-gram scale
        topic_recall_scale: int = 400,  # topic n-gram scale
        state_scale: int = 400,         # v18.0: document state scale (was 50 in v17)
        vsa_scale: int = 800,           # v18.0: VSA binding energy scale
        dense_am_scale: int = 1200,     # v18.1: Dense AM energy scale
        same_word_penalty: int = 200,
        closed_class_double_penalty: int = 50000,
        max_closed_class_run: int = 2,
        interpolated: bool = True,      # v17.3: NOW FORWARDED (was silently False)
        kn_backoff: bool = True,        # v17.3: NOW FORWARDED (was silently False)
    ):
        self.multiscale_recall = multiscale_recall
        self.document_state = document_state
        self.pos_system = pos_system
        self.vsa_encoder = vsa_encoder
        self.dense_am = dense_am
        self.recall_scale = recall_scale
        self.pos_recall_scale = pos_recall_scale
        self.topic_recall_scale = topic_recall_scale
        self.state_scale = state_scale
        self.vsa_scale = vsa_scale
        self.dense_am_scale = dense_am_scale
        self.same_word_penalty = same_word_penalty
        self.closed_class_double_penalty = closed_class_double_penalty
        self.max_closed_class_run = max_closed_class_run
        self.interpolated = interpolated
        self.kn_backoff = kn_backoff

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

        E(w) = E_recall(w) + E_dense_am(w) + E_vsa(w) + E_state(w) + E_hard(w)

        Returns energies array (int64), LOWER = more likely.
        """
        n_candidates = len(candidate_words)
        energies = np.zeros(n_candidates, dtype=np.int64)

        # 1. Multi-scale recall energy (PRIMARY)
        # v17.3 FIX: Now forwards interpolated and kn_backoff
        recall_energy = self.multiscale_recall.compute_energy(
            context_words, candidate_words,
            longest_only=not self.interpolated,
            interpolated=self.interpolated,
            kn_backoff=self.kn_backoff,
        )
        energies += recall_energy

        # 2. Dense AM energy (v18.1 NEW)
        # Nonlinear pattern matching — sharper energy landscape than linear recall
        if self.dense_am is not None and self.dense_am.built:
            dense_am_energy = self.dense_am.compute_energy(
                context_words, candidate_words,
            )
            energies += dense_am_energy

        # 3. VSA binding energy (v18.0)
        if self.vsa_encoder is not None and self.vsa_encoder.built:
            vsa_energy = self._compute_vsa_energy(context_words, candidate_words)
            energies += vsa_energy

        # 4. Document state energy (SECONDARY)
        state_energy = self.document_state.compute_energy(
            candidate_words, state_scale=self.state_scale
        )
        energies += state_energy

        # 5. Hard constraints
        # Same-word penalty
        if prev_word >= 0:
            for i, w in enumerate(candidate_words):
                if int(w) == prev_word:
                    energies[i] += self.same_word_penalty

        # Closed-class double penalty
        if closed_class_run >= self.max_closed_class_run:
            closed_class_ids = set()
            for i, w in enumerate(candidate_words):
                w_int = int(w)
                allowed = self.pos_system.allowed_types.get(w_int, set())
                if any(t in {POS2IDX[c] for c in CLOSED_CLASS} for t in allowed):
                    energies[i] += self.closed_class_double_penalty

        # POS type penalty (if current_type is specified)
        if current_type >= 0:
            for i, w in enumerate(candidate_words):
                w_int = int(w)
                allowed = self.pos_system.allowed_types.get(w_int, set())
                if allowed and current_type not in allowed:
                    energies[i] += 5000  # Hard constraint

        return energies

    def _compute_vsa_energy(
        self,
        context_words: List[int],
        candidate_words: np.ndarray,
    ) -> np.ndarray:
        """
        Compute VSA binding energy for candidate words.

        E_vsa(w) = -similarity(context_encoding, R[w]) * vsa_scale / max_sim

        The context encoding is computed by superposing the VSA encodings
        of the recent context words. Words whose VSA code is more similar
        to the context code get lower energy (more likely).

        Args:
            context_words: List of context word IDs.
            candidate_words: Array of candidate word IDs.

        Returns:
            np.ndarray of int64 energies, shape (len(candidate_words),).
        """
        # Compute context encoding from recent words
        # Use POS and topic info from the document state if available
        context_pos_ids = None
        context_topic_ids = None

        # Build context POS/topic IDs from word-level assignments
        if self.document_state.word_topics is not None:
            context_topic_ids = []
            for w_id in context_words:
                if 0 <= w_id < len(self.document_state.word_topics):
                    context_topic_ids.append(int(self.document_state.word_topics[w_id]))
                else:
                    context_topic_ids.append(0)

        if self.pos_system is not None:
            context_pos_ids = []
            for w_id in context_words:
                types = self.pos_system.allowed_types.get(w_id, {0})
                context_pos_ids.append(min(types))  # dominant POS

        context_encoding = self.vsa_encoder.compute_context_encoding(
            context_word_ids=context_words,
            context_pos_ids=context_pos_ids,
            context_topic_ids=context_topic_ids,
            window=10,  # use last 10 tokens for VSA context
        )

        return self.vsa_encoder.compute_vsa_energy(
            context_encoding,
            candidate_words,
            vsa_scale=self.vsa_scale,
        )
