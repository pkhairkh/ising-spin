import numpy as np
from typing import Dict, List, Optional
from ising_spin.recall import MultiScaleRecall
from ising_spin.state import DocumentState
from ising_spin.vocabulary.pos import POSTypeSystem, CLOSED_CLASS, POS2IDX


class EnergyComputer:
    """
    Combines all energy signals for word selection.
    
    Energy hierarchy (v17):
      1. Multi-Scale Recall (PRIMARY): word + POS + topic n-gram energies
      2. Document State (SECONDARY): topic/mode/tense/negation/specificity/argument conditioning
      3. Hard constraints: POS type penalties, same-word penalty, closed-class double penalty
    
    The key v17 insight: we no longer have recall + weak perturbation layers.
    Instead, we have recall at MULTIPLE SCALES, each independently constraining
    the prediction. The document state provides discourse-level coherence that
    no n-gram (however abstract) can capture.
    """
    
    def __init__(
        self,
        multiscale_recall: MultiScaleRecall,
        document_state: DocumentState,
        pos_system: POSTypeSystem,
        recall_scale: int = 1600,      # word n-gram scale (primary)
        pos_recall_scale: int = 800,    # POS n-gram scale (secondary)
        topic_recall_scale: int = 400,  # topic n-gram scale (tertiary)
        state_scale: int = 200,         # document state scale
        same_word_penalty: int = 200,
        closed_class_double_penalty: int = 50000,
        max_closed_class_run: int = 2,
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
        
        E(w) = E_recall_multiscale(w) + E_state(w) + E_hard(w)
        
        Returns energies array (int64), LOWER = more likely.
        """
        n_candidates = len(candidate_words)
        energies = np.zeros(n_candidates, dtype=np.int64)
        
        # 1. Multi-scale recall energy (PRIMARY)
        recall_energy = self.multiscale_recall.compute_energy(
            context_words, candidate_words,
            word_scale=self.recall_scale,
            pos_scale=self.pos_recall_scale,
            topic_scale=self.topic_recall_scale,
        )
        energies += recall_energy
        
        # 2. Document state energy (SECONDARY)
        state_energy = self.document_state.compute_energy(
            candidate_words, state_scale=self.state_scale
        )
        energies += state_energy
        
        # 3. Hard constraints
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
