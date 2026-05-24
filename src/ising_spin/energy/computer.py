import numpy as np
from typing import List

from ..recall import MultiScaleRecall
from ..state import DocumentState
from ..utils import validate_array
from ..vocabulary.pos import POSTypeSystem, CLOSED_CLASS, POS2IDX


class EnergyComputer:
    """
    Combines all energy signals for word selection.

    Energy hierarchy (v17):
      1. Multi-Scale Recall (PRIMARY): word + POS + topic n-gram energies
      2. Document State (SECONDARY): topic/mode/tense/negation/specificity/argument conditioning
      3. Hard constraints: POS type penalties, same-word penalty, closed-class double penalty

    Refactored with:
      - Relative imports for intra-package references
      - Vectorized numpy boolean indexing for all hard constraints (no Python for-loops)
      - Input validation via ..utils.validate_array and explicit type checks
      - Pre-computed closed-class POS ID set (avoids rebuilding per call)

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

        E(w) = E_recall_multiscale(w) + E_state(w) + E_hard(w)

        All hard constraints use vectorized numpy boolean indexing instead of
        Python for-loops for performance on large candidate sets.

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
            TypeError: If context_words is not a list, candidate_words is not
                an ndarray, or current_type/prev_word are not ints.
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

        # 3. Hard constraints (vectorized)

        # Same-word penalty: vectorized comparison
        if prev_word >= 0:
            energies[candidate_words == prev_word] += self.same_word_penalty

        # Closed-class double penalty: boolean mask over candidates whose
        # allowed POS types intersect with the closed-class POS ID set
        if closed_class_run >= self.max_closed_class_run:
            is_closed_class = np.array([
                bool(self.pos_system.allowed_types.get(int(w), set()) & self._closed_class_pos_ids)
                for w in candidate_words
            ], dtype=bool)
            energies[is_closed_class] += self.closed_class_double_penalty

        # POS type penalty: boolean mask for words whose allowed types
        # exclude the required current_type
        if current_type >= 0:
            violates_pos = np.array([
                (allowed := self.pos_system.allowed_types.get(int(w), set()))
                and current_type not in allowed
                for w in candidate_words
            ], dtype=bool)
            energies[violates_pos] += 5000  # Hard constraint

        return energies
