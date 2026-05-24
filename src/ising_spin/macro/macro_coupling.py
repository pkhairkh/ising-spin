"""
Macro-Spin Coupling — Combines entity, phase, and scene energies.

This is the CROSS-SCALE COUPLING layer of the hierarchical spin glass.
It combines the energy contributions from all macro-spin modules:

  E_macro(w) = E_entity(w) + E_phase(w) + E_scene(w)

This energy is ADDITIVE with the existing Hamiltonian:
  E_total(w) = E_recall + E_state + E_coupling + E_reservoir + E_vsa
             + E_macro + E_hard

The macro-spin energy creates LONG-RANGE BIAS in word selection:
  - Active entities bias toward entity-affiliated words (pronouns, associated verbs)
  - Narrative phase biases toward phase-appropriate vocabulary
  - Active scenes bias toward scene-appropriate words

This is the key architectural innovation that gives the model
correlation length ξ >> 400 tokens.

Memory budget (V=2000, TinyStories):
  - Entity affinity: 128 × 2000 × 2 = 512 KB
  - Phase affinity: 6 × 2000 × 2 = 24 KB
  - Scene affinity: 64 × 2000 × 2 = 256 KB
  Total: ~792 KB (negligible on Pi 5)
"""

import numpy as np
from typing import Dict, List, Optional

from .entity_tracker import EntityTracker
from .narrative_phase import NarrativePhaseTracker
from .scene_tracker import SceneTracker


class MacroSpinCoupling:
    """
    Cross-scale coupling layer combining all macro-spin energies.

    This implements the hierarchical spin glass energy contribution:

      E_macro(w) = E_entity(w) + E_phase(w) + E_scene(w)

    Each macro-spin module computes its energy independently using
    persistent macro-spin state (not exponential decay). The energies
    are summed, following Ising model physics: E = Σ E_i.

    The macro-spin energy scale should be comparable to recall scale
    to have meaningful influence on word selection. A good starting
    point: entity_scale=800 (half of word_recall_scale=1600).

    All arithmetic is integer-only.
    """

    def __init__(
        self,
        entity_tracker: Optional[EntityTracker] = None,
        phase_tracker: Optional[NarrativePhaseTracker] = None,
        scene_tracker: Optional[SceneTracker] = None,
        # Energy scales — default to half of primary recall scale
        entity_scale: int = 800,
        phase_scale: int = 600,
        scene_scale: int = 400,
        # Global macro scale multiplier
        macro_scale: int = 1,
    ):
        """
        Initialize MacroSpinCoupling.

        Args:
            entity_tracker: EntityTracker instance (or None to disable).
            phase_tracker: NarrativePhaseTracker instance (or None to disable).
            scene_tracker: SceneTracker instance (or None to disable).
            entity_scale: Energy scale for entity coupling (default 800).
            phase_scale: Energy scale for phase coupling (default 600).
            scene_scale: Energy scale for scene coupling (default 400).
            macro_scale: Global scale multiplier (default 1).
        """
        self.entity_tracker = entity_tracker
        self.phase_tracker = phase_tracker
        self.scene_tracker = scene_tracker

        self.entity_scale = entity_scale
        self.phase_scale = phase_scale
        self.scene_scale = scene_scale
        self.macro_scale = macro_scale

    def compute_energy(
        self,
        candidate_words: np.ndarray,
    ) -> np.ndarray:
        """
        Compute combined macro-spin energy for candidate words.

        E_macro(w) = E_entity(w) + E_phase(w) + E_scene(w)

        Each module only contributes if it has been built and has
        active macro-spins.  If all modules are inactive, E_macro = 0.

        Args:
            candidate_words: Array of candidate word IDs, shape (n,).

        Returns:
            np.ndarray of int64 energies, shape (n,).
            LOWER energy = more likely under macro-spin coupling.
        """
        n_candidates = len(candidate_words)
        total_energy = np.zeros(n_candidates, dtype=np.int64)

        # Entity macro-spin energy
        if self.entity_tracker is not None and self.entity_tracker.built:
            entity_energy = self.entity_tracker.compute_energy(
                candidate_words, entity_scale=self.entity_scale
            )
            total_energy += entity_energy

        # Narrative phase macro-spin energy
        if self.phase_tracker is not None and self.phase_tracker.built:
            phase_energy = self.phase_tracker.compute_energy(
                candidate_words, phase_scale=self.phase_scale
            )
            total_energy += phase_energy

        # Scene macro-spin energy
        if self.scene_tracker is not None and self.scene_tracker.built:
            scene_energy = self.scene_tracker.compute_energy(
                candidate_words, scene_scale=self.scene_scale
            )
            total_energy += scene_energy

        # Apply global macro scale multiplier
        if self.macro_scale != 1:
            total_energy = (total_energy * self.macro_scale)

        return total_energy

    def update(self, word_id: int, word_str: Optional[str] = None) -> None:
        """
        Update all macro-spin modules with the current word.

        This is called after each word is generated, allowing all
        macro-spin modules to update their state (activation decay,
        phase transitions, scene detection).

        Args:
            word_id: Integer token ID.
            word_str: Optional string form of the word.
        """
        if self.entity_tracker is not None:
            self.entity_tracker.update(word_id, word_str)
        if self.phase_tracker is not None:
            self.phase_tracker.update(word_id, word_str)
        if self.scene_tracker is not None:
            self.scene_tracker.update(word_id, word_str)

    def reset(self) -> None:
        """Reset all macro-spin modules for a new document."""
        if self.entity_tracker is not None:
            self.entity_tracker.reset()
        if self.phase_tracker is not None:
            self.phase_tracker.reset()
        if self.scene_tracker is not None:
            self.scene_tracker.reset()

    @property
    def built(self) -> bool:
        """Whether any macro-spin module has been built."""
        return (
            (self.entity_tracker is not None and self.entity_tracker.built)
            or (self.phase_tracker is not None and self.phase_tracker.built)
            or (self.scene_tracker is not None and self.scene_tracker.built)
        )

    def build(
        self,
        sequences: List[List[int]],
        idx2word: Optional[Dict[int, str]] = None,
        raw_texts: Optional[List[str]] = None,
    ) -> "MacroSpinCoupling":
        """
        Build all macro-spin affinity matrices from training data.

        Args:
            sequences: List of training sequences.
            idx2word: Mapping from word ID to word string.
            raw_texts: Optional list of original (pre-tokenization) text strings.

        Returns:
            self
        """
        print("  Building Macro-Spin Layer...")

        if self.entity_tracker is not None and not self.entity_tracker.built:
            print("    Building Entity Tracker affinity...")
            self.entity_tracker.build(sequences, idx2word=idx2word, raw_texts=raw_texts)

        if self.phase_tracker is not None and not self.phase_tracker.built:
            print("    Building Narrative Phase Tracker affinity...")
            self.phase_tracker.build(sequences, idx2word=idx2word)

        if self.scene_tracker is not None and not self.scene_tracker.built:
            print("    Building Scene Tracker affinity...")
            self.scene_tracker.build(sequences, idx2word=idx2word)

        print("  Macro-Spin Layer build complete.")

        return self

    def get_diagnostics(self) -> Dict:
        """Get diagnostic information from all macro-spin modules."""
        diagnostics = {'built': self.built}

        if self.entity_tracker is not None:
            diagnostics['entity'] = self.entity_tracker.get_diagnostics()
        if self.phase_tracker is not None:
            diagnostics['phase'] = self.phase_tracker.get_diagnostics()
        if self.scene_tracker is not None:
            diagnostics['scene'] = self.scene_tracker.get_diagnostics()

        return diagnostics
