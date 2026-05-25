"""
Narrative Phase Tracker — Macro-spin for narrative arc coherence.

In the Ising spin glass, this implements a macro-spin with discrete
states and ENERGY BARRIERS between transitions.  The narrative phase
persists across hundreds of tokens and biases word selection toward
vocabulary appropriate for the current phase.

Architecture:
  - Phase states: SETTING(0), INTRODUCTION(1), RISING(2), CLIMAX(3),
    RESOLUTION(4), END(5)
  - Phase transitions driven by:
      1. Soft time progression: after N tokens, bias toward next phase
      2. Trigger words: questions → RISING, exclamations → CLIMAX,
         "finally"/"happily" → RESOLUTION
      3. Entity density: new entities → INTRODUCTION, entity conflicts → RISING
  - Phase-word affinity: (6, vocab_size) int16 matrix
      Learned from training: words that typically appear in each phase.

Energy contribution:
  E_phase(w) = -phase_word_affinity[phase_id, w] * phase_scale / MAX_AFFINITY

  This creates a LONG-RANGE BIAS: vocabulary shifts as the narrative
  progresses through its arc, maintaining coherence over 400+ tokens.

Key physics: Phase transitions require overcoming an ENERGY BARRIER.
The phase doesn't flip randomly on every token — it persists until
a strong enough trigger (time progression + trigger words) pushes it
over the barrier.  This is the spin glass metastable state.

Memory budget (V=2000):
  - phase_word_affinity: 6 × 2000 × 2 bytes = 24 KB
  - Phase state: negligible
  Total: ~24 KB (trivial)
"""

import numpy as np
from typing import Dict, List, Optional


# --- Phase constants ---
PHASE_SETTING = 0       # Scene description, background
PHASE_INTRODUCTION = 1  # Characters introduced, plot setup
PHASE_RISING = 2        # Conflict, tension building
PHASE_CLIMAX = 3        # Peak tension, turning point
PHASE_RESOLUTION = 4    # Conflict resolved, denouement
PHASE_END = 5           # Conclusion, "happily ever after"

N_PHASES = 6

PHASE_NAMES = {
    PHASE_SETTING: "SETTING",
    PHASE_INTRODUCTION: "INTRODUCTION",
    PHASE_RISING: "RISING",
    PHASE_CLIMAX: "CLIMAX",
    PHASE_RESOLUTION: "RESOLUTION",
    PHASE_END: "END",
}

# --- Phase transition trigger words ---
# Maps words to phase transitions: (target_phase, barrier_reduction)
# barrier_reduction: how much the energy barrier is reduced by this word
RISING_TRIGGERS = {
    "but", "however", "suddenly", "then", "when", "although",
    "problem", "trouble", "wrong", "afraid", "worried", "scared",
    "danger", "lost", "dark", "strange", "mysterious", "surprised",
    "?",  # Questions often signal rising action
}
CLIMAX_TRIGGERS = {
    "!", "shouted", "cried", "screamed", "rushed", "grabbed",
    "fight", "battle", "crash", "bang", "exploded", "never",
    "couldn't", "wouldn't", "must", "had to",
}
RESOLUTION_TRIGGERS = {
    "finally", "happily", "safe", "found", "smiled", "laughed",
    "glad", "relieved", "friend", "together", "home", "better",
    "okay", "good", "wonderful", "beautiful", "perfect",
}
END_TRIGGERS = {
    "end", "ever after", "lived", "conclusion", "finished",
    "done", "last", "final", "goodbye",
}
INTRODUCTION_TRIGGERS = {
    "once", "upon", "there", "named", "called", "met",
    "introduced", "first", "new", "lived", "was a",
}

# --- Phase transition thresholds (in tokens) ---
# These define the soft time progression toward the next phase
# A phase transition is proposed when position exceeds the threshold,
# but still requires overcoming the energy barrier
PHASE_THRESHOLDS = {
    PHASE_SETTING: 20,        # Setting established in ~20 tokens
    PHASE_INTRODUCTION: 50,   # Characters introduced by ~50 tokens
    PHASE_RISING: 120,        # Rising action until ~120 tokens
    PHASE_CLIMAX: 240,        # Climax around ~240 tokens
    PHASE_RESOLUTION: 330,    # Resolution by ~330 tokens
    PHASE_END: 380,           # End by ~380 tokens
}


class NarrativePhaseTracker:
    """
    Macro-spin for narrative arc coherence with energy barriers.

    The narrative phase persists across hundreds of tokens, creating
    a long-range correlation in word selection.  Phase transitions
    require overcoming an energy barrier, preventing random flipping.

    All arithmetic is integer-only.
    """

    # Energy barrier for phase transitions (in arbitrary units)
    # Higher = harder to flip phase
    # With progression_rate=8 and thresholds above, a text of 400 tokens
    # naturally transitions through all phases
    DEFAULT_BARRIER = 200

    # Phase-word affinity normalization
    AFFINITY_Q = 8  # Q8 fixed-point
    MAX_AFFINITY = 255  # Max value after Q8 normalization

    def __init__(
        self,
        vocab_size: int,
        phase_scale: int = 600,
        transition_barrier: int = 500,
        idx2word: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize NarrativePhaseTracker.

        Args:
            vocab_size: Vocabulary size V.
            phase_scale: Energy scale for phase coupling (default 600).
            transition_barrier: Energy barrier for phase transitions (default 500).
            idx2word: Optional mapping from word ID to word string.
        """
        self.vocab_size = vocab_size
        self.phase_scale = phase_scale
        self.transition_barrier = transition_barrier
        self.idx2word = idx2word

        # Current phase
        self.phase: int = PHASE_SETTING
        self._position: int = 0

        # Phase-word affinity matrix: (N_PHASES, V) int16
        self.affinity: Optional[np.ndarray] = None
        self._phase_counts: Optional[np.ndarray] = None

        # Whether affinity has been built
        self._built = False

        # Diagnostics
        self._stats = {
            'phase_transitions': 0,
            'trigger_transitions': 0,
            'time_transitions': 0,
        }

    def reset(self) -> None:
        """Reset phase for a new document."""
        self.phase = PHASE_SETTING
        self._position = 0
        self._stats = {
            'phase_transitions': 0,
            'trigger_transitions': 0,
            'time_transitions': 0,
        }

    def _compute_transition_field(self, word_str: Optional[str] = None) -> int:
        """
        Compute the effective field driving phase transitions.

        The field is a sum of:
        1. Time progression: soft bias toward next phase based on position
        2. Trigger words: strong bias from specific words
        3. Barrier: the energy barrier resisting transitions

        If field > barrier, the phase transitions.

        Returns:
            Net field value (positive = toward transition).
        """
        field = 0

        # Time progression: bias increases as position exceeds phase threshold
        threshold = PHASE_THRESHOLDS.get(self.phase, 400)
        if self._position > threshold:
            # Field increases with distance past threshold
            # progression_rate controls how quickly the phase advances
            overshoot = self._position - threshold
            progression_rate = 8  # Field units per token past threshold
            field += overshoot * progression_rate

        # Trigger words
        if word_str is not None:
            w = word_str.lower().strip()
            if w in RISING_TRIGGERS and self.phase < PHASE_RISING:
                field += 200
            if w in CLIMAX_TRIGGERS and self.phase < PHASE_CLIMAX:
                field += 300
            if w in RESOLUTION_TRIGGERS and self.phase >= PHASE_RISING:
                field += 250
            if w in END_TRIGGERS and self.phase >= PHASE_RESOLUTION:
                field += 400
            if w in INTRODUCTION_TRIGGERS and self.phase == PHASE_SETTING:
                field += 150

        return field

    def _determine_target_phase(self, word_str: Optional[str] = None) -> int:
        """
        Determine the target phase based on current state and triggers.

        Returns:
            Target phase ID.
        """
        # Default: progress to next phase
        target = min(self.phase + 1, PHASE_END)

        # Trigger-based overrides
        if word_str is not None:
            w = word_str.lower().strip()
            if w in INTRODUCTION_TRIGGERS and self.phase <= PHASE_SETTING:
                target = PHASE_INTRODUCTION
            elif w in RISING_TRIGGERS and self.phase <= PHASE_INTRODUCTION:
                target = PHASE_RISING
            elif w in CLIMAX_TRIGGERS and self.phase <= PHASE_RISING:
                target = PHASE_CLIMAX
            elif w in RESOLUTION_TRIGGERS and self.phase >= PHASE_RISING:
                target = PHASE_RESOLUTION
            elif w in END_TRIGGERS and self.phase >= PHASE_RESOLUTION:
                target = PHASE_END

        return target

    def update(self, word_id: int, word_str: Optional[str] = None) -> None:
        """
        Update narrative phase based on the current word and position.

        Implements the macro-spin dynamics: compute transition field,
        compare to barrier, and flip if field exceeds barrier.

        Args:
            word_id: Integer token ID.
            word_str: Optional string form of the word.
        """
        self._position += 1

        # Compute transition field
        field = self._compute_transition_field(word_str)

        # Check if field exceeds barrier → phase transition
        if field > self.transition_barrier:
            target = self._determine_target_phase(word_str)
            if target != self.phase:
                old_phase = self.phase
                self.phase = target
                self._stats['phase_transitions'] += 1

                # Track what triggered the transition
                if word_str is not None and word_str.lower().strip() in (
                    RISING_TRIGGERS | CLIMAX_TRIGGERS | RESOLUTION_TRIGGERS |
                    END_TRIGGERS | INTRODUCTION_TRIGGERS
                ):
                    self._stats['trigger_transitions'] += 1
                else:
                    self._stats['time_transitions'] += 1

    # ===================================================================
    # BUILD: Learn phase-word affinity from training data
    # ===================================================================

    def build(
        self,
        sequences: List[List[int]],
        idx2word: Optional[Dict[int, str]] = None,
    ) -> "NarrativePhaseTracker":
        """
        Build phase-word affinity matrix from training data.

        For each position in each training sequence, we assign a
        narrative phase based on relative position within the sequence,
        then accumulate word-phase co-occurrence counts.

        The phase assignment uses the phase thresholds scaled to the
        sequence length:
          - Setting: first 10% of sequence
          - Introduction: 10-25%
          - Rising: 25-55%
          - Climax: 55-75%
          - Resolution: 75-95%
          - End: 95-100%

        Args:
            sequences: List of training sequences.
            idx2word: Mapping from word ID to word string.

        Returns:
            self
        """
        if idx2word is not None:
            self.idx2word = idx2word

        V = self.vocab_size

        # Accumulate co-occurrence counts: (N_PHASES, V) int32
        phase_word_counts = np.zeros((N_PHASES, V), dtype=np.int32)
        phase_totals = np.zeros(N_PHASES, dtype=np.int32)

        n_sequences = len(sequences)
        total_positions = 0

        for seq_idx, seq in enumerate(sequences):
            if len(seq) < 3:
                continue

            seq_len = len(seq)
            # Phase boundaries relative to sequence position
            # Using percentage-based boundaries
            boundaries = [
                (0.00, PHASE_SETTING),
                (0.10, PHASE_INTRODUCTION),
                (0.25, PHASE_RISING),
                (0.55, PHASE_CLIMAX),
                (0.75, PHASE_RESOLUTION),
                (0.95, PHASE_END),
            ]

            for pos, word_id in enumerate(seq):
                if word_id < 0 or word_id >= V:
                    continue

                # Determine phase from relative position
                relative_pos = pos / max(1, seq_len - 1)
                assigned_phase = PHASE_SETTING
                for boundary, phase_id in boundaries:
                    if relative_pos >= boundary:
                        assigned_phase = phase_id

                phase_word_counts[assigned_phase, word_id] += 1
                phase_totals[assigned_phase] += 1
                total_positions += 1

            # Progress reporting
            if (seq_idx + 1) % 50000 == 0:
                print(f"    NarrativePhaseTracker.build(): {seq_idx+1}/{n_sequences} sequences")

        # Normalize affinity: Q8 * count / max_across_phases
        # We want to capture which words are DISTINCTIVE for each phase
        # Use log-likelihood ratio: log2(P(word|phase) / P(word))
        self.affinity = np.zeros((N_PHASES, V), dtype=np.int16)

        total_words = max(1, int(phase_word_counts.sum()))

        for phase_id in range(N_PHASES):
            phase_total = max(1, int(phase_totals[phase_id]))

            for w in range(V):
                count = int(phase_word_counts[phase_id, w])
                if count == 0:
                    continue

                # P(word|phase) = count / phase_total
                # P(word) = sum_over_phases(count) / total_words
                total_word_count = int(phase_word_counts[:, w].sum())
                if total_word_count == 0:
                    continue

                # Log-likelihood ratio in fixed-point
                # ratio = (count / phase_total) / (total_word_count / total_words)
                #       = (count * total_words) / (phase_total * total_word_count)
                numerator = count * total_words
                denominator = phase_total * total_word_count

                if denominator > 0 and numerator > denominator:
                    # Word is over-represented in this phase → positive affinity
                    ratio = numerator // denominator
                    if ratio >= 2:
                        # log2(ratio) using bit length
                        log2_ratio = ratio.bit_length() - 1
                        affinity_val = min(log2_ratio * (1 << self.AFFINITY_Q), 32767)
                        self.affinity[phase_id, w] = affinity_val
                elif denominator > 0 and numerator > 0:
                    # Word is under-represented → negative affinity
                    ratio = denominator // max(1, numerator)
                    if ratio >= 2:
                        log2_ratio = ratio.bit_length() - 1
                        affinity_val = min(log2_ratio * (1 << self.AFFINITY_Q), 32767)
                        self.affinity[phase_id, w] = -affinity_val

        self._phase_counts = phase_totals
        self._built = True

        mem_mb = self.affinity.nbytes / (1024 * 1024)
        print(f"    NarrativePhaseTracker.build(): {n_sequences} sequences, "
              f"{total_positions} positions, memory={mem_mb:.3f} MB")

        return self

    # ===================================================================
    # ENERGY: Compute narrative phase macro-spin energy
    # ===================================================================

    def compute_energy(
        self,
        candidate_words: np.ndarray,
        phase_scale: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute narrative phase macro-spin energy for candidate words.

        E_phase(w) = -affinity[phase_id, w] * phase_scale / MAX_AFFINITY

        Words with POSITIVE affinity for the current phase get LOWER
        energy (more likely).  Words with NEGATIVE affinity get HIGHER
        energy (less likely).

        This creates a LONG-RANGE BIAS: as the narrative progresses
        through its arc, vocabulary shifts to match the phase.

        Args:
            candidate_words: Array of candidate word IDs.
            phase_scale: Override energy scale.

        Returns:
            np.ndarray of int64 energies, shape (n_candidates,).
        """
        n_candidates = len(candidate_words)
        if not self._built or self.affinity is None:
            return np.zeros(n_candidates, dtype=np.int64)

        scale = phase_scale if phase_scale is not None else self.phase_scale

        # Look up affinity for current phase
        safe_candidates = np.clip(candidate_words, 0, self.vocab_size - 1)
        aff = self.affinity[self.phase, safe_candidates].astype(np.int64)

        # Energy = -affinity * scale / MAX_AFFINITY
        if self.MAX_AFFINITY > 0:
            energies = -(aff * scale) // self.MAX_AFFINITY
        else:
            energies = np.zeros(n_candidates, dtype=np.int64)

        return energies

    @property
    def built(self) -> bool:
        """Whether the affinity matrix has been built."""
        return self._built

    def get_diagnostics(self) -> Dict:
        """Get diagnostic information about narrative phase."""
        return {
            'phase': PHASE_NAMES.get(self.phase, "UNKNOWN"),
            'phase_id': self.phase,
            'position': self._position,
            'stats': self._stats.copy(),
        }
