"""
Evolving integer state variables for discourse coherence.

The discrete analog of a transformer's residual stream.  Each variable
captures a different aspect of "what has happened so far" and "what should
come next".  They span the ENTIRE document, not just the n-gram window.

ALL state variables are integers.  ZERO floating-point.
ALL updates are deterministic rules based on the current word.

State variables:
  topic (1-16):     Current discourse topic (from TopicAssigner)
  mode (1-8):       Discourse mode
  entity (1-64):    Currently focused entity (hash of recent proper nouns)
  tense (1-4):      Tense
  negation (1-3):   Negation scope
  specificity (1-4): Specificity level
  argument_pos (1-6): Argument position

v18.2 NEW: Factorial State Coupling
  The 7 state variables are no longer independent. Pairwise compatibility
tables capture correlations (e.g., topic=SCIENCE co-occurs with
mode=DESCRIPTION, but rarely with mode=ARGUMENT). Mean-field inference
iteratively refines state values using coupling, and coupling energy
penalizes unlikely state combinations.

  Coupled pairs (5 most informative out of C(7,2)=21):
    1. topic × mode       (16×8)  — topic determines discourse mode
    2. topic × tense      (16×4)  — topics have tense preferences
    3. mode × tense       (8×4)   — narrative=past, instruction=present
    4. mode × argument    (8×6)   — argument structure depends on mode
    5. tense × negation   (4×3)   — negation more common in past tense

Key insight: State variables carry information ACROSS THE ENTIRE DOCUMENT.
When "however" appears at position 50, it sets mode=ARGUMENT, and that
persists until the next mode-changing word at position 80+.  No n-gram
can capture this.
"""

from typing import Dict, List, Optional

import numpy as np

from ising_spin.sampling.boltzmann import int_log2_fine
from ising_spin.vocabulary.pos import (
    COARSE_POS_TAGS,
    POS2IDX,
    N_POS,
    NOUN_LIKE,
    VERB_LIKE,
    OPEN_CLASS,
    CLOSED_CLASS,
)


# ===========================================================================
# WORD SETS FOR DETERMINISTIC STATE-UPDATE RULES
# ===========================================================================

# --- Mode triggers ---
_MODE_ARGUMENT_WORDS = frozenset({
    "however", "but", "although", "yet", "nevertheless", "nonetheless",
    "conversely", "on", "contrary", "instead", "rather",
})
_MODE_LIST_WORDS = frozenset({
    "first", "second", "third", "next", "then", "finally", "lastly",
    "fourth", "fifth", "additionally", "furthermore", "moreover",
    "also", "secondly", "thirdly", "firstly",
})
_MODE_QUESTION_WORDS = frozenset({
    "?", "how", "why", "what", "when", "where", "who", "whom",
    "whose", "which", "whether",
})
_MODE_DESCRIPTION_WORDS = frozenset({
    "for", "such", "like", "including", "particularly", "especially",
    "notably", "specifically", "namely", "that", "this",
})
_MODE_INSTRUCTION_WORDS = frozenset({
    "should", "must", "need", "can", "could", "may", "might",
    "shall", "will", "would", "ensure", "make", "let", "allow",
    "require", "avoid", "remember", "consider", "try",
})
_MODE_COMPARISON_WORDS = frozenset({
    "compared", "unlike", "similarly", "whereas", "while", "likewise",
    "correspondingly", "equivalently", "in", "contrast", "conversely",
    "both", "same", "different", "differ", "than",
})
_MODE_SUMMARY_WORDS = frozenset({
    "therefore", "thus", "overall", "consequently", "hence", "accordingly",
    "in", "conclusion", "summarize", "summary", "briefly", "altogether",
    "result", "essentially", "ultimately",
})
_MODE_DESCRIPTION_PREP_WORDS = frozenset({
    "in", "at", "on", "with", "from", "into", "through", "within",
    "among", "upon", "around", "between", "across", "along",
})

# --- Tense triggers ---
_PAST_TENSE_ENDINGS_EXCLUSIONS = frozenset({
    "red", "bed", "fed", "led", "bid", "hid", "rid", "shed",
    "bled", "bred", "fled", "sled", "tread",
})
_FUTURE_WORDS = frozenset({
    "will", "shall", "going", "gonna",
})
_MODAL_VERBS = frozenset({
    "can", "could", "may", "might", "must", "shall", "should",
    "will", "would",
})

# --- Negation triggers ---
_NEGATION_WORDS = frozenset({
    "not", "n't", "never", "no", "neither", "nor", "nobody",
    "nothing", "nowhere", "hardly", "barely", "scarcely",
    "without", "rarely", "seldom",
})

# --- Specificity triggers ---
_NUMERIC_CHARS = frozenset("0123456789")
_QUOTE_MARKS = frozenset({'"', "'", "``", "''", "\u201c", "\u201d"})

# --- Argument position triggers ---
_ARG_PREMISE_WORDS = frozenset({
    "because", "since", "due", "owing", "caused", "resulted",
    "stemmed", "arose", "given", "assuming", "based",
})
_ARG_CLAIM_WORDS = frozenset({
    "therefore", "thus", "so", "hence", "consequently", "accordingly",
    "implies", "suggests", "indicates", "follows", "proves",
})
_ARG_EVIDENCE_WORDS = frozenset({
    "for", "example", "evidence", "data", "studies", "research",
    "shown", "demonstrated", "found", "observed", "reported",
    "survey", "statistics", "findings", "results",
})
_ARG_COUNTER_WORDS = frozenset({
    "however", "but", "although", "conversely", "yet", "despite",
    "whereas", "while", "nevertheless", "on", "contrary",
    "counter", "oppose", "challenge", "dispute",
})
_ARG_REBUTTAL_WORDS = frozenset({
    "nevertheless", "nonetheless", "still", "even", "although",
    "admittedly", "granted", "true", "sure", "certainly",
})
_ARG_CONCLUSION_WORDS = frozenset({
    "in", "conclusion", "overall", "therefore", "thus", "hence",
    "finally", "summarize", "summary", "conclude", "concluding",
    "ultimately", "altogether", "briefly",
})


class DocumentState:
    """
    Evolving integer state variables for discourse coherence.

    The discrete analog of a transformer's residual stream.
    Each variable captures a different aspect of "what has happened so far"
    and "what should come next".  They span the ENTIRE document, not just
    the n-gram window.

    State variables:
      topic (1-16):     Current discourse topic (from TopicAssigner)
      mode (1-8):       Discourse mode: NARRATIVE, LIST, ARGUMENT, QUESTION,
                        DESCRIPTION, INSTRUCTION, COMPARISON, SUMMARY
      entity (1-64):    Currently focused entity (hash of recent proper nouns)
      tense (1-4):      PAST, PRESENT, FUTURE, INFINITIVE
      negation (1-3):   AFFIRMATIVE, NEGATED, SCOPED_NEGATION
      specificity (1-4): ABSTRACT, SPECIFIC, NUMERIC, QUOTED
      argument_pos (1-6): PREMISE, CLAIM, EVIDENCE, COUNTER, REBUTTAL, CONCLUSION

    ALL updates are deterministic rules based on the current word.
    ALL state variables are integers.  ZERO floating-point.
    """

    # Mode constants
    MODE_NARRATIVE = 1
    MODE_LIST = 2
    MODE_ARGUMENT = 3
    MODE_QUESTION = 4
    MODE_DESCRIPTION = 5
    MODE_INSTRUCTION = 6
    MODE_COMPARISON = 7
    MODE_SUMMARY = 8

    # Tense constants
    TENSE_PAST = 1
    TENSE_PRESENT = 2
    TENSE_FUTURE = 3
    TENSE_INFINITIVE = 4

    # Negation constants
    NEG_AFFIRMATIVE = 1
    NEG_NEGATED = 2
    NEG_SCOPED = 3

    # Specificity constants
    SPEC_ABSTRACT = 1
    SPEC_SPECIFIC = 2
    SPEC_NUMERIC = 3
    SPEC_QUOTED = 4

    # Argument position constants
    ARG_PREMISE = 1
    ARG_CLAIM = 2
    ARG_EVIDENCE = 3
    ARG_COUNTER = 4
    ARG_REBUTTAL = 5
    ARG_CONCLUSION = 6

    def __init__(
        self,
        vocab_size: int,
        n_topics: int = 16,
        pos_system=None,
        word_topics=None,
    ):
        """
        Initialize DocumentState.

        Args:
            vocab_size: Size of the vocabulary.
            n_topics: Number of topics (from TopicAssigner).
            pos_system: Optional POSTypeSystem instance for POS lookups.
            word_topics: Optional (vocab_size,) int8 array of per-word topic
                         assignments (from TopicAssigner.word_topics).
        """
        self.vocab_size = vocab_size
        self.n_topics = n_topics
        self.pos_system = pos_system
        self.word_topics = word_topics

        # State variables — set by reset()
        self.topic: int = 1
        self.mode: int = self.MODE_NARRATIVE
        self.entity: int = 1
        self.tense: int = self.TENSE_PRESENT
        self.negation: int = self.NEG_AFFIRMATIVE
        self.specificity: int = self.SPEC_ABSTRACT
        self.argument_pos: int = self.ARG_PREMISE

        # Negation scope tracking
        self._negated_word_count: int = 0  # words since last negation trigger
        self._scoped_word_count: int = 0   # words in scoped negation

        # Compatibility tables — built during training, None until then
        self.topic_word_counts: Optional[np.ndarray] = None      # (n_topics+1, vocab_size) int64
        self.mode_word_counts: Optional[np.ndarray] = None       # (9, vocab_size) int64
        self.tense_word_counts: Optional[np.ndarray] = None      # (5, vocab_size) int64
        self.negation_word_counts: Optional[np.ndarray] = None   # (4, vocab_size) int64
        self.specificity_word_counts: Optional[np.ndarray] = None  # (5, vocab_size) int64
        self.argument_word_counts: Optional[np.ndarray] = None   # (7, vocab_size) int64

        # v18.2: Pairwise compatibility tables for factorial state coupling
        # 5 coupled pairs: (var_i_name, var_j_name, shape_i, shape_j)
        self.coupling_pairs = [
            ("topic", "mode",       self.n_topics + 1, 9),    # 17×9
            ("topic", "tense",      self.n_topics + 1, 5),    # 17×5
            ("mode",  "tense",      9, 5),                    # 9×5
            ("mode",  "argument_pos", 9, 7),                  # 9×7
            ("tense", "negation",   5, 4),                     # 5×4
        ]

        # Pairwise co-occurrence counts: compat_{i,j}[val_i, val_j]
        self.pair_compat_tables: Optional[Dict[str, np.ndarray]] = None

        # Mean-field parameters
        self.mf_iterations: int = 5       # number of mean-field iterations
        self.mf_lambda_q15: int = 16384   # coupling strength in Q15 (0.5)
        self._coupling_built = False

        self._built = False

    # ===================================================================
    # BUILD: Create compatibility tables from training data
    # ===================================================================

    def build(
        self,
        sequences: List[List[int]],
        word_pos_tags: Optional[List[List[str]]] = None,
        idx2word: Optional[Dict[int, str]] = None,
    ) -> "DocumentState":
        """
        Build state-word compatibility tables from training data.

        For each position in each sequence, we:
        1. Record what the state WAS before that word (using deterministic rules)
        2. Increment: counts_table[state_val, word_id] += 1

        This gives us θ[state_var, state_val, word_id] = count tables
        reflecting actual state-word co-occurrences from training data.

        Args:
            sequences: List of integer token-id sequences.
            word_pos_tags: Optional parallel list of POS tag sequences.
                           If None, rule-based POS assignment is used.
            idx2word: Mapping from word ID to word string. REQUIRED for
                      state update rules to fire correctly. If not provided,
                      falls back to pos_system.idx2word (if available).

        Returns:
            self
        """
        V = self.vocab_size

        # Allocate compatibility tables
        # +1 on first dim to allow index 0 = "unspecified/default"
        self.topic_word_counts = np.zeros((self.n_topics + 1, V), dtype=np.int64)
        self.mode_word_counts = np.zeros((9, V), dtype=np.int64)        # 0=unspecified, 1-8
        self.tense_word_counts = np.zeros((5, V), dtype=np.int64)       # 0=unspecified, 1-4
        self.negation_word_counts = np.zeros((4, V), dtype=np.int64)    # 0=unspecified, 1-3
        self.specificity_word_counts = np.zeros((5, V), dtype=np.int64) # 0=unspecified, 1-4
        self.argument_word_counts = np.zeros((7, V), dtype=np.int64)    # 0=unspecified, 1-6

        n_seqs = len(sequences)
        total_positions = 0

        for seq_idx, seq in enumerate(sequences):
            # Reset state for each new document/sequence
            self.reset()

            pos_tags = None
            if word_pos_tags is not None and seq_idx < len(word_pos_tags):
                pos_tags = word_pos_tags[seq_idx]

            for pos_i, word_id in enumerate(seq):
                if word_id < 0 or word_id >= V:
                    continue

                # Get word string for rule evaluation
                word_str = None
                pos_tag = None
                # Priority: (1) idx2word parameter, (2) pos_system.idx2word
                if idx2word is not None:
                    word_str = idx2word.get(word_id)
                elif self.pos_system is not None and hasattr(self.pos_system, 'idx2word'):
                    word_str = self.pos_system.idx2word.get(word_id)
                if pos_tags is not None and pos_i < len(pos_tags):
                    pos_tag = pos_tags[pos_i]

                # --- Record current state BEFORE updating ---
                topic_val = self.topic
                mode_val = self.mode
                tense_val = self.tense
                neg_val = self.negation
                spec_val = self.specificity
                arg_val = self.argument_pos

                # Increment compatibility tables
                self.topic_word_counts[topic_val, word_id] += 1
                self.mode_word_counts[mode_val, word_id] += 1
                self.tense_word_counts[tense_val, word_id] += 1
                self.negation_word_counts[neg_val, word_id] += 1
                self.specificity_word_counts[spec_val, word_id] += 1
                self.argument_word_counts[arg_val, word_id] += 1

                # --- Update state for next word ---
                self.update(
                    word_id,
                    word_str=word_str,
                    pos_tag=pos_tag,
                )

                total_positions += 1

        self._built = True
        print(f"  DocumentState.build(): {n_seqs} sequences, {total_positions} positions")
        self._print_table_stats()

        return self

    # ===================================================================
    # BUILD COUPLING: Pairwise compatibility tables (v18.2)
    # ===================================================================

    def build_coupling(
        self,
        sequences: List[List[int]],
        idx2word: Optional[Dict[int, str]] = None,
        mf_iterations: int = 5,
        mf_lambda_q15: int = 16384,
    ) -> "DocumentState":
        """
        Build pairwise compatibility tables from training data (v18.2).

        For each coupled pair of state variables (i, j), we count how
        often each (val_i, val_j) combination occurs in training data.
        The resulting compatibility tables capture correlations like:
          - topic=SCIENCE often co-occurs with mode=DESCRIPTION
          - mode=ARGUMENT often co-occurs with tense=PAST
          - negation=NEGATED rarely co-occurs with mode=LIST

        The compatibility table stores log-likelihood ratios:
          compat[i,j][val_i, val_j] = log2(joint_count / expected_if_independent)

        where expected_if_independent = (marginal_i * marginal_j) / total.
        Positive values mean the pair co-occurs MORE than expected
        (compatible), negative means LESS than expected (incompatible).

        All integer, using int_log2_fine() for log2 computation.

        Args:
            sequences: List of integer token-id sequences.
            idx2word: Mapping from word ID to word string.
            mf_iterations: Number of mean-field iterations (default 5).
            mf_lambda_q15: Coupling strength in Q15 (default 16384 ≈ 0.5).

        Returns:
            self
        """
        self.mf_iterations = mf_iterations
        self.mf_lambda_q15 = mf_lambda_q15

        # State variable ranges (index 0 = unspecified/default)
        var_ranges = {
            "topic": self.n_topics + 1,        # 0..16
            "mode": 9,                         # 0..8
            "tense": 5,                        # 0..4
            "negation": 4,                     # 0..3
            "specificity": 5,                  # 0..4
            "argument_pos": 7,                 # 0..6
        }

        # Initialize pairwise count tables
        pair_counts = {}
        pair_marginals_i = {}
        pair_marginals_j = {}

        for pair_name, (var_i, var_j, shape_i, shape_j) in self._iter_coupling_pairs():
            pair_counts[pair_name] = np.zeros((shape_i, shape_j), dtype=np.int64)
            pair_marginals_i[pair_name] = np.zeros(shape_i, dtype=np.int64)
            pair_marginals_j[pair_name] = np.zeros(shape_j, dtype=np.int64)

        # Count co-occurrences from training data
        total_positions = 0

        for seq in sequences:
            self.reset()

            for pos_i, word_id in enumerate(seq):
                if word_id < 0 or word_id >= self.vocab_size:
                    continue

                # Get word string
                word_str = None
                if idx2word is not None:
                    word_str = idx2word.get(word_id)
                elif self.pos_system is not None and hasattr(self.pos_system, 'idx2word'):
                    word_str = self.pos_system.idx2word.get(word_id)

                # Record current state BEFORE update
                state_snapshot = self.get_state_vector()

                # Count pairwise co-occurrences
                for pair_name, (var_i, var_j, _, _) in self._iter_coupling_pairs():
                    val_i = state_snapshot[var_i]
                    val_j = state_snapshot[var_j]
                    pair_counts[pair_name][val_i, val_j] += 1
                    pair_marginals_i[pair_name][val_i] += 1
                    pair_marginals_j[pair_name][val_j] += 1

                # Update state
                self.update(word_id, word_str=word_str)
                total_positions += 1

        # Build compatibility tables from counts
        # compat[i,j][val_i, val_j] = log2(observed / expected) * scale
        # where expected = marg_i * marg_j / total
        # Positive = co-occurs more than expected (compatible)
        # Negative = co-occurs less than expected (incompatible)
        COMPAT_SCALE = 64  # Q6 scaling for compatibility values

        self.pair_compat_tables = {}

        for pair_name, (var_i, var_j, shape_i, shape_j) in self._iter_coupling_pairs():
            counts = pair_counts[pair_name]
            marg_i = pair_marginals_i[pair_name]
            marg_j = pair_marginals_j[pair_name]
            total = int(counts.sum())

            compat = np.zeros((shape_i, shape_j), dtype=np.int16)

            if total > 0:
                for vi in range(shape_i):
                    for vj in range(shape_j):
                        observed = int(counts[vi, vj])
                        if observed == 0:
                            # Unobserved pair: negative compatibility (penalty)
                            compat[vi, vj] = -COMPAT_SCALE * 5
                            continue

                        mi = int(marg_i[vi])
                        mj = int(marg_j[vj])
                        if mi == 0 or mj == 0:
                            continue

                        # Expected count if independent
                        expected = (mi * mj) // total

                        if expected == 0:
                            expected = 1

                        # Log-likelihood ratio: log2(observed / expected)
                        if observed >= expected:
                            # More than expected: positive compat
                            ratio = observed // max(1, expected)
                            if ratio >= 2:
                                log2_ratio = int_log2_fine(ratio)
                                compat[vi, vj] = min(
                                    (log2_ratio * COMPAT_SCALE) >> 8,
                                    32767
                                )
                            else:
                                compat[vi, vj] = 0
                        else:
                            # Less than expected: negative compat
                            ratio = expected // max(1, observed)
                            if ratio >= 2:
                                log2_ratio = int_log2_fine(ratio)
                                compat[vi, vj] = max(
                                    -((log2_ratio * COMPAT_SCALE) >> 8),
                                    -32768
                                )
                            else:
                                compat[vi, vj] = 0

            self.pair_compat_tables[pair_name] = compat

        self._coupling_built = True

        print(f"  DocumentState.build_coupling(): {len(sequences)} sequences, "
              f"{total_positions} positions, {len(self.pair_compat_tables)} pairs")
        for pair_name, compat in self.pair_compat_tables.items():
            compat_range = f"[{int(compat.min())}, {int(compat.max())}]"
            print(f"    {pair_name}: shape={compat.shape}, range={compat_range}")

        return self

    def _iter_coupling_pairs(self):
        """
        Iterate over coupling pairs, yielding (pair_name, (var_i, var_j, shape_i, shape_j)).
        """
        for var_i, var_j, shape_i, shape_j in self.coupling_pairs:
            pair_name = f"{var_i}_x_{var_j}"
            yield pair_name, (var_i, var_j, shape_i, shape_j)

    # ===================================================================
    # MEAN-FIELD INFERENCE (v18.2)
    # ===================================================================

    def run_mean_field(self) -> None:
        """
        Run mean-field inference to refine state variables via coupling.

        For each iteration:
          For each state variable i:
            Compute field from all coupled variables:
              field_i(val) = sum_{j ~ i} lambda * compat_{i,j}[val, val_j]
            Set val_i = argmax(field_i) over all possible values

        This iteratively adjusts state values to be more consistent with
        each other. For example, if topic=SCIENCE and mode=ARGUMENT are
        rarely compatible, the coupling will push one of them to a more
        consistent value.

        The coupling strength lambda_q15 controls how strongly the
        coupling influences the state. lambda=0 means no coupling
        (independent variables), lambda=0.5 means moderate coupling.

        All integer. The field computation is O(K) per variable where
        K is the number of coupled pairs (5 here), and the argmax is
        over the range of the variable (max 17 for topic).
        Total: ~5 iterations × 6 variables × 5 pairs × 17 values = ~2550 ops.
        """
        if not self._coupling_built or self.pair_compat_tables is None:
            return

        # State variable ranges (for argmax)
        var_info = {
            "topic":       (1, self.n_topics + 1),   # 1..16
            "mode":        (1, 9),                   # 1..8
            "tense":       (1, 5),                   # 1..4
            "negation":    (1, 4),                   # 1..3
            "specificity": (1, 5),                   # 1..4
            "argument_pos": (1, 7),                  # 1..6
        }

        lambda_q15 = self.mf_lambda_q15

        for _iteration in range(self.mf_iterations):
            # For each variable, compute field from coupled variables
            for var_name, (val_min, val_max) in var_info.items():
                if var_name == "entity":
                    continue  # Entity has 64 values, skip coupling

                current_val = getattr(self, var_name)

                # Compute field for each possible value
                best_val = current_val
                best_field = -2**30  # very negative

                for candidate_val in range(val_min, val_max):
                    field = 0

                    # Sum contributions from all coupled pairs
                    for pair_name, (var_i, var_j, _, _) in self._iter_coupling_pairs():
                        if var_i == var_name:
                            # This variable is the first in the pair
                            val_j = getattr(self, var_j)
                            compat = self.pair_compat_tables[pair_name]
                            if 0 <= candidate_val < compat.shape[0] and 0 <= val_j < compat.shape[1]:
                                field += (lambda_q15 * int(compat[candidate_val, val_j])) >> 15
                        elif var_j == var_name:
                            # This variable is the second in the pair
                            val_i = getattr(self, var_i)
                            compat = self.pair_compat_tables[pair_name]
                            if 0 <= val_i < compat.shape[0] and 0 <= candidate_val < compat.shape[1]:
                                field += (lambda_q15 * int(compat[val_i, candidate_val])) >> 15

                    if field > best_field:
                        best_field = field
                        best_val = candidate_val

                # Update state variable to best value
                # Only update if the coupling suggests a different value
                # (keep the deterministic rule result as the default)
                if best_val != current_val:
                    setattr(self, var_name, best_val)

    # ===================================================================
    # COUPLING ENERGY (v18.2)
    # ===================================================================

    def compute_coupling_energy(
        self,
        coupling_scale: int = 200,
    ) -> int:
        """
        Compute coupling energy for the current state configuration.

        E_coupling = -sum_{pairs (i,j)} lambda * compat_table[val_i, val_j]

        The coupling energy penalizes unlikely state combinations.
        If the current state values are compatible (positive compat),
        the energy is negative (preferred). If incompatible (negative
        compat), the energy is positive (penalized).

        This is a SCALAR energy, not per-candidate. It affects the
        overall energy baseline but doesn't differentiate between
        candidate words directly. Its primary role is to ensure the
        mean-field inference produces consistent state values, which
        then influence word selection through the state-word
        compatibility tables.

        Args:
            coupling_scale: Scaling factor for coupling energy (default 200).

        Returns:
            Integer coupling energy (int64). Negative = compatible state.
        """
        if not self._coupling_built or self.pair_compat_tables is None:
            return 0

        total_compat = 0

        for pair_name, (var_i, var_j, _, _) in self._iter_coupling_pairs():
            val_i = getattr(self, var_i)
            val_j = getattr(self, var_j)
            compat = self.pair_compat_tables[pair_name]

            if 0 <= val_i < compat.shape[0] and 0 <= val_j < compat.shape[1]:
                total_compat += int(compat[val_i, val_j])

        # E_coupling = -lambda * total_compat * coupling_scale / Q15
        # With lambda in Q15, the division brings it back to integer scale
        energy = -(self.mf_lambda_q15 * total_compat * coupling_scale) >> 15

        return energy

    def _print_table_stats(self):
        """Print statistics about the compatibility tables."""
        tables = {
            "topic": self.topic_word_counts,
            "mode": self.mode_word_counts,
            "tense": self.tense_word_counts,
            "negation": self.negation_word_counts,
            "specificity": self.specificity_word_counts,
            "argument": self.argument_word_counts,
        }
        for name, table in tables.items():
            if table is not None:
                nonzero_rows = int((table.sum(axis=1) > 0).sum())
                total = int(table.sum())
                print(f"    {name}: shape={table.shape}, "
                      f"nonzero_rows={nonzero_rows}, total={total}")

    # ===================================================================
    # RESET: Initialize state for a new document
    # ===================================================================

    def reset(self):
        """Reset all state variables to defaults for a new document."""
        self.topic = 1
        self.mode = self.MODE_NARRATIVE
        self.entity = 1
        self.tense = self.TENSE_PRESENT
        self.negation = self.NEG_AFFIRMATIVE
        self.specificity = self.SPEC_ABSTRACT
        self.argument_pos = self.ARG_PREMISE
        self._negated_word_count = 0
        self._scoped_word_count = 0

        # v18.2: Also reset mean-field coupling (no coupling needed at start)
        # Mean-field will run after each word update if coupling is built

    # ===================================================================
    # UPDATE: Deterministic state transitions
    # ===================================================================

    def update(
        self,
        word_id: int,
        word_str: Optional[str] = None,
        pos_tag: Optional[str] = None,
        topic_id: Optional[int] = None,
    ):
        """
        Update all state variables based on the current word.

        ALL updates are deterministic rules.  ALL state variables are integers.

        Args:
            word_id: Integer token ID of the current word.
            word_str: Optional string form of the word for rule evaluation.
            pos_tag: Optional POS tag string.
            topic_id: Optional explicit topic assignment.
        """
        # Resolve word string for rule evaluation
        w = word_str.lower() if word_str else ""

        # --- Topic update ---
        self._update_topic(word_id, w, topic_id)

        # --- Mode update ---
        self._update_mode(w, pos_tag)

        # --- Entity update ---
        self._update_entity(word_str, pos_tag)

        # --- Tense update ---
        self._update_tense(w, pos_tag)

        # --- Negation update ---
        self._update_negation(w)

        # --- Specificity update ---
        self._update_specificity(w, word_str, pos_tag)

        # --- Argument position update ---
        self._update_argument(w)

    def _update_topic(
        self,
        word_id: int,
        w: str,
        topic_id: Optional[int],
    ):
        """Update topic state variable."""
        if topic_id is not None and 1 <= topic_id <= self.n_topics:
            self.topic = topic_id
        elif self.word_topics is not None and 0 <= word_id < len(self.word_topics):
            wt = int(self.word_topics[word_id])
            if 1 <= wt <= self.n_topics:
                self.topic = wt
        # Otherwise: keep current topic (inertia)

    def _update_mode(self, w: str, pos_tag: Optional[str]):
        """Update mode state variable based on discourse markers."""
        if not w:
            return

        # Check triggers in priority order
        if w in _MODE_ARGUMENT_WORDS:
            self.mode = self.MODE_ARGUMENT
            return

        if w in _MODE_LIST_WORDS:
            self.mode = self.MODE_LIST
            return

        if w in _MODE_QUESTION_WORDS:
            self.mode = self.MODE_QUESTION
            return

        if w in _MODE_COMPARISON_WORDS:
            self.mode = self.MODE_COMPARISON
            return

        if w in _MODE_SUMMARY_WORDS:
            self.mode = self.MODE_SUMMARY
            return

        if w in _MODE_INSTRUCTION_WORDS:
            self.mode = self.MODE_INSTRUCTION
            return

        if w in _MODE_DESCRIPTION_PREP_WORDS:
            self.mode = self.MODE_DESCRIPTION
            return

        # "for example", "such as" are multi-word; check single words
        if w in _MODE_DESCRIPTION_WORDS:
            self.mode = self.MODE_DESCRIPTION
            return

        # Past-tense verbs → NARRATIVE
        if self._is_past_tense_verb(w, pos_tag):
            self.mode = self.MODE_NARRATIVE
            return

        # Default: keep current mode (inertia)

    def _update_entity(self, word_str: Optional[str], pos_tag: Optional[str]):
        """Update entity state variable — hash of proper nouns."""
        if word_str is None:
            return

        # Detect proper noun: capitalized word (not at sentence start)
        # or explicit POS tag
        is_proper = False

        if pos_tag is not None and pos_tag in ("NNP", "NNPS"):
            is_proper = True
        elif word_str and word_str[0].isupper() and len(word_str) > 1:
            # Check it's not just a sentence-initial word by looking at
            # common sentence starters vs. actual names
            # Heuristic: if it's uppercase and not a common sentence starter
            sentence_starters = {
                "The", "A", "An", "This", "That", "These", "Those",
                "It", "He", "She", "We", "They", "There", "Here",
                "When", "Where", "How", "Why", "What", "Which",
                "If", "But", "And", "Or", "So", "Yet", "For",
                "In", "On", "At", "To", "From", "With", "By",
                "As", "Not", "No", "Each", "Every", "All",
                "Both", "Neither", "Either", "Some", "Any",
                "My", "Your", "His", "Her", "Its", "Our", "Their",
                "Is", "Are", "Was", "Were", "Has", "Have", "Had",
                "Do", "Does", "Did", "Can", "Could", "Will", "Would",
                "Should", "May", "Might", "Must", "Shall",
            }
            if word_str not in sentence_starters:
                is_proper = True

        if is_proper:
            # Hash word to 1-64 range using simple integer hash
            h = 0
            for ch in word_str:
                h = (h * 31 + ord(ch)) & 0xFFFFFFFF
            self.entity = (h % 64) + 1

        # Otherwise: keep current entity (inertia)

    def _update_tense(self, w: str, pos_tag: Optional[str]):
        """Update tense state variable."""
        if not w:
            return

        # "will", "shall", "going" → FUTURE
        # (Checked BEFORE modal verbs since "will"/"shall" are both)
        if w in _FUTURE_WORDS:
            self.tense = self.TENSE_FUTURE
            return

        # Modal verbs that don't indicate tense keep current tense
        if w in _MODAL_VERBS:
            return

        # Words ending in "ing" → PRESENT (continuous)
        if w.endswith("ing") and len(w) > 4:
            self.tense = self.TENSE_PRESENT
            return

        # Words ending in "ed" (not false positives) → PAST
        if self._is_past_tense_verb(w, pos_tag):
            self.tense = self.TENSE_PAST
            return

        # "to" followed by verb → INFINITIVE
        # We mark "to" as potentially infinitive; actual verb check
        # happens at the next word during generation
        if w == "to":
            self.tense = self.TENSE_INFINITIVE
            return

        # Default: keep current tense (inertia)

    def _update_negation(self, w: str):
        """Update negation state variable with scope tracking."""
        if not w:
            return

        # Negation triggers → NEGATED
        if w in _NEGATION_WORDS:
            self.negation = self.NEG_NEGATED
            self._negated_word_count = 0
            self._scoped_word_count = 0
            return

        # State machine for negation scope
        if self.negation == self.NEG_NEGATED:
            self._negated_word_count += 1
            # After 1 content word following negation → SCOPED
            if self._negated_word_count >= 1:
                self.negation = self.NEG_SCOPED
                self._scoped_word_count = 0
        elif self.negation == self.NEG_SCOPED:
            self._scoped_word_count += 1
            # After 2+ words in SCOPED → AFFIRMATIVE (scope expires)
            if self._scoped_word_count >= 2:
                self.negation = self.NEG_AFFIRMATIVE
                self._negated_word_count = 0
                self._scoped_word_count = 0

    def _update_specificity(
        self,
        w: str,
        word_str: Optional[str],
        pos_tag: Optional[str],
    ):
        """Update specificity state variable."""
        if not w and word_str is None:
            return

        # Numbers, dates → NUMERIC
        if w and any(c in _NUMERIC_CHARS for c in w):
            cleaned = w.replace(".", "").replace(",", "").replace("-", "")
            if cleaned.isdigit():
                self.specificity = self.SPEC_NUMERIC
                return

        # Quoted text markers → QUOTED
        if word_str and word_str in _QUOTE_MARKS:
            self.specificity = self.SPEC_QUOTED
            return
        if w in ('"', "'", "``", "''"):
            self.specificity = self.SPEC_QUOTED
            return

        # Proper nouns, specific names → SPECIFIC
        if pos_tag is not None and pos_tag in ("NNP", "NNPS"):
            self.specificity = self.SPEC_SPECIFIC
            return
        if word_str and len(word_str) > 1 and word_str[0].isupper():
            # Potential proper noun
            sentence_starters_lower = {
                "the", "a", "an", "this", "that", "it", "he", "she",
                "we", "they", "there", "here", "when", "where", "how",
                "why", "what", "which", "if", "but", "and", "or", "so",
            }
            if w not in sentence_starters_lower:
                self.specificity = self.SPEC_SPECIFIC
                return

        # Abstract nouns, general terms → ABSTRACT
        # Heuristic: words ending in typical abstract suffixes
        if w and len(w) > 4:
            abstract_suffixes = (
                "tion", "sion", "ment", "ness", "ity", "ism",
                "ence", "ance", "dom", "ship", "hood",
            )
            for suffix in abstract_suffixes:
                if w.endswith(suffix):
                    self.specificity = self.SPEC_ABSTRACT
                    return

        # Default: keep current specificity (inertia)

    def _update_argument(self, w: str):
        """Update argument position state variable."""
        if not w:
            return

        if w in _ARG_PREMISE_WORDS:
            self.argument_pos = self.ARG_PREMISE
            return

        if w in _ARG_CLAIM_WORDS:
            self.argument_pos = self.ARG_CLAIM
            return

        if w in _ARG_EVIDENCE_WORDS:
            self.argument_pos = self.ARG_EVIDENCE
            return

        if w in _ARG_COUNTER_WORDS:
            self.argument_pos = self.ARG_COUNTER
            return

        if w in _ARG_REBUTTAL_WORDS:
            self.argument_pos = self.ARG_REBUTTAL
            return

        if w in _ARG_CONCLUSION_WORDS:
            self.argument_pos = self.ARG_CONCLUSION
            return

        # Default: keep current argument position (inertia)

    # ===================================================================
    # HELPERS
    # ===================================================================

    def _is_past_tense_verb(self, w: str, pos_tag: Optional[str]) -> bool:
        """
        Check if word is likely a past-tense verb.
        Uses POS tag if available, otherwise morphological heuristics.
        """
        if pos_tag is not None and pos_tag == "VBD":
            return True

        if not w or len(w) < 3:
            return False

        if w.endswith("ed"):
            # Exclude false positives: words that naturally end in "ed"
            # like "red", "bed", "fed", "shed", etc.
            if w in _PAST_TENSE_ENDINGS_EXCLUSIONS:
                return False
            # "ed" suffix with a consonant before it is likely past tense
            if len(w) >= 3 and w[-3] not in "aeiou":
                return True
            # "-ied" pattern (e.g. "carried", "studied")
            if w.endswith("ied") and len(w) > 4:
                return True
            # Short words ending in "ed" that are likely past tense
            # e.g. "used", "made" (not ending in ed), but "aged", "tired"
            if len(w) >= 4:
                return True

        return False

    # ===================================================================
    # ENERGY: State-conditioned energy computation
    # ===================================================================

    def compute_energy(
        self,
        candidate_words: np.ndarray,
        state_scale: int = 200,
    ) -> np.ndarray:
        """
        Compute state-conditioned energy for candidate words.

        For each state variable, looks up how often each candidate word
        co-occurred with the current state value in training data.
        Words that are compatible with the current state get lower energy
        (preferred); incompatible words get higher energy (penalized).

        Uses integer log2 ratio:
            E_state(w) = log2(total / count) * state_scale  if count > 0
            E_state(w) = max_state_energy                    if count == 0

        This is equivalent to -log2(P(w|state)) up to a constant.

        Args:
            candidate_words: 1-D int array of candidate word IDs.
            state_scale: Integer scaling factor for state energy.

        Returns:
            energies: 1-D int64 array of state energy for each candidate.
        """
        n_candidates = len(candidate_words)
        energies = np.zeros(n_candidates, dtype=np.int64)

        if not self._built:
            return energies

        max_state_energy = state_scale * 20  # penalty for unseen state-word pairs

        # Collect (current_value, counts_table) pairs
        state_vars = [
            (self.topic, self.topic_word_counts),
            (self.mode, self.mode_word_counts),
            (self.tense, self.tense_word_counts),
            (self.negation, self.negation_word_counts),
            (self.specificity, self.specificity_word_counts),
            (self.argument_pos, self.argument_word_counts),
        ]

        for current_val, counts_table in state_vars:
            if counts_table is None:
                continue

            n_rows = counts_table.shape[0]
            if current_val < 0 or current_val >= n_rows:
                continue

            row = counts_table[current_val]  # (vocab_size,) int64
            total = int(row.sum())
            if total <= 0:
                continue

            # Vectorized lookup for all candidates at once
            counts = row[candidate_words]  # fancy indexing → (n_candidates,) int64

            # Compute energy: log2(total / max(count, 1)) * state_scale
            # For count == 0: max penalty
            # For count > 0 and count < total: log2 ratio
            # For count == total: energy = 0 (only word in this state)

            safe_counts = np.maximum(counts, np.int64(1))

            with np.errstate(divide='ignore', invalid='ignore'):
                ratios = total // safe_counts  # integer division

            # Compute energy using int_log2_fine for each ratio
            # int_log2_fine returns log2(x) * 256
            # Energy = (int_log2_fine(ratio) * state_scale) >> 8
            for i in range(n_candidates):
                c = int(counts[i])
                if c == 0:
                    energies[i] += max_state_energy
                elif c >= total:
                    # Word dominates this state value — no penalty
                    pass
                else:
                    ratio = int(ratios[i])
                    if ratio >= 2:
                        log2_ratio = int_log2_fine(ratio)
                        energy = (log2_ratio * state_scale) >> 8
                        energies[i] += energy
                    # ratio < 2 means count > total/2 — word is very
                    # compatible, minimal energy

        return energies

    def compute_energy_vectorized(
        self,
        candidate_words: np.ndarray,
        state_scale: int = 200,
    ) -> np.ndarray:
        """
        Fully vectorized state-conditioned energy computation.

        Same logic as compute_energy but avoids the per-candidate Python
        loop for int_log2_fine by pre-computing a log2 LUT for the
        encountered ratios.

        Args:
            candidate_words: 1-D int array of candidate word IDs.
            state_scale: Integer scaling factor for state energy.

        Returns:
            energies: 1-D int64 array of state energy for each candidate.
        """
        n_candidates = len(candidate_words)
        energies = np.zeros(n_candidates, dtype=np.int64)

        if not self._built:
            return energies

        max_state_energy = state_scale * 20

        state_vars = [
            (self.topic, self.topic_word_counts),
            (self.mode, self.mode_word_counts),
            (self.tense, self.tense_word_counts),
            (self.negation, self.negation_word_counts),
            (self.specificity, self.specificity_word_counts),
            (self.argument_pos, self.argument_word_counts),
        ]

        for current_val, counts_table in state_vars:
            if counts_table is None:
                continue

            n_rows = counts_table.shape[0]
            if current_val < 0 or current_val >= n_rows:
                continue

            row = counts_table[current_val]
            total = int(row.sum())
            if total <= 0:
                continue

            counts = row[candidate_words].astype(np.int64)

            # Mask for unseen words
            unseen_mask = counts == 0
            # Mask for dominant words (count >= total)
            dominant_mask = (counts > 0) & (counts >= total)
            # Mask for normal words (0 < count < total)
            normal_mask = (counts > 0) & (counts < total)

            # Apply max penalty for unseen
            energies[unseen_mask] += max_state_energy

            # Compute log2 ratios for normal words
            normal_indices = np.where(normal_mask)[0]
            if len(normal_indices) > 0:
                normal_counts = counts[normal_indices]
                ratios = total // np.maximum(normal_counts, np.int64(1))

                # Vectorized int_log2_fine via LUT
                # Build a temporary LUT for the unique ratios
                unique_ratios = np.unique(ratios)
                log2_lut = {}
                for r in unique_ratios:
                    r_int = int(r)
                    if r_int >= 2:
                        log2_lut[r_int] = (int_log2_fine(r_int) * state_scale) >> 8
                    else:
                        log2_lut[r_int] = 0

                # Apply energies
                for i, idx in enumerate(normal_indices):
                    r_int = int(ratios[i])
                    energies[idx] += log2_lut.get(r_int, 0)

            # Dominant words: energy += 0 (no penalty)

        return energies

    # ===================================================================
    # QUERY: Get current state as dict
    # ===================================================================

    def get_state_vector(self) -> Dict[str, int]:
        """
        Return current state as a dict of variable name → integer value.

        Returns:
            Dict with keys: topic, mode, entity, tense, negation,
            specificity, argument_pos
        """
        return {
            "topic": self.topic,
            "mode": self.mode,
            "entity": self.entity,
            "tense": self.tense,
            "negation": self.negation,
            "specificity": self.specificity,
            "argument_pos": self.argument_pos,
        }

    def __repr__(self) -> str:
        sv = self.get_state_vector()
        mode_names = {
            1: "NARRATIVE", 2: "LIST", 3: "ARGUMENT", 4: "QUESTION",
            5: "DESCRIPTION", 6: "INSTRUCTION", 7: "COMPARISON", 8: "SUMMARY",
        }
        tense_names = {1: "PAST", 2: "PRESENT", 3: "FUTURE", 4: "INFINITIVE"}
        neg_names = {1: "AFFIRMATIVE", 2: "NEGATED", 3: "SCOPED"}
        spec_names = {1: "ABSTRACT", 2: "SPECIFIC", 3: "NUMERIC", 4: "QUOTED"}
        arg_names = {
            1: "PREMISE", 2: "CLAIM", 3: "EVIDENCE", 4: "COUNTER",
            5: "REBUTTAL", 6: "CONCLUSION",
        }
        return (
            f"DocumentState("
            f"topic={sv['topic']}, "
            f"mode={mode_names.get(sv['mode'], '?')}, "
            f"entity={sv['entity']}, "
            f"tense={tense_names.get(sv['tense'], '?')}, "
            f"neg={neg_names.get(sv['negation'], '?')}, "
            f"spec={spec_names.get(sv['specificity'], '?')}, "
            f"arg={arg_names.get(sv['argument_pos'], '?')})"
        )
