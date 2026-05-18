"""
POS Type System for the Coupled Ising-Potts Language Model.

Implements the type-value decomposition:
  - Type layer (Potts): sigma_i in {1,...,T} — POS tag / syntactic category
  - Value layer: w_i in {1,...,V} — specific word from vocabulary

The coupling structure follows Haydarov, Omirov & Rozikov (arXiv:2502.12014):
  H(s, sigma) = -J sum_{<x,y>} s(x)*s(y) * delta_{sigma(x), sigma(y)}

where the Ising interaction is active only when Potts states agree.
This implements syntactically-gated co-occurrence.

Additional grammar penalties follow Marcolli's implicational coupling:
  H_V = sum_ell J_ell * delta_{X_ell,p1, Y_ell,p2}

All operations are integer-only in the generation loop.
"""

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple
import json
import numpy as np


# Coarse POS tags (reduced from Penn Treebank for tractability)
COARSE_POS_TAGS = [
    "NOUN",     # nouns (NN, NNS, NNP, NNPS)
    "VERB",     # verbs (VB, VBD, VBG, VBN, VBP, VBZ)
    "ADJ",      # adjectives (JJ, JJR, JJS)
    "ADV",      # adverbs (RB, RBR, RBS)
    "DET",      # determiners (DT, WDT)
    "PREP",     # prepositions (IN)
    "PRON",     # pronouns (PRP, PRP$, WP, WP$)
    "AUX",      # auxiliaries / modals (MD)
    "CONJ",     # conjunctions (CC)
    "PART",     # particles (RP, TO)
    "NUM",      # numbers (CD)
    "PUNCT",    # punctuation
    "X",        # other / unknown
]

POS2IDX = {tag: i for i, tag in enumerate(COARSE_POS_TAGS)}
IDX2POS = {i: tag for i, tag in enumerate(COARSE_POS_TAGS)}
N_POS = len(COARSE_POS_TAGS)

# Sets of POS tags for grammar rule definitions
OPEN_CLASS = {"NOUN", "VERB", "ADJ", "ADV"}
CLOSED_CLASS = {"DET", "PREP", "PRON", "AUX", "CONJ", "PART"}
NOUN_LIKE = {"NOUN", "PRON", "NUM"}
VERB_LIKE = {"VERB", "AUX"}
MODIFIER = {"ADJ", "ADV"}


class POSTypeSystem:
    """
    Integer-only POS type system for the coupled Ising-Potts model.

    Components:
      - J_type[t, t', dist]: type-type coupling (POS bigram counts × distance)
      - I_emit[w, t]: emission weight (count of word w tagged as type t)
      - allowed_types[w]: set of types word w can have (from emission)
      - grammar_penalties: list of (condition, penalty) pairs for hard constraints
    """

    def __init__(self, vocab_size: int, n_types: int = N_POS, window: int = 5):
        self.vocab_size = vocab_size
        self.n_types = n_types
        self.window = window

        # Type-type coupling: J_type[t1, t2, dist] = integer count
        # Flattened to J_type[t1 * n_types + t2] = dict of dist->count
        self.J_type = np.zeros((n_types, n_types), dtype=np.int64)

        # Distance-weighted type couplings
        self.J_type_by_dist: Dict[int, Dict[Tuple[int, int], int]] = {}

        # Emission weights: I_emit[w, t] = count(word w tagged as type t)
        self.I_emit = np.zeros((vocab_size, n_types), dtype=np.int64)

        # Allowed types per word (derived from I_emit)
        self.allowed_types: Dict[int, Set[int]] = {}

        # Grammar penalties (list of constraint specs)
        self.grammar_penalties: List[Dict] = []

        # Precomputed type proposal distributions
        self.type_cumsum: Optional[np.ndarray] = None

    def assign_pos_rules(self, word: str, word_idx: int) -> List[int]:
        """
        Simple rule-based POS assignment for English words.
        Uses morphological and positional heuristics — no FP, no ML model.

        Returns list of possible POS tag indices for this word.
        """
        w = word.lower()
        tags = []

        # Punctuation
        if not any(c.isalnum() for c in w):
            tags.append(POS2IDX["PUNCT"])
            return tags if tags else [POS2IDX["X"]]

        # Numbers
        if w.replace(".", "").replace(",", "").replace("-", "").isdigit():
            tags.append(POS2IDX["NUM"])

        # Determiners
        if w in {"the", "a", "an", "this", "that", "these", "those",
                 "some", "any", "all", "each", "every", "no", "both",
                 "either", "neither", "my", "your", "his", "her", "its",
                 "our", "their"}:
            tags.append(POS2IDX["DET"])

        # Pronouns
        if w in {"i", "me", "you", "he", "him", "she", "her", "it",
                 "we", "us", "they", "them", "myself", "yourself",
                 "himself", "herself", "itself", "ourselves",
                 "themselves", "who", "whom", "which", "what", "that"}:
            tags.append(POS2IDX["PRON"])

        # Auxiliaries / Modals
        if w in {"is", "am", "are", "was", "were", "be", "been", "being",
                 "have", "has", "had", "having", "do", "does", "did",
                 "can", "could", "will", "would", "shall", "should",
                 "may", "might", "must"}:
            tags.append(POS2IDX["AUX"])

        # Conjunctions
        if w in {"and", "or", "but", "nor", "for", "yet", "so",
                 "although", "because", "since", "unless", "while",
                 "if", "then", "when", "where", "whether"}:
            tags.append(POS2IDX["CONJ"])

        # Particles
        if w in {"to", "not", "up", "down", "out", "off", "on",
                 "in", "away", "over"}:
            tags.append(POS2IDX["PART"])

        # Prepositions
        if w in {"of", "in", "to", "for", "with", "on", "at", "from",
                 "by", "about", "as", "into", "through", "during",
                 "before", "after", "above", "below", "between",
                 "under", "over", "against", "within", "without",
                 "among", "upon", "toward", "towards"}:
            tags.append(POS2IDX["PREP"])

        # Adjectives (morphological: -y, -ful, -less, -ous, -ive, -able, -al)
        if (w.endswith("ful") or w.endswith("less") or w.endswith("ous") or
            w.endswith("ive") or w.endswith("able") or w.endswith("ible") or
            w.endswith("al") or w.endswith("ial") or w.endswith("ent") or
            w.endswith("ant") or w.endswith("ic") or w.endswith("ical")):
            tags.append(POS2IDX["ADJ"])

        # Adverbs (morphological: -ly)
        if w.endswith("ly"):
            tags.append(POS2IDX["ADV"])

        # Verbs (morphological: -ing, -ed, -ize, -ify, -ate, -en)
        if (w.endswith("ing") or w.endswith("ed") or w.endswith("ize") or
            w.endswith("ify") or w.endswith("ate") or w.endswith("en") or
            w.endswith("es") or w.endswith("ied") or w.endswith("ied")):
            tags.append(POS2IDX["VERB"])

        # Nouns (morphological: -tion, -sion, -ment, -ness, -ity, -ism, -ist)
        if (w.endswith("tion") or w.endswith("sion") or w.endswith("ment") or
            w.endswith("ness") or w.endswith("ity") or w.endswith("ism") or
            w.endswith("ist") or w.endswith("ence") or w.endswith("ance") or
            w.endswith("er") or w.endswith("or") or w.endswith("dom") or
            w.endswith("ship") or w.endswith("hood")):
            tags.append(POS2IDX["NOUN"])

        # NOUN: always include as possibility for content words
        # Most English words can function as nouns in context
        if len(w) >= 2 and w[0].isalpha():
            if POS2IDX["NOUN"] not in tags:
                tags.append(POS2IDX["NOUN"])

        # VERB: many English words can also function as verbs
        if len(w) >= 3 and w[0].isalpha() and POS2IDX["VERB"] not in tags and POS2IDX["AUX"] not in tags:
            tags.append(POS2IDX["VERB"])

        # Ensure every word has at least one tag
        if not tags:
            tags = [POS2IDX["NOUN"]]

        return tags

    def build_from_vocabulary(
        self,
        word2idx: Dict[str, int],
        idx2word: Dict[int, str],
    ) -> "POSTypeSystem":
        """
        Build emission weights from vocabulary using rule-based POS assignment.

        For each word, assign possible POS tags and set I_emit[w, t] = 1
        for each allowed type. Multiple tags get equal weight.
        """
        for idx, word in idx2word.items():
            if idx >= self.vocab_size:
                continue
            # Skip special tokens
            if word.startswith("<") and word.endswith(">"):
                self.I_emit[idx, POS2IDX["X"]] = 1
                self.allowed_types[idx] = {POS2IDX["X"]}
                continue

            tags = self.assign_pos_rules(word, idx)
            self.allowed_types[idx] = set(tags)
            for t in tags:
                self.I_emit[idx, t] = 1  # equal weight for each possible tag

        return self

    def compute_type_couplings(
        self,
        sequences: List[List[int]],
        idx2word: Dict[int, str],
        min_count: int = 1,
        scaling: int = 10,
    ) -> "POSTypeSystem":
        """
        Compute type-type couplings from sequences using assigned POS tags.

        For each pair of words (w_i, w_j) within the window, increment
        J_type[tag(w_i), tag(w_j)] by 1. If a word has multiple possible
        tags, increment each combination.

        Pure integer counting.
        """
        type_bigram = Counter()  # (t1, t2) -> count
        type_bigram_by_dist: Dict[int, Counter] = defaultdict(Counter)

        for seq in sequences:
            # Assign POS tags to each position
            # For multi-tag words, use the FIRST (primary) tag
            # Priority order: closed-class tags first, then NOUN, then VERB
            # This matches linguistic reality: function words are unambiguous,
            # content words default to NOUN with VERB as secondary
            TAG_PRIORITY = {
                POS2IDX["PUNCT"]: 0, POS2IDX["DET"]: 1, POS2IDX["PRON"]: 2,
                POS2IDX["AUX"]: 3, POS2IDX["CONJ"]: 4, POS2IDX["PART"]: 5,
                POS2IDX["PREP"]: 6, POS2IDX["NUM"]: 7, POS2IDX["ADV"]: 8,
                POS2IDX["ADJ"]: 9, POS2IDX["NOUN"]: 10, POS2IDX["VERB"]: 11,
                POS2IDX["X"]: 12,
            }
            seq_tags = []
            for w in seq:
                if w in self.allowed_types and self.allowed_types[w]:
                    tags = list(self.allowed_types[w])
                    # Pick the tag with highest priority (lowest number)
                    best_t = min(tags, key=lambda t: TAG_PRIORITY.get(t, 99))
                    seq_tags.append(best_t)
                else:
                    seq_tags.append(POS2IDX["X"])

            # Count type bigrams within window
            for i, t1 in enumerate(seq_tags):
                for j_offset in range(1, self.window + 1):
                    j = i + j_offset
                    if j < len(seq_tags):
                        t2 = seq_tags[j]
                        type_bigram[(t1, t2)] += 1
                        type_bigram_by_dist[j_offset][(t1, t2)] += 1

        # Fill J_type matrix
        for (t1, t2), count in type_bigram.items():
            self.J_type[t1, t2] = count * scaling

        # Fill distance-specific type couplings
        self.J_type_by_dist = {}
        for dist, counts in type_bigram_by_dist.items():
            self.J_type_by_dist[dist] = {}
            for (t1, t2), count in counts.items():
                if count * scaling > 0:
                    self.J_type_by_dist[dist][(t1, t2)] = count * scaling

        return self

    def build_grammar_penalties(self, penalty_strength: int = 50) -> "POSTypeSystem":
        """
        Define grammar penalty constraints as integer quadratic penalties.

        Each constraint has the form:
            penalty = P * violation(types, values)

        where P is a large integer and violation() returns 0 or 1.
        All delta functions are integer comparisons.

        Constraints implemented:
          1. DET must be followed by NOUN-like word (within distance 2)
          2. AUX must be followed by VERB (within distance 2)
          3. PREP must be followed by NOUN-like or DET (within distance 2)
          4. Subject-verb agreement: NOUN_sing should pair with VERB_sing
          5. No two DETs in a row
          6. No two PREPs in a row
          7. ADJ should be near NOUN (within distance 2)
          8. CONJ should connect same-type words
        """
        P = penalty_strength

        # Constraint 1: DET → NOUN (within distance 2)
        self.grammar_penalties.append({
            "name": "DET_NOUN",
            "type1": POS2IDX["DET"],
            "type2_set": [POS2IDX[t] for t in NOUN_LIKE],
            "max_dist": 2,
            "penalty": P,
            "direction": "forward",
        })

        # Constraint 2: AUX → VERB
        self.grammar_penalties.append({
            "name": "AUX_VERB",
            "type1": POS2IDX["AUX"],
            "type2_set": [POS2IDX[t] for t in VERB_LIKE],
            "max_dist": 2,
            "penalty": P,
            "direction": "forward",
        })

        # Constraint 3: PREP → NOUN/DET
        self.grammar_penalties.append({
            "name": "PREP_NOUN",
            "type1": POS2IDX["PREP"],
            "type2_set": [POS2IDX[t] for t in NOUN_LIKE] + [POS2IDX["DET"]],
            "max_dist": 3,
            "penalty": P,
            "direction": "forward",
        })

        # Constraint 4: No two DETs in a row
        self.grammar_penalties.append({
            "name": "NO_DOUBLE_DET",
            "type1": POS2IDX["DET"],
            "type2_set": [POS2IDX["DET"]],
            "max_dist": 1,
            "penalty": P * 2,  # stronger penalty for obvious violations
            "direction": "both",
            "forbid": True,  # penalize same-type adjacency
        })

        # Constraint 5: No two PREPs in a row
        self.grammar_penalties.append({
            "name": "NO_DOUBLE_PREP",
            "type1": POS2IDX["PREP"],
            "type2_set": [POS2IDX["PREP"]],
            "max_dist": 1,
            "penalty": P * 2,
            "direction": "both",
            "forbid": True,
        })

        # Constraint 6: ADJ should be near NOUN
        self.grammar_penalties.append({
            "name": "ADJ_NOUN",
            "type1": POS2IDX["ADJ"],
            "type2_set": [POS2IDX["NOUN"]],
            "max_dist": 2,
            "penalty": P // 2,  # softer: ADJ-NOUN is common but not required
            "direction": "forward",
        })

        # Constraint 7: CONJ should connect compatible types
        self.grammar_penalties.append({
            "name": "CONJ_COMPAT",
            "type1": POS2IDX["CONJ"],
            "type2_set": list(range(self.n_types)),  # any type is ok
            "max_dist": 2,
            "penalty": P // 3,  # soft compatibility
            "direction": "both",
        })

        return self

    def compute_grammar_penalty(
        self, types: List[int], pos: int, proposed_type: int
    ) -> int:
        """
        Compute grammar penalty for having proposed_type at position pos.

        Returns integer penalty (0 if no violations).
        Pure integer comparison and addition.
        """
        total_penalty = 0

        for constraint in self.grammar_penalties:
            type1 = constraint["type1"]
            type2_set = set(constraint["type2_set"])
            max_dist = constraint["max_dist"]
            penalty = constraint["penalty"]
            direction = constraint["direction"]
            forbid = constraint.get("forbid", False)

            # Check if proposed_type is the trigger type
            if proposed_type == type1:
                # Check neighbors for violation
                for d in range(1, max_dist + 1):
                    # Forward
                    if direction in ("forward", "both"):
                        j = pos + d
                        if j < len(types):
                            neighbor_type = types[j]
                            if forbid:
                                # Penalize if neighbor IS in type2_set
                                if neighbor_type in type2_set:
                                    total_penalty += penalty
                            else:
                                # Penalize if neighbor is NOT in type2_set
                                if neighbor_type not in type2_set:
                                    total_penalty += penalty // d  # decay with distance

                    # Backward
                    if direction in ("backward", "both"):
                        j = pos - d
                        if j >= 0:
                            neighbor_type = types[j]
                            if forbid:
                                if neighbor_type in type2_set:
                                    total_penalty += penalty
                            else:
                                if neighbor_type not in type2_set:
                                    total_penalty += penalty // d

            # Check if proposed_type violates a neighbor's constraint
            # (neighbor is type1, proposed should be in type2_set)
            if not forbid:
                for d in range(1, max_dist + 1):
                    if direction in ("forward", "both"):
                        j = pos - d  # look backward for a trigger
                        if j >= 0 and types[j] == type1:
                            if proposed_type not in type2_set:
                                total_penalty += penalty // d

                    if direction in ("backward", "both"):
                        j = pos + d  # look forward for a trigger
                        if j < len(types) and types[j] == type1:
                            if proposed_type not in type2_set:
                                total_penalty += penalty // d

        return total_penalty

    def get_allowed_words_for_type(self, type_idx: int) -> List[int]:
        """Get all word indices that can have this type."""
        col = self.I_emit[:, type_idx]
        return [int(i) for i in range(self.vocab_size) if col[i] > 0]

    def get_type_for_word(self, word_idx: int) -> int:
        """Get the most likely type for a word."""
        if word_idx in self.allowed_types and self.allowed_types[word_idx]:
            return max(self.allowed_types[word_idx],
                       key=lambda t: int(self.I_emit[word_idx, t]))
        return POS2IDX["X"]

    def precompute_type_distribution(self) -> np.ndarray:
        """
        Precompute type proposal distribution from J_type.
        Returns cumulative sum for sampling.
        """
        # Marginal type distribution from J_type
        type_counts = self.J_type.sum(axis=1)  # sum over t2
        if type_counts.sum() == 0:
            type_counts = np.ones(self.n_types, dtype=np.int64)
        self.type_cumsum = np.cumsum(type_counts)
        return self.type_cumsum

    def save(self, path: str):
        """Save type system to disk."""
        np.save(f"{path}_J_type.npy", self.J_type)
        np.save(f"{path}_I_emit.npy", self.I_emit)

        # Save allowed types as JSON
        allowed_ser = {str(k): list(v) for k, v in self.allowed_types.items()}
        with open(f"{path}_allowed.json", "w") as f:
            json.dump(allowed_ser, f)

        # Save grammar penalties
        with open(f"{path}_penalties.json", "w") as f:
            json.dump(self.grammar_penalties, f)

        # Save distance-specific type couplings
        j_by_dist_ser = {}
        for dist, couplings in self.J_type_by_dist.items():
            j_by_dist_ser[str(dist)] = {
                f"{t1},{t2}": c for (t1, t2), c in couplings.items()
            }
        with open(f"{path}_J_type_by_dist.json", "w") as f:
            json.dump(j_by_dist_ser, f)

        meta = {
            "vocab_size": self.vocab_size,
            "n_types": self.n_types,
            "window": self.window,
        }
        with open(f"{path}_meta.json", "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> "POSTypeSystem":
        """Load type system from disk."""
        with open(f"{path}_meta.json") as f:
            meta = json.load(f)

        ts = cls(
            vocab_size=meta["vocab_size"],
            n_types=meta["n_types"],
            window=meta["window"],
        )
        ts.J_type = np.load(f"{path}_J_type.npy")
        ts.I_emit = np.load(f"{path}_I_emit.npy")

        with open(f"{path}_allowed.json") as f:
            allowed_ser = json.load(f)
        ts.allowed_types = {int(k): set(v) for k, v in allowed_ser.items()}

        with open(f"{path}_penalties.json") as f:
            ts.grammar_penalties = json.load(f)

        try:
            with open(f"{path}_J_type_by_dist.json") as f:
                j_by_dist_ser = json.load(f)
            ts.J_type_by_dist = {}
            for dist_str, couplings_dict in j_by_dist_ser.items():
                dist = int(dist_str)
                ts.J_type_by_dist[dist] = {}
                for key_str, count in couplings_dict.items():
                    t1, t2 = map(int, key_str.split(","))
                    ts.J_type_by_dist[dist][(t1, t2)] = count
        except FileNotFoundError:
            ts.J_type_by_dist = {}

        ts.precompute_type_distribution()
        return ts
