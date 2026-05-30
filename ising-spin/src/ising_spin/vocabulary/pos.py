"""
Integer-only POS type system for the Attractor Language Machine.

Components:
  - I_emit[w, t]: emission weight (count of word w tagged as type t)
  - allowed_types[w]: set of types word w can have
  - grammar_penalties: list of (condition, penalty) pairs for hard constraints
  - J_type[t1, t2]: type-type coupling (POS bigram counts)
"""

from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np

# ===========================================================================
# COARSE POS TAGS
# ===========================================================================

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

NOUN_LIKE = {"NOUN", "PRON", "NUM"}
VERB_LIKE = {"VERB", "AUX"}
OPEN_CLASS = {"NOUN", "VERB", "ADJ", "ADV"}
CLOSED_CLASS = {"DET", "PREP", "PRON", "AUX", "CONJ", "PART"}


class POSTypeSystem:
    """
    Integer-only POS type system for the Attractor Language Machine.

    Components:
      - I_emit[w, t]: emission weight (count of word w tagged as type t)
      - allowed_types[w]: set of types word w can have
      - grammar_penalties: list of (condition, penalty) pairs for hard constraints
      - J_type[t1, t2]: type-type coupling (POS bigram counts)
    """

    def __init__(self, vocab_size: int, n_types: int = N_POS, window: int = 5):
        self.vocab_size = vocab_size
        self.n_types = n_types
        self.window = window
        self.J_type = np.zeros((n_types, n_types), dtype=np.int64)
        self.J_type_by_dist: Dict[int, Dict[Tuple[int, int], int]] = {}
        self.I_emit = np.zeros((vocab_size, n_types), dtype=np.int64)
        self.allowed_types: Dict[int, Set[int]] = {}
        self.grammar_penalties: List[Dict] = []

    def assign_pos_rules(self, word: str, word_idx: int) -> List[int]:
        """Rule-based POS assignment for English words. No FP, no ML model."""
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

        # Adjectives (morphological)
        if (w.endswith("ful") or w.endswith("less") or w.endswith("ous") or
            w.endswith("ive") or w.endswith("able") or w.endswith("ible") or
            w.endswith("al") or w.endswith("ial") or w.endswith("ent") or
            w.endswith("ant") or w.endswith("ic") or w.endswith("ical")):
            tags.append(POS2IDX["ADJ"])

        # Adverbs (morphological: -ly)
        if w.endswith("ly"):
            tags.append(POS2IDX["ADV"])

        # Verbs (morphological)
        if (w.endswith("ing") or w.endswith("ed") or w.endswith("ize") or
            w.endswith("ify") or w.endswith("ate") or w.endswith("en") or
            w.endswith("es") or w.endswith("ied")):
            tags.append(POS2IDX["VERB"])

        # Nouns (morphological)
        if (w.endswith("tion") or w.endswith("sion") or w.endswith("ment") or
            w.endswith("ness") or w.endswith("ity") or w.endswith("ism") or
            w.endswith("ist") or w.endswith("ence") or w.endswith("ance") or
            w.endswith("er") or w.endswith("or") or w.endswith("dom") or
            w.endswith("ship") or w.endswith("hood")):
            tags.append(POS2IDX["NOUN"])

        # Most English words can function as nouns
        if len(w) >= 2 and w[0].isalpha():
            if POS2IDX["NOUN"] not in tags:
                tags.append(POS2IDX["NOUN"])

        # Many English words can also function as verbs
        if len(w) >= 3 and w[0].isalpha() and POS2IDX["VERB"] not in tags and POS2IDX["AUX"] not in tags:
            tags.append(POS2IDX["VERB"])

        if not tags:
            tags = [POS2IDX["NOUN"]]

        return tags

    def build_from_vocabulary(
        self, word2idx: Dict[str, int], idx2word: Dict[int, str]
    ) -> "POSTypeSystem":
        """Build emission weights from vocabulary using rule-based POS assignment."""
        for idx, word in idx2word.items():
            if idx >= self.vocab_size:
                continue
            if word.startswith("<") and word.endswith(">"):
                self.I_emit[idx, POS2IDX["X"]] = 1
                self.allowed_types[idx] = {POS2IDX["X"]}
                continue
            tags = self.assign_pos_rules(word, idx)
            self.allowed_types[idx] = set(tags)
            for t in tags:
                self.I_emit[idx, t] = 1
        return self

    def compute_type_couplings(
        self, sequences: List[List[int]], idx2word: Dict[int, str],
        min_count: int = 1, scaling: int = 10
    ) -> "POSTypeSystem":
        """Compute type-type couplings from sequences. Pure integer counting."""
        TAG_PRIORITY = {
            POS2IDX["PUNCT"]: 0, POS2IDX["DET"]: 1, POS2IDX["PRON"]: 2,
            POS2IDX["AUX"]: 3, POS2IDX["CONJ"]: 4, POS2IDX["PART"]: 5,
            POS2IDX["PREP"]: 6, POS2IDX["NUM"]: 7, POS2IDX["ADV"]: 8,
            POS2IDX["ADJ"]: 9, POS2IDX["NOUN"]: 10, POS2IDX["VERB"]: 11,
            POS2IDX["X"]: 12,
        }
        type_bigram = Counter()
        type_bigram_by_dist: Dict[int, Counter] = defaultdict(Counter)

        for seq in sequences:
            seq_tags = []
            for w in seq:
                if w in self.allowed_types and self.allowed_types[w]:
                    tags = list(self.allowed_types[w])
                    best_t = min(tags, key=lambda t: TAG_PRIORITY.get(t, 99))
                    seq_tags.append(best_t)
                else:
                    seq_tags.append(POS2IDX["X"])

            for i, t1 in enumerate(seq_tags):
                for j_offset in range(1, self.window + 1):
                    j = i + j_offset
                    if j < len(seq_tags):
                        t2 = seq_tags[j]
                        type_bigram[(t1, t2)] += 1
                        type_bigram_by_dist[j_offset][(t1, t2)] += 1

        for (t1, t2), count in type_bigram.items():
            self.J_type[t1, t2] = count * scaling

        self.J_type_by_dist = {}
        for dist, counts in type_bigram_by_dist.items():
            self.J_type_by_dist[dist] = {}
            for (t1, t2), count in counts.items():
                if count * scaling > 0:
                    self.J_type_by_dist[dist][(t1, t2)] = count * scaling
        return self

    def build_grammar_penalties(self, penalty_strength: int = 50) -> "POSTypeSystem":
        """Define grammar penalty constraints as integer quadratic penalties."""
        P = penalty_strength

        self.grammar_penalties = [
            {"name": "DET_NOUN", "type1": POS2IDX["DET"],
             "type2_set": [POS2IDX[t] for t in NOUN_LIKE],
             "max_dist": 2, "penalty": P, "direction": "forward"},
            {"name": "AUX_VERB", "type1": POS2IDX["AUX"],
             "type2_set": [POS2IDX[t] for t in VERB_LIKE],
             "max_dist": 2, "penalty": P, "direction": "forward"},
            {"name": "PREP_NOUN", "type1": POS2IDX["PREP"],
             "type2_set": [POS2IDX[t] for t in NOUN_LIKE] + [POS2IDX["DET"]],
             "max_dist": 3, "penalty": P, "direction": "forward"},
            {"name": "NO_DOUBLE_DET", "type1": POS2IDX["DET"],
             "type2_set": [POS2IDX["DET"]],
             "max_dist": 1, "penalty": P * 2, "direction": "both", "forbid": True},
            {"name": "NO_DOUBLE_PREP", "type1": POS2IDX["PREP"],
             "type2_set": [POS2IDX["PREP"]],
             "max_dist": 1, "penalty": P * 2, "direction": "both", "forbid": True},
            {"name": "ADJ_NOUN", "type1": POS2IDX["ADJ"],
             "type2_set": [POS2IDX["NOUN"]],
             "max_dist": 2, "penalty": P // 2, "direction": "forward"},
            {"name": "CONJ_COMPAT", "type1": POS2IDX["CONJ"],
             "type2_set": list(range(self.n_types)),
             "max_dist": 2, "penalty": P // 3, "direction": "both"},
        ]
        return self

    def compute_grammar_penalty(
        self, types: List[int], pos: int, proposed_type: int
    ) -> int:
        """Compute grammar penalty for having proposed_type at position pos. Pure integer."""
        total_penalty = 0

        for constraint in self.grammar_penalties:
            type1 = constraint["type1"]
            type2_set = set(constraint["type2_set"])
            max_dist = constraint["max_dist"]
            penalty = constraint["penalty"]
            direction = constraint["direction"]
            forbid = constraint.get("forbid", False)

            if proposed_type == type1:
                for d in range(1, max_dist + 1):
                    if direction in ("forward", "both"):
                        j = pos + d
                        if j < len(types):
                            neighbor_type = types[j]
                            if forbid:
                                if neighbor_type in type2_set:
                                    total_penalty += penalty
                            else:
                                if neighbor_type not in type2_set:
                                    total_penalty += penalty // d
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

            if not forbid:
                for d in range(1, max_dist + 1):
                    if direction in ("forward", "both"):
                        j = pos - d
                        if j >= 0 and types[j] == type1:
                            if proposed_type not in type2_set:
                                total_penalty += penalty // d
                    if direction in ("backward", "both"):
                        j = pos + d
                        if j < len(types) and types[j] == type1:
                            if proposed_type not in type2_set:
                                total_penalty += penalty // d

        return total_penalty
