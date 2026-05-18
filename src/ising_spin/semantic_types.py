"""
Semantic Type System for the Ising Spin Language Model.

Clusters words into semantic types using integer-only operations,
then builds a compatibility matrix S[sem_type_i, sem_type_j] that
gates the lexical coupling (Ising-Potts gating mechanism).

The gating follows: J_effective[i,j] = S[sem_type(w_i), sem_type(w_j)] * J_PMI[w_i, w_j]

All operations are integer-only in the generation loop.

Semantic types are derived from co-occurrence patterns using:
  1. Integer co-occurrence clustering (k-means with integer centroids)
  2. WordNet super-senses (if available, as integer mapping)
  3. Distributional similarity with integer vectors (Random Indexing)
"""

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple
import json
import numpy as np


# Default semantic super-senses (based on WordNet lexicographer classes)
SEMANTIC_SUPERTYPES = [
    "PERSON",       # people, groups
    "ANIMAL",       # animals
    "ARTIFACT",     # objects, tools, buildings
    "SUBSTANCE",    # materials, elements
    "FOOD",         # food, drink
    "EVENT",        # events, actions
    "STATE",        # states, conditions
    "LOCATION",     # places
    "TIME",         # temporal
    "COMMUNICATION",# speech, writing
    "COGNITION",    # thinking, knowing
    "MOTION",       # movement
    "PERCEPTION",   # seeing, hearing
    "EMOTION",      # feelings
    "QUANTITY",     # numbers, measures
    "RELATION",     # connections, comparisons
    "POSSESSION",   # owning, giving
    "BODY",         # body parts
    "NATURAL",      # natural phenomena
    "OTHER",        # catch-all
]

SEM2IDX = {s: i for i, s in enumerate(SEMANTIC_SUPERTYPES)}
N_SEM = len(SEMANTIC_SUPERTYPES)


class SemanticTypeSystem:
    """
    Integer-only semantic type system.

    Components:
      - word_to_sem[w]: semantic type index for word w
      - S[t, t']: compatibility matrix (integer: +K compatible, -K incompatible)
      - J_sem[w, w']: semantic coupling (Hebbian, gated by S)
    """

    def __init__(
        self,
        vocab_size: int,
        n_sem_types: int = N_SEM,
        compatibility_strength: int = 3,
    ):
        self.vocab_size = vocab_size
        self.n_sem_types = n_sem_types
        self.compatibility_strength = compatibility_strength

        # Word -> semantic type mapping
        self.word_to_sem = np.zeros(vocab_size, dtype=np.int64)
        # Default: all words start as "OTHER"
        self.word_to_sem[:] = n_sem_types - 1

        # Semantic compatibility matrix: S[t, t'] = integer
        self.S = np.zeros((n_sem_types, n_sem_types), dtype=np.int64)

        # Semantic coupling: J_sem[w, w'] = gated Hebbian coupling
        self.J_sem = np.zeros((vocab_size, vocab_size), dtype=np.int64)

        # Co-occurrence vectors for clustering (integer sparse)
        self.cooc_vectors: Optional[np.ndarray] = None

        # Semantic type names for display
        self.type_names = SEMANTIC_SUPERTYPES[:n_sem_types]

    def _assign_semantic_rules(
        self, word: str, idx2word: Dict[int, str] = None
    ) -> int:
        """
        Rule-based semantic type assignment using morphological and
        lexical heuristics. No ML, no FP — pure string matching.
        """
        w = word.lower()

        # Person indicators
        if (w.endswith("er") or w.endswith("or") or w.endswith("ist") or
            w.endswith("ian") or w.endswith("man") or w.endswith("woman") or
            w.endswith("person") or w.endswith("people")):
            if w not in {"her", "other", "under", "over", "water", "after",
                         "either", "neither", "whether", "order", "matter",
                         "consider", "enter", "offer", "other", "proper"}:
                return SEM2IDX["PERSON"]

        # Location indicators
        if (w.endswith("land") or w.endswith("town") or w.endswith("city") or
            w.endswith("country") or w.endswith("ville") or w.endswith("burg") or
            w.endswith("stan") or w.endswith("shire")):
            return SEM2IDX["LOCATION"]

        # Time indicators
        if w in {"today", "tomorrow", "yesterday", "now", "then", "later",
                 "morning", "evening", "night", "day", "week", "month", "year",
                 "hour", "minute", "second", "monday", "tuesday", "wednesday",
                 "thursday", "friday", "saturday", "sunday", "january",
                 "february", "march", "april", "may", "june", "july",
                 "august", "september", "october", "november", "december",
                 "spring", "summer", "autumn", "winter", "always", "never",
                 "sometimes", "often", "rarely", "already", "soon", "recently"}:
            return SEM2IDX["TIME"]

        # Food indicators
        if (w.endswith("food") or w.endswith("drink") or w.endswith("meat") or
            w.endswith("fruit") or w.endswith("bread") or w.endswith("cake") or
            w in {"water", "milk", "bread", "rice", "meat", "fish", "egg",
                  "tea", "coffee", "beer", "wine", "sugar", "salt", "flour",
                  "butter", "cheese", "apple", "orange", "chicken", "beef"}):
            return SEM2IDX["FOOD"]

        # Emotion indicators
        if (w.endswith("ness") and w in {"happiness", "sadness", "loneliness",
                                          "madness", "darkness", "illness"} or
            w in {"love", "hate", "fear", "anger", "joy", "sadness", "happy",
                  "sad", "angry", "afraid", "scared", "excited", "worried",
                  "anxious", "proud", "ashamed", "grateful", "hopeful"}):
            return SEM2IDX["EMOTION"]

        # Body part indicators
        if (w in {"head", "hand", "foot", "feet", "eye", "eyes", "ear",
                  "nose", "mouth", "heart", "brain", "arm", "leg", "back",
                  "face", "hair", "skin", "bone", "blood", "finger", "toe"}):
            return SEM2IDX["BODY"]

        # Motion indicators
        if (w in {"go", "come", "walk", "run", "move", "travel", "drive",
                  "fly", "swim", "climb", "jump", "fall", "rise", "turn",
                  "leave", "arrive", "enter", "exit", "cross", "pass"} or
            w.endswith("ing") and len(w) > 5):
            return SEM2IDX["MOTION"]

        # Communication indicators
        if (w in {"say", "tell", "speak", "talk", "write", "read", "call",
                  "ask", "answer", "explain", "describe", "discuss", "argue",
                  "agree", "disagree", "claim", "state", "suggest", "mention"} or
            w.endswith("tion") or w.endswith("sion")):
            return SEM2IDX["COMMUNICATION"]

        # Cognition indicators
        if (w in {"think", "know", "believe", "understand", "learn",
                  "remember", "forget", "imagine", "consider", "realize",
                  "discover", "assume", "suppose", "doubt", "wonder",
                  "science", "research", "theory", "hypothesis", "evidence",
                  "analysis", "logic", "reason", "idea", "concept"}):
            return SEM2IDX["COGNITION"]

        # Quantity indicators
        if (w in {"many", "much", "few", "little", "lot", "some", "all",
                  "none", "half", "third", "quarter", "number", "amount",
                  "total", "sum", "average", "rate", "percent", "ratio"} or
            w.replace(".", "").replace(",", "").replace("-", "").isdigit()):
            return SEM2IDX["QUANTITY"]

        # Event indicators
        if (w.endswith("ment") or w.endswith("event") or w.endswith("cess") or
            w in {"war", "election", "revolution", "accident", "disaster",
                  "ceremony", "festival", "game", "match", "battle",
                  "meeting", "conference", "party", "concert"}):
            return SEM2IDX["EVENT"]

        # Artifact indicators
        if (w.endswith("tion") or w.endswith("tool") or w.endswith("machine") or
            w.endswith("device") or w.endswith("system") or w.endswith("structure") or
            w in {"book", "car", "house", "door", "window", "table", "chair",
                  "computer", "phone", "road", "bridge", "building", "ship",
                  "plane", "train", "weapon", "instrument"}):
            return SEM2IDX["ARTIFACT"]

        # Natural phenomena
        if (w in {"sun", "moon", "star", "earth", "sky", "cloud", "rain",
                  "snow", "wind", "fire", "ocean", "river", "mountain",
                  "forest", "tree", "flower", "grass", "weather", "light",
                  "dark", "energy", "force", "gravity", "heat", "cold"}):
            return SEM2IDX["NATURAL"]

        # Default
        return SEM2IDX["OTHER"]

    def build_from_vocabulary(
        self,
        word2idx: Dict[str, int],
        idx2word: Dict[int, str],
    ) -> "SemanticTypeSystem":
        """
        Assign semantic types to all words using rule-based heuristics.
        Pure string matching — no FP, no ML.
        """
        for idx, word in idx2word.items():
            if idx >= self.vocab_size:
                continue
            self.word_to_sem[idx] = self._assign_semantic_rules(word, idx2word)

        return self

    def compute_compatibility_matrix(
        self,
        sequences: List[List[int]],
        min_cooc: int = 2,
    ) -> "SemanticTypeSystem":
        """
        Compute semantic compatibility matrix S from co-occurrence statistics.

        S[t, t'] = compatibility of semantic types t and t'.

        Computed as: if sem_types t and t' co-occur more than expected,
        S = +K (compatible). If less, S = -K (incompatible). If neutral, S = 0.

        Uses integer-only comparison (same log-floor PMI logic).
        """
        # Count semantic type co-occurrences within window
        sem_cooc = Counter()  # (t, t') -> count
        sem_marginal = Counter()  # t -> count

        for seq in sequences:
            # Assign semantic types
            seq_sems = [int(self.word_to_sem[w]) for w in seq]

            # Count
            for s in seq_sems:
                sem_marginal[s] += 1

            for i, s1 in enumerate(seq_sems):
                for j in range(i + 1, min(i + 6, len(seq_sems))):
                    s2 = seq_sems[j]
                    sem_cooc[(s1, s2)] += 1
                    sem_cooc[(s2, s1)] += 1

        total = sum(sem_marginal.values())
        K = self.compatibility_strength

        # Compute compatibility using integer association test
        for t1 in range(self.n_sem_types):
            for t2 in range(self.n_sem_types):
                cooc = sem_cooc.get((t1, t2), 0)
                m1 = sem_marginal.get(t1, 0)
                m2 = sem_marginal.get(t2, 0)

                if cooc < min_cooc or m1 == 0 or m2 == 0 or total == 0:
                    self.S[t1, t2] = 0
                    continue

                # Integer association: compare observed vs expected
                # expected = m1 * m2 / total (but we avoid FP)
                # Instead: compare cooc * total vs m1 * m2
                numerator = int(cooc) * int(total)
                denominator = int(m1) * int(m2)

                if numerator > denominator * 2:
                    # Strong positive association
                    self.S[t1, t2] = K
                elif numerator > denominator:
                    # Moderate positive association
                    self.S[t1, t2] = K // 2
                elif numerator < denominator // 2:
                    # Strong negative association (repulsion)
                    self.S[t1, t2] = -K
                elif numerator < denominator:
                    # Moderate negative association
                    self.S[t1, t2] = -K // 2
                else:
                    self.S[t1, t2] = 0

        # Self-compatibility is always positive
        for t in range(self.n_sem_types):
            self.S[t, t] = K

        return self

    def compute_hebbian_coupling(
        self,
        sequences: List[List[int]],
        hebbian_weight: int = 1,
    ) -> "SemanticTypeSystem":
        """
        Compute Hebbian (sentence-level) coupling gated by semantic compatibility.

        J_sem[w, w'] = S[sem_type(w), sem_type(w')] * cooc_count(w, w')

        The S matrix gates the Hebbian coupling — only semantically
        compatible words get positive coupling.
        """
        # Count sentence-level co-occurrence
        for seq in sequences:
            words_in_seq = set(seq)
            for w in words_in_seq:
                for w2 in words_in_seq:
                    if w != w2 and w < self.vocab_size and w2 < self.vocab_size:
                        s1 = int(self.word_to_sem[w])
                        s2 = int(self.word_to_sem[w2])
                        compat = int(self.S[s1, s2])
                        # Hebbian + semantic gating
                        self.J_sem[w, w2] += compat * hebbian_weight

        return self

    def gate_pmi_coupling(
        self, J_PMI: np.ndarray, alpha: int = 3, beta: int = 1
    ) -> np.ndarray:
        """
        Combine PMI and semantically-gated Hebbian couplings.

        J_total = alpha * J_PMI + beta * J_sem

        Both terms are integer matrices.
        """
        return alpha * J_PMI + beta * self.J_sem

    def save(self, path: str):
        """Save semantic type system to disk."""
        np.save(f"{path}_word_to_sem.npy", self.word_to_sem)
        np.save(f"{path}_S.npy", self.S)
        np.save(f"{path}_J_sem.npy", self.J_sem)

        meta = {
            "vocab_size": self.vocab_size,
            "n_sem_types": self.n_sem_types,
            "compatibility_strength": self.compatibility_strength,
            "type_names": self.type_names,
        }
        with open(f"{path}_meta.json", "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> "SemanticTypeSystem":
        """Load semantic type system from disk."""
        with open(f"{path}_meta.json") as f:
            meta = json.load(f)

        sts = cls(
            vocab_size=meta["vocab_size"],
            n_sem_types=meta["n_sem_types"],
            compatibility_strength=meta["compatibility_strength"],
        )
        sts.type_names = meta.get("type_names", SEMANTIC_SUPERTYPES[:sts.n_sem_types])
        sts.word_to_sem = np.load(f"{path}_word_to_sem.npy")
        sts.S = np.load(f"{path}_S.npy")
        sts.J_sem = np.load(f"{path}_J_sem.npy")

        return sts
