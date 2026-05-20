"""
Ising-Enhanced N-Gram Language Model.

A non-neural language model where:
  1. N-gram recall provides the PRIMARY next-word signal
  2. PMI couplings provide SECONDARY signal when recall misses
  3. POS grammar provides HARD CONSTRAINTS on word types
  4. Integer Boltzmann sampling provides STOCHASTIC selection

The Ising model contributes through:
  - PMI coupling matrix J[w,w'] = log-floor PMI (word affinities)
  - Local field h[w] = self-information (unigram frequency)
  - Energy function: E(w|ctx) = -J[w,ctx] - h[w] + penalties
  - Temperature-controlled stochastic selection

INTEGER-ONLY CONSTRAINT (enforced):
  - ALL generation-path computation uses integer arithmetic
  - Boltzmann sampling via pre-computed lookup table (NO np.exp in hot loop)
  - The ONLY floating-point is in building the lookup table at __init__ time

References:
  - Levy & Goldberg (2014): Word2Vec as log-PMI matrix factorization
  - Marcolli (2015): Implicational couplings in syntax
  - Haydarov, Omirov & Rozikov (arXiv:2502.12014): Ising-Potts coupling
  - Creutz (1983): Demon algorithm for integer MCMC acceptance
"""

import math
import json
import time
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Set


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


# ===========================================================================
# VOCABULARY
# ===========================================================================

class Vocabulary:
    """
    Integer-only vocabulary mapping between words and indices.

    Special tokens:
        <UNK>=0, <BOS>=1, <EOS>=2, <PAD>=3
    """

    UNK = "<UNK>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    PAD = "<PAD>"
    SPECIALS = [UNK, BOS, EOS, PAD]

    def __init__(self, min_freq: int = 5, max_size: Optional[int] = None):
        self.min_freq = min_freq
        self.max_size = max_size
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_counts: Counter = Counter()
        self._built = False

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer. Pure string manipulation."""
        tokens = []
        for word in text.split():
            stripped = word.strip()
            if not stripped:
                continue
            # Split off leading punctuation
            while stripped and not stripped[0].isalnum():
                tokens.append(stripped[0])
                stripped = stripped[1:]
            # Split off trailing punctuation
            tail = []
            while stripped and not stripped[-1].isalnum():
                tail.append(stripped[-1])
                stripped = stripped[:-1]
            if stripped:
                tokens.append(stripped.lower())
            tokens.extend(reversed(tail))
        return tokens

    def build(self, texts: List[str]) -> "Vocabulary":
        """Build vocabulary from a list of text strings. Pure integer counting."""
        for text in texts:
            tokens = self._tokenize(text)
            self.word_counts.update(tokens)

        idx = 0
        for special in self.SPECIALS:
            self.word2idx[special] = idx
            self.idx2word[idx] = special
            idx += 1

        filtered = [
            (word, count)
            for word, count in self.word_counts.most_common()
            if count >= self.min_freq and word not in self.SPECIALS
        ]
        if self.max_size is not None:
            filtered = filtered[:self.max_size]

        for word, count in filtered:
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            idx += 1

        self._built = True
        return self

    def encode(self, text: str) -> List[int]:
        """Encode a text string to a list of integer token indices."""
        tokens = self._tokenize(text)
        unk_idx = self.word2idx[self.UNK]
        return [self.word2idx.get(t, unk_idx) for t in tokens]

    def decode(self, indices: List[int]) -> str:
        """Decode a list of integer token indices to a text string."""
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, self.UNK)
            if word in (self.BOS, self.EOS, self.PAD):
                continue
            words.append(word)
        return " ".join(words)

    def __len__(self) -> int:
        return len(self.word2idx)


# ===========================================================================
# POS TYPE SYSTEM
# ===========================================================================

class POSTypeSystem:
    """
    Integer-only POS type system for the coupled Ising-Potts model.

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


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_fineweb_edu(
    n_samples: int = 50000,
    split: str = "train",
    subset: str = "sample-10BT",
    min_length: int = 20,
    max_length: int = 2000,
) -> List[str]:
    """Load text samples from the fineweb-edu dataset on HuggingFace."""
    from datasets import load_dataset

    print(f"Loading fineweb-edu ({subset}, split={split})...")

    dataset = None
    for name in ["HuggingFaceFW/fineweb-edu", "HuggingFW/fineweb-edu"]:
        try:
            dataset = load_dataset(name, name=subset, split=split, streaming=True)
            print(f"  Loaded from '{name}' with subset '{subset}'")
            break
        except Exception:
            continue

    if dataset is None:
        for name in ["HuggingFaceFW/fineweb-edu", "HuggingFW/fineweb-edu"]:
            try:
                dataset = load_dataset(name, split=split, streaming=True)
                print(f"  Loaded from '{name}' without subset")
                break
            except Exception:
                continue

    if dataset is None:
        raise RuntimeError(
            "Could not load fineweb-edu. Please check internet and HuggingFace access."
        )

    texts = []
    scanned = 0
    for example in dataset:
        scanned += 1
        if len(texts) >= n_samples:
            break
        text = example.get("text", "").strip()
        if min_length <= len(text) <= max_length:
            texts.append(text)
        if scanned % 10000 == 0:
            print(f"  Scanned {scanned} examples, collected {len(texts)} texts...")
        if scanned > n_samples * 5:
            break

    print(f"Loaded {len(texts)} texts from fineweb-edu (scanned {scanned}).")
    return texts


def tokenize_texts(texts: List[str], vocab: Vocabulary) -> List[List[int]]:
    """Tokenize a list of texts using the vocabulary. Pure integer encoding."""
    sequences = []
    for text in texts:
        tokens = vocab.encode(text)
        if len(tokens) > 0:
            sequences.append(tokens)
    return sequences


def truncate_sequences(
    sequences: List[List[int]], max_len: int = 50
) -> List[List[int]]:
    """Truncate sequences to max_len and filter empty ones. Pure integer operation."""
    return [seq[:max_len] for seq in sequences if len(seq) > 3]


# ===========================================================================
# INTEGER BOLTZMANN SAMPLER
# ===========================================================================

class IntegerBoltzmannSampler:
    """
    Boltzmann sampling using ONLY integer arithmetic in the hot path.

    Pre-computes a lookup table at initialization:
        table[delta] = round(SCALE * exp(-beta * delta))

    At generation time, sampling is pure integer:
        1. deltas = energies - E_min (non-negative integers)
        2. weights = table[deltas] (integer array lookup)
        3. Cumulative sum (integer addition)
        4. Binary search (integer comparison)
    """

    def __init__(self, beta: float = 0.1, max_delta: int = 5000, scale: int = 1 << 30):
        self.beta = beta
        self.scale = scale
        fine_max = min(max_delta, 1000)
        self.table = np.zeros(fine_max + 1, dtype=np.int64)
        for d in range(fine_max + 1):
            raw = math.exp(-beta * d)
            self.table[d] = max(0, int(round(scale * raw)))
        self.max_delta = fine_max

    def sample(self, energies: np.ndarray) -> int:
        """Sample from Boltzmann distribution P(i) ~ exp(-beta * E_i). Integer-only."""
        if len(energies) <= 1:
            return 0

        e_min = int(energies.min())
        deltas = (energies - e_min).astype(np.int64)
        deltas = np.clip(deltas, 0, self.max_delta)

        weights = self.table[deltas]
        total = int(weights.sum())
        if total <= 0:
            return np.random.randint(len(energies))

        r = np.random.randint(0, total)
        cumsum = np.cumsum(weights)
        idx = int(np.searchsorted(cumsum, r, side='right'))
        return min(idx, len(energies) - 1)


# ===========================================================================
# N-GRAM INDEX
# ===========================================================================

class NGramIndex:
    """
    Multi-level n-gram index for exact token recall.

    This is the PRIMARY generation mechanism. When it hits, it produces
    coherent text. When it misses, the Ising PMI model takes over.
    """

    def __init__(self, max_n: int = 5, min_count: int = 1):
        self.max_n = max_n
        self.min_count = min_count
        self.index: Dict[int, Dict[Tuple, Counter]] = {
            k: {} for k in range(1, max_n + 1)
        }
        self.context_totals: Dict[int, Dict[Tuple, int]] = {
            k: {} for k in range(1, max_n + 1)
        }
        self._built = False

    def build(self, sequences: List[List[int]]) -> "NGramIndex":
        """Build n-gram index from tokenized sequences. Integer counting only."""
        for seq in sequences:
            start = 0
            for i, w in enumerate(seq):
                if w >= 4:
                    start = i
                    break

            for t in range(start, len(seq)):
                for k in range(1, self.max_n + 1):
                    if t - k < start:
                        break
                    context = tuple(seq[t-k:t])
                    continuation = seq[t]
                    if any(w < 4 for w in context) or continuation < 4:
                        continue
                    if context not in self.index[k]:
                        self.index[k][context] = Counter()
                    self.index[k][context][continuation] += 1
                    self.context_totals[k][context] = (
                        self.context_totals[k].get(context, 0) + 1
                    )

        # Prune low-count continuations
        for k in range(1, self.max_n + 1):
            for context in list(self.index[k].keys()):
                low_count = [
                    w for w, c in self.index[k][context].items()
                    if c < self.min_count
                ]
                for w in low_count:
                    del self.index[k][context][w]
                    self.context_totals[k][context] -= 1
                if not self.index[k][context]:
                    del self.index[k][context]
                    del self.context_totals[k][context]

        self._built = True
        for k in range(1, self.max_n + 1):
            n_ctx = len(self.index[k])
            n_cont = sum(len(v) for v in self.index[k].values())
            print(f"    {k}-gram: {n_ctx:,} contexts, {n_cont:,} continuations")
        return self

    def lookup(self, context_words: List[int]) -> Dict[int, List[Tuple[int, int, int]]]:
        """Look up n-gram continuations. Returns {k: [(word, count, total), ...]}."""
        results = {}
        for k in range(min(self.max_n, len(context_words)), 0, -1):
            context = tuple(context_words[-k:])
            if context in self.index[k]:
                total = self.context_totals[k][context]
                conts = self.index[k][context].most_common()
                results[k] = [(word, count, total) for word, count in conts]
        return results

    def get_recall_bonus(
        self,
        context_words: List[int],
        candidate_words: np.ndarray,
        recall_scale: int = 100,
        context_weight_factor: int = 4,
        longest_only: bool = True,
    ) -> np.ndarray:
        """
        Compute recall bonus for candidate words based on n-gram matches.

        Uses ONLY the longest matching context by default — prevents common-word inflation.
        For k >= 3: raw bonus (strong signal). For k < 3: normalized by total.
        """
        n_candidates = len(candidate_words)
        bonuses = np.zeros(n_candidates, dtype=np.int64)

        matches = self.lookup(context_words)
        if not matches:
            return bonuses

        if longest_only and matches:
            best_k = max(matches.keys())
            matches = {best_k: matches[best_k]}

        for k, continuations in matches.items():
            context_weight = context_weight_factor ** (k - 1)
            cont_lookup = {}
            for word, count, total in continuations:
                if k >= 3:
                    bonus = count * recall_scale * context_weight
                else:
                    bonus = (count * recall_scale * context_weight) // max(1, total)
                if word not in cont_lookup or bonus > cont_lookup[word]:
                    cont_lookup[word] = int(bonus)

            for i, w in enumerate(candidate_words):
                if int(w) in cont_lookup:
                    bonuses[i] += cont_lookup[int(w)]

        return bonuses

    def get_best_copy_candidate(
        self,
        context_words: List[int],
        min_context_length: int = 3,
        min_confidence: float = 0.3,
    ) -> Optional[Tuple[int, int, int]]:
        """Find best word for direct copying (highest-confidence n-gram match)."""
        matches = self.lookup(context_words)
        for k in sorted(matches.keys(), reverse=True):
            if k < min_context_length:
                break
            continuations = matches[k]
            if not continuations:
                continue
            best_word, best_count, total = continuations[0]
            if best_count * 10 >= total * int(min_confidence * 10):
                return (best_word, best_count, total)
        return None


# ===========================================================================
# PMI COUPLING COMPUTATION
# ===========================================================================

def compute_log_floor_pmi(
    cooc: int, marginal_i: int, marginal_j: int, total: int, cap: int = 15
) -> int:
    """
    Compute log-floor PMI using ONLY integer arithmetic and bit operations.

    PMI(i,j) = log2(C(i,j)*N / (C(i)*C(j)))
             = sign * (bit_length(ratio) - 1)

    Novel: bit_length() as floor(log2()) for integer PMI.
    """
    if cooc == 0 or marginal_i == 0 or marginal_j == 0 or total == 0:
        return 0

    num = int(cooc) * int(total)
    denom = int(marginal_i) * int(marginal_j)

    if num == 0 or denom == 0:
        return 0

    sign = 1 if num > denom else -1
    ratio = max(num, denom) // min(num, denom)

    pmi = sign * (ratio.bit_length() - 1)
    return max(-cap, min(cap, pmi))


def compute_pmi_couplings(
    sequences: List[List[int]],
    vocab_size: int,
    window: int = 5,
    min_count: int = 2,
    pmi_cap: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute PMI coupling matrix J and local field h from sequences.

    J[w, w'] = log-floor PMI(w, w') for co-occurring words within window
    h[w] = self-information = floor(log2(N/count(w)))

    Returns (J, h) as int64 arrays.
    """
    V = vocab_size

    # Count unigrams
    unigram = np.zeros(V, dtype=np.int64)
    for seq in sequences:
        for w in seq:
            unigram[w] += 1
    total_tokens = int(unigram.sum())

    # Count windowed co-occurrences
    cooc_counts = Counter()
    for seq in sequences:
        for i, w in enumerate(seq):
            for j in range(i + 1, min(i + window + 1, len(seq))):
                cooc_counts[(w, seq[j])] += 1

    # Compute PMI coupling matrix
    J = np.zeros((V, V), dtype=np.int64)
    for (w, w2), count in cooc_counts.items():
        if count >= min_count:
            pmi = compute_log_floor_pmi(
                int(count), int(unigram[w]), int(unigram[w2]),
                total_tokens, cap=pmi_cap
            )
            J[w, w2] = pmi
            J[w2, w] = pmi  # Symmetric

    # Compute local field (self-information)
    h = np.ones(V, dtype=np.int64)
    for w in range(V):
        if unigram[w] > 0 and total_tokens > unigram[w]:
            ratio = total_tokens // int(unigram[w])
            if ratio >= 2:
                h[w] = ratio.bit_length() - 1

    n_nonzero = int(np.count_nonzero(J))
    print(f"    PMI matrix: {n_nonzero:,} non-zero entries out of {V*V:,}")
    print(f"    PMI range: [{int(J[J != 0].min()) if n_nonzero > 0 else 0}, "
          f"{int(J.max())}]")

    return J, h


# ===========================================================================
# ISING-ENHANCED N-GRAM LANGUAGE MODEL
# ===========================================================================

class IsingLM:
    """
    Ising-Enhanced N-Gram Language Model.

    Architecture (honest):
      1. POS type selection: Grammar-driven with hard constraints
      2. N-gram recall: Primary next-word signal (when available)
      3. PMI coupling: Secondary signal (when recall misses)
      4. Integer Boltzmann: Temperature-controlled stochastic selection

    Parameters (6 generation params, not 30+):
      - recall_scale, pmi_weight, field_weight
      - beta_type, beta_word
      - ising_enabled (ablation switch)
    """

    CLOSED_CLASS = {POS2IDX["DET"], POS2IDX["PREP"], POS2IDX["PART"],
                    POS2IDX["PRON"], POS2IDX["AUX"], POS2IDX["CONJ"]}

    HARD_TYPE_CONSTRAINTS = {
        POS2IDX["PART"]: [POS2IDX["VERB"]],
        POS2IDX["AUX"]: [POS2IDX["VERB"], POS2IDX["ADV"]],
    }

    def __init__(
        self,
        vocab: Vocabulary,
        ngram_index: NGramIndex,
        J: np.ndarray,
        h: np.ndarray,
        types: POSTypeSystem,
        recall_scale: int = 1000,
        pmi_weight: int = 3,
        field_weight: int = 1,
        beta_type: float = 0.01,
        beta_word: float = 0.15,
        copy_enabled: bool = True,
        copy_min_context: int = 2,
        copy_min_confidence: float = 0.25,
        same_word_penalty: int = 50000,
        max_closed_class_run: int = 2,
        ising_enabled: bool = True,
    ):
        self.vocab = vocab
        self.ngram_index = ngram_index
        self.J = J
        self.h = h
        self.types = types
        self.vocab_size = len(vocab)
        self.window = 5

        self.recall_scale = recall_scale
        self.pmi_weight = pmi_weight
        self.field_weight = field_weight
        self.ising_enabled = ising_enabled

        self.type_sampler = IntegerBoltzmannSampler(beta=beta_type, max_delta=500)
        self.word_sampler = IntegerBoltzmannSampler(beta=beta_word, max_delta=500)

        self.copy_enabled = copy_enabled
        self.copy_min_context = copy_min_context
        self.copy_min_confidence = copy_min_confidence
        self.same_word_penalty = same_word_penalty
        self.max_closed_class_run = max_closed_class_run

        # Build type-word index
        self.type_words: Dict[int, List[int]] = {}
        for t in range(N_POS):
            col = types.I_emit[:, t]
            self.type_words[t] = [int(i) for i in range(len(col)) if col[i] > 0]

        # Pre-compute allowed transitions from grammar
        self.allowed_transitions: Set[Tuple[int, int]] = set()
        for t1 in range(N_POS):
            for t2 in range(N_POS):
                penalty = types.compute_grammar_penalty([t1], 0, t2)
                if penalty < 500:
                    self.allowed_transitions.add((t1, t2))

        # Diagnostics
        self._stats = {
            'total_positions': 0, 'recall_hit': 0, 'copy_used': 0,
            'pmi_only': 0, 'same_word_blocked': 0, 'closed_loop_blocked': 0,
        }

    def _get_word_type(self, word_idx: int) -> int:
        """Get primary POS type for a word."""
        if word_idx in self.types.allowed_types and self.types.allowed_types[word_idx]:
            return max(
                self.types.allowed_types[word_idx],
                key=lambda t: int(self.types.I_emit[word_idx, t])
            )
        return POS2IDX["X"]

    def _get_valid_next_types(self, prev_type: int, types_history: List[int]) -> List[int]:
        """Get valid next POS types with hard constraints + anti-loop."""
        valid = [t for t in range(N_POS) if (prev_type, t) in self.allowed_transitions]
        if not valid:
            valid = list(range(N_POS))

        # Hard type constraints (e.g. PART -> VERB)
        if prev_type in self.HARD_TYPE_CONSTRAINTS:
            constrained = self.HARD_TYPE_CONSTRAINTS[prev_type]
            constrained_valid = [t for t in valid if t in constrained]
            if constrained_valid:
                valid = constrained_valid

        # Closed-class anti-loop
        closed_run = 0
        for t in reversed(types_history):
            if t in self.CLOSED_CLASS:
                closed_run += 1
            else:
                break
        if closed_run >= self.max_closed_class_run:
            open_types = [t for t in valid if t not in self.CLOSED_CLASS]
            if open_types:
                valid = open_types
                self._stats['closed_loop_blocked'] += 1

        return valid

    def _compute_type_energy(self, pos: int, type_idx: int, types_history: List[int]) -> int:
        """Compute energy for a POS type at position pos. Pure integer."""
        energy = 0
        types_for_check = list(types_history) + [type_idx]
        penalty = self.types.compute_grammar_penalty(
            types_for_check, len(types_history), type_idx
        )
        energy += penalty
        if len(types_history) > 0 and type_idx == types_history[-1]:
            if type_idx not in (POS2IDX['NOUN'], POS2IDX['X']):
                energy += 50
        return energy

    def _compute_word_energy(
        self, pos: int, candidate_words: np.ndarray, word_type: int,
        context_words: List[int], context_types: List[int], recall_hit: bool,
    ) -> np.ndarray:
        """
        Compute energy for candidate words.

        E(w) = -recall_bonus(w)          [PRIMARY: n-gram match signal]
             - pmi_coupling(w, ctx)       [SECONDARY: Ising PMI signal]
             - field(w)                   [TERTIARY: unigram frequency]
             + penalties                  [HARD: grammar, anti-repetition]

        All integer arithmetic.
        """
        n_candidates = len(candidate_words)
        energies = np.zeros(n_candidates, dtype=np.int64)

        # === RECALL BONUS (primary signal) ===
        recall_bonuses = self.ngram_index.get_recall_bonus(
            context_words=context_words,
            candidate_words=candidate_words,
            recall_scale=self.recall_scale,
            context_weight_factor=4,
            longest_only=True,
        )
        energies -= recall_bonuses

        # === PMI COUPLING (Ising model — secondary signal) ===
        if self.ising_enabled and len(context_words) > 0:
            context_start = max(0, len(context_words) - self.window)
            ctx = context_words[context_start:]
            if ctx:
                ctx_arr = np.array(ctx, dtype=np.int64)
                coupling_block = self.J[np.ix_(candidate_words, ctx_arr)]
                coupling_sums = coupling_block.sum(axis=1)
                if recall_hit and recall_bonuses.max() > 0:
                    energies -= (coupling_sums * self.pmi_weight) // 10
                else:
                    energies -= coupling_sums * self.pmi_weight

        # === LOCAL FIELD (unigram — tertiary signal) ===
        field_vals = self.h[candidate_words] * self.field_weight
        if recall_hit and recall_bonuses.max() > 0:
            energies -= (field_vals * 1) // 10
        else:
            energies -= field_vals

        # === TYPE COMPATIBILITY (hard constraint) ===
        for i, w in enumerate(candidate_words):
            w_int = int(w)
            if w_int < self.types.I_emit.shape[0]:
                if int(self.types.I_emit[w_int, word_type]) <= 0:
                    energies[i] += 500

        # === SAME-WORD PENALTY ===
        if len(context_words) >= 1:
            prev_word = context_words[-1]
            for i, w in enumerate(candidate_words):
                if int(w) == prev_word:
                    energies[i] += self.same_word_penalty

        # === CLOSED-CLASS DOUBLE PENALTY ===
        if word_type in self.CLOSED_CLASS and len(context_types) >= 1:
            prev_type = context_types[-1]
            if word_type == POS2IDX["DET"] and prev_type == POS2IDX["DET"]:
                energies += 50000
            elif word_type == POS2IDX["PREP"] and prev_type == POS2IDX["PREP"]:
                energies += 50000

        # === REPETITION PENALTY (recent context) ===
        if len(context_words) > 0:
            recent = set(context_words[-5:])
            for i, w in enumerate(candidate_words):
                if int(w) in recent:
                    energies[i] += 200

        return energies

    def generate(self, prompt: str = "the", length: int = 20) -> Dict:
        """
        Generate text autoregressively.

        At each position:
          1. Choose POS type (grammar + hard constraints)
          2. Check n-gram recall for type override
          3. Check copy mechanism
          4. Compute energy (recall + PMI + field + penalties)
          5. Integer Boltzmann sample

        All energy computation and sampling is integer-only.
        """
        # Resolve prompt
        prompt_idx = self.vocab.word2idx.get(prompt)
        if prompt_idx is None:
            prompt_idx = self.vocab.word2idx.get(prompt.lower())
        if prompt_idx is None:
            prompt_idx = 4

        prompt_type = self._get_word_type(prompt_idx)
        words = [prompt_idx]
        types = [prompt_type]
        consecutive_copies = 0
        diagnostics = []

        for pos in range(1, length):
            # === STEP 1: Choose POS type ===
            valid_types = self._get_valid_next_types(types[-1], types)

            # Check if recall suggests a type override
            recall_type_override = None
            if len(words) >= 2:
                recall_matches = self.ngram_index.lookup(words)
                if recall_matches:
                    best_k = max(recall_matches.keys())
                    best_conts = recall_matches[best_k]
                    if best_k >= 2 and best_conts:
                        best_word, best_count, best_total = best_conts[0]
                        if best_count * 3 >= best_total:
                            recall_type = self._get_word_type(best_word)
                            if recall_type in valid_types:
                                recall_type_override = recall_type

            if recall_type_override is not None:
                chosen_type = recall_type_override
            else:
                type_energies = np.array([
                    self._compute_type_energy(pos, t, types)
                    for t in valid_types
                ], dtype=np.int64)
                type_idx = self.type_sampler.sample(type_energies)
                chosen_type = valid_types[type_idx]

            # === STEP 2: Check copy mechanism ===
            copy_word = None
            if self.copy_enabled and len(words) >= self.copy_min_context:
                copy_candidate = self.ngram_index.get_best_copy_candidate(
                    context_words=words,
                    min_context_length=self.copy_min_context,
                    min_confidence=self.copy_min_confidence,
                )
                if copy_candidate is not None:
                    copy_word_idx, _, _ = copy_candidate
                    if copy_word_idx < self.types.I_emit.shape[0]:
                        if int(self.types.I_emit[copy_word_idx, chosen_type]) > 0:
                            if len(words) >= 1 and copy_word_idx == words[-1]:
                                copy_word_idx = None
                            elif consecutive_copies >= 6:
                                copy_word_idx = None
                            else:
                                copy_word = copy_word_idx
                                consecutive_copies += 1
                                self._stats['copy_used'] += 1

            if copy_word is None:
                consecutive_copies = 0

            # === STEP 3: Choose word ===
            candidate_list = self.type_words.get(chosen_type, [])
            if not candidate_list:
                candidate_list = list(range(min(200, self.vocab_size)))
            candidate_words = np.array(candidate_list, dtype=np.int64)

            # Top-k filtering by field strength
            if len(candidate_words) > 300:
                field_vals = self.h[candidate_words]
                top_k = np.argsort(field_vals)[-300:]
                candidate_words = candidate_words[top_k]

            # Check recall availability
            recall_matches = self.ngram_index.lookup(words)
            recall_hit = bool(recall_matches)

            # Compute energy (integer-only)
            word_energies = self._compute_word_energy(
                pos, candidate_words, chosen_type,
                words, types, recall_hit
            )

            # Integer Boltzmann sample
            if copy_word is not None:
                chosen_word = copy_word
            else:
                word_idx = self.word_sampler.sample(word_energies)
                chosen_word = int(candidate_words[word_idx])

            words.append(chosen_word)
            types.append(chosen_type)

            self._stats['total_positions'] += 1
            if recall_hit:
                self._stats['recall_hit'] += 1
            else:
                self._stats['pmi_only'] += 1

            diagnostics.append({
                'pos': pos,
                'type': IDX2POS.get(chosen_type, "UNK"),
                'word': self.vocab.idx2word.get(chosen_word, "<UNK>"),
                'copy': copy_word is not None,
                'recall_hit': recall_hit,
            })

        text = self.vocab.decode(words)
        type_names = [IDX2POS.get(t, "UNK") for t in types]

        return {
            'text': text,
            'words': words,
            'types': types,
            'type_names': type_names,
            'diagnostics': diagnostics,
        }

    def generate_raw(self, length: int = 20) -> Tuple[List[int], List[int]]:
        """Generate with a random prompt."""
        start_idx = np.random.randint(4, min(54, self.vocab_size))
        prompt = self.vocab.idx2word.get(start_idx, "the")
        result = self.generate(prompt=prompt, length=length)
        return result['words'], result['types']

    def get_stats(self) -> Dict:
        """Get generation statistics."""
        stats = self._stats.copy()
        total = max(1, stats['total_positions'])
        stats['recall_hit_rate'] = stats['recall_hit'] / total
        stats['copy_rate'] = stats['copy_used'] / total
        stats['pmi_only_rate'] = stats['pmi_only'] / total
        stats['ising_enabled'] = self.ising_enabled
        return stats


# ===========================================================================
# MODEL: Training + Generation Pipeline
# ===========================================================================

class IsingLMModel:
    """
    Complete model: training pipeline + generation.

    Training:
      1. Load corpus
      2. Build vocabulary
      3. Build POS type system
      4. Compute PMI couplings
      5. Build n-gram index
      6. Create generator(s)

    Generation:
      - With Ising (default)
      - Without Ising (ablation baseline)
    """

    def __init__(
        self,
        vocab_min_freq: int = 5,
        vocab_max_size: int = 3000,
        ngram_max_n: int = 5,
        ngram_min_count: int = 1,
        pmi_window: int = 5,
        pmi_min_count: int = 2,
        pmi_cap: int = 10,
        recall_scale: int = 300,
        pmi_weight: int = 5,
        field_weight: int = 1,
        beta_type: float = 0.01,
        beta_word: float = 0.1,
        copy_enabled: bool = True,
        copy_min_context: int = 3,
        copy_min_confidence: float = 0.4,
        same_word_penalty: int = 50000,
        max_closed_class_run: int = 2,
        ising_enabled: bool = True,
    ):
        self.vocab_min_freq = vocab_min_freq
        self.vocab_max_size = vocab_max_size
        self.ngram_max_n = ngram_max_n
        self.ngram_min_count = ngram_min_count
        self.pmi_window = pmi_window
        self.pmi_min_count = pmi_min_count
        self.pmi_cap = pmi_cap
        self.recall_scale = recall_scale
        self.pmi_weight = pmi_weight
        self.field_weight = field_weight
        self.beta_type = beta_type
        self.beta_word = beta_word
        self.copy_enabled = copy_enabled
        self.copy_min_context = copy_min_context
        self.copy_min_confidence = copy_min_confidence
        self.same_word_penalty = same_word_penalty
        self.max_closed_class_run = max_closed_class_run
        self.ising_enabled = ising_enabled

        self.vocab: Optional[Vocabulary] = None
        self.types: Optional[POSTypeSystem] = None
        self.J: Optional[np.ndarray] = None
        self.h: Optional[np.ndarray] = None
        self.ngram_index: Optional[NGramIndex] = None
        self.generator: Optional[IsingLM] = None
        self.baseline_generator: Optional[IsingLM] = None
        self.sequences: Optional[List[List[int]]] = None

    def train(self, n_samples: int = 20000) -> "IsingLMModel":
        """Train the model from FineWeb-Edu corpus."""
        print("=" * 70)
        print("ISING-ENHANCED N-GRAM LANGUAGE MODEL — TRAINING")
        print("=" * 70)
        print(f"\n  Architecture: N-gram (primary) + Ising PMI (secondary)")
        print(f"  Integer-only hot path: Lookup-table Boltzmann (NO np.exp)")
        print(f"  Ising enabled: {self.ising_enabled}")
        print()

        t0 = time.time()

        # Step 1: Load corpus
        print("[1/5] Loading corpus...")
        texts = load_fineweb_edu(n_samples=n_samples)
        print(f"  Loaded {len(texts)} texts ({time.time()-t0:.1f}s)")

        # Step 2: Build vocabulary
        print("\n[2/5] Building vocabulary...")
        self.vocab = Vocabulary(
            min_freq=self.vocab_min_freq,
            max_size=self.vocab_max_size,
        )
        self.vocab.build(texts)
        print(f"  Vocabulary: {len(self.vocab)} words")

        # Step 3: Build POS type system
        print("\n[3/5] Building POS type system...")
        self.types = POSTypeSystem(
            vocab_size=len(self.vocab),
            window=self.pmi_window,
        )
        self.types.build_from_vocabulary(self.vocab.word2idx, self.vocab.idx2word)
        self.types.build_grammar_penalties(penalty_strength=60)
        sequences = tokenize_texts(texts, self.vocab)
        sequences = truncate_sequences(sequences, max_len=20)
        self.sequences = sequences
        self.types.compute_type_couplings(sequences, self.vocab.idx2word)
        n_typed = sum(1 for w in range(len(self.vocab)) if w in self.types.allowed_types)
        print(f"  POS system built: {N_POS} types, {n_typed} words typed")

        # Step 4: Compute PMI couplings
        print("\n[4/5] Computing PMI couplings...")
        self.J, self.h = compute_pmi_couplings(
            sequences, len(self.vocab),
            window=self.pmi_window,
            min_count=self.pmi_min_count,
            pmi_cap=self.pmi_cap,
        )

        # Step 5: Build n-gram index
        print("\n[5/5] Building n-gram index...")
        self.ngram_index = NGramIndex(
            max_n=self.ngram_max_n,
            min_count=self.ngram_min_count,
        )
        self.ngram_index.build(sequences)

        # Build generators
        print("\nBuilding generators...")
        self._build_generators()

        t_total = time.time() - t0
        print(f"\nTraining complete: {t_total:.1f}s")
        return self

    def _build_generators(self):
        """Build Ising and ablation generators."""
        gen_kwargs = dict(
            vocab=self.vocab,
            ngram_index=self.ngram_index,
            J=self.J, h=self.h, types=self.types,
            recall_scale=self.recall_scale,
            field_weight=self.field_weight,
            beta_type=self.beta_type,
            beta_word=self.beta_word,
            copy_enabled=self.copy_enabled,
            copy_min_context=self.copy_min_context,
            copy_min_confidence=self.copy_min_confidence,
            same_word_penalty=self.same_word_penalty,
            max_closed_class_run=self.max_closed_class_run,
        )

        # Main generator (with Ising)
        self.generator = IsingLM(
            **gen_kwargs,
            pmi_weight=self.pmi_weight,
            ising_enabled=self.ising_enabled,
        )

        # Ablation baseline (without Ising)
        self.baseline_generator = IsingLM(
            **gen_kwargs,
            pmi_weight=0,
            ising_enabled=False,
        )

    def generate_with_trace(self, prompt: str = "the", length: int = 20) -> Dict:
        """Generate text with full diagnostics."""
        if self.generator is None:
            self._build_generators()
        result = self.generator.generate(prompt=prompt, length=length)
        result['stats'] = self.generator.get_stats()
        return result

    def generate_raw(self, length: int = 20) -> Tuple[List[int], List[int]]:
        """Generate with random prompt."""
        if self.generator is None:
            self._build_generators()
        return self.generator.generate_raw(length=length)

    def evaluate_grammar(self, words, types):
        """Evaluate grammar quality of a generated sequence."""
        n_det_noun = 0
        n_det_non_noun = 0
        n_repeated = 0
        n_prep_noun = 0
        n_prep_non_noun = 0

        for i in range(len(types) - 1):
            t1, t2 = types[i], types[i + 1]
            if t1 == POS2IDX["DET"]:
                if t2 in {POS2IDX["NOUN"], POS2IDX["PRON"], POS2IDX["NUM"]}:
                    n_det_noun += 1
                else:
                    n_det_non_noun += 1
            if t1 == POS2IDX["PREP"]:
                if t2 in {POS2IDX["NOUN"], POS2IDX["PRON"], POS2IDX["DET"]}:
                    n_prep_noun += 1
                else:
                    n_prep_non_noun += 1

        for i in range(len(words) - 1):
            if words[i] == words[i + 1] and words[i] >= 4:
                n_repeated += 1

        return {
            "det_noun": n_det_noun,
            "det_non_noun": n_det_non_noun,
            "prep_noun": n_prep_noun,
            "prep_non_noun": n_prep_non_noun,
            "repeated_words": n_repeated,
        }

    @property
    def vocab_size(self):
        return len(self.vocab) if self.vocab else 0
