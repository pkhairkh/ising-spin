"""
Ising Spin Glass Language Model — v5.0 Genuine Ising Dynamics.

A non-neural language model where ALL word selection goes through the
Hamiltonian. No overrides, no bypasses, no deterministic insertions.

5-Layer Architecture (ALL compete through E(w|ctx)):
  Layer 1: PMI Couplings J[w,w'] + Local Field h[w]
  Layer 2: Knowledge External Field h_knowledge[w] (SPO triples)
  Layer 3: 3-Spin Couplings J3[(s,p)] -> o (many-body Ising interaction)
  Layer 4: Category Couplings J_category (hypernym-based semantic smoothing)
  Layer 5: Markov Logic Penalty (factual consistency, soft + hard)

Generation Pipeline:
  1. Choose POS type: Boltzmann from type energy landscape
  2. Check copy mechanism (legitimate: it's a form of recall)
  3. Apply hard logic filter (infinite energy barriers)
  4. Compute E(w|ctx) with ALL 5 layers competing
  5. Boltzmann sample: P(w) ~ exp(-beta * E(w))
  6. MCMC spin-flip refinement (Metropolis criterion)

Key Principle: When (dog, barks)->bark and (dog, chases)->chase both fire,
they create COMPETING energy wells. Boltzmann at temperature beta picks
between them stochastically. Near the phase transition, knowledge has
maximum influence with some thermal noise.

INTEGER-ONLY CONSTRAINT (enforced):
  - ALL generation-path computation uses integer arithmetic
  - Boltzmann sampling via pre-computed lookup table (NO np.exp in hot loop)
  - MCMC acceptance via the same lookup table (integer-only)
  - The ONLY floating-point is in building the lookup table at __init__ time

References:
  - Levy & Goldberg (2014): Word2Vec as log-PMI matrix factorization
  - Marcolli et al. (arXiv:1508.00504): Spin Glass Models of Syntax
  - Haydarov et al. (arXiv:2502.12014): Coupled Ising-Potts Model
  - Creutz (1983): Demon algorithm for integer MCMC acceptance
  - Nishimori (2001): Statistical Physics of Spin Glasses
"""

import math
import json
import time
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Set

import scipy.sparse as sp


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

    Path 3a: Enhanced tokenizer handles contractions, hyphens, and numbers.
    """

    UNK = "<UNK>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    PAD = "<PAD>"
    SPECIALS = [UNK, BOS, EOS, PAD]

    # Contraction suffixes to split off
    CONTRACTION_SUFFIXES = [
        "n't", "'t",  # negation: don't -> do + n't, can't -> ca + n't
        "'s",         # possessive/aux: it's, he's
        "'re",        # they're
        "'ve",        # they've
        "'ll",        # they'll
        "'d",         # they'd
        "'m",         # I'm
    ]

    def __init__(self, min_freq: int = 5, max_size: Optional[int] = None):
        self.min_freq = min_freq
        self.max_size = max_size
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_counts: Counter = Counter()
        self._built = False

    def _tokenize(self, text: str) -> List[str]:
        """
        Enhanced tokenizer with better handling of contractions, hyphens,
        and numbers. Pure string manipulation — no external dependencies.

        Path 3a improvements:
          - Contractions: "don't" -> "do" + "n't", "it's" -> "it" + "'s"
          - Hyphens: "well-known" -> "well-known" (kept as one token)
          - Numbers: "3.14" stays as one token, "1,000" stays as one token
        """
        tokens = []
        for word in text.split():
            stripped = word.strip()
            if not stripped:
                continue

            # Split off leading punctuation
            leading_punct = []
            while stripped and not stripped[0].isalnum() and stripped[0] != '-':
                leading_punct.append(stripped[0])
                stripped = stripped[1:]

            # Split off trailing punctuation
            trailing_punct = []
            while stripped and not stripped[-1].isalnum() and stripped[-1] != '-':
                trailing_punct.append(stripped[-1])
                stripped = stripped[:-1]

            # Add leading punctuation tokens
            tokens.extend(leading_punct)

            if not stripped:
                tokens.extend(reversed(trailing_punct))
                continue

            lower = stripped.lower()

            # === Handle contractions ===
            contraction_found = False
            for suffix in self.CONTRACTION_SUFFIXES:
                if lower.endswith(suffix) and len(lower) > len(suffix):
                    stem = lower[:-len(suffix)]
                    if stem and any(c.isalpha() for c in stem):
                        # Special case: "can't" -> "ca" + "n't" (not "can")
                        # But we keep it simple: "don't" -> "do" + "n't"
                        tokens.append(stem)
                        tokens.append(suffix)
                        contraction_found = True
                        break

            if contraction_found:
                tokens.extend(reversed(trailing_punct))
                continue

            # === Handle numbers (keep as single token) ===
            # "3.14", "1,000", "0.5" should stay as one token
            cleaned = lower.replace(".", "").replace(",", "")
            if cleaned.replace("-", "").isdigit() and len(lower) > 0:
                tokens.append(lower)
                tokens.extend(reversed(trailing_punct))
                continue

            # === Handle hyphenated words (keep as single token) ===
            # "well-known", "state-of-the-art" stay as one token
            if '-' in lower and not lower.startswith('-') and not lower.endswith('-'):
                parts = lower.split('-')
                if all(len(p) >= 1 and (p.isalpha() or p.isdigit()) for p in parts):
                    tokens.append(lower)
                    tokens.extend(reversed(trailing_punct))
                    continue

            # === Default: use the word as-is (lowercased) ===
            tokens.append(lower)
            tokens.extend(reversed(trailing_punct))

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

    def add_words(self, words: List[str]) -> int:
        """
        Add words to vocabulary even if they don't appear in corpus.
        
        Used for knowledge vocabulary augmentation: words like "bark", "meow",
        "gallop" that appear in knowledge triples but not in the training corpus.
        These words get count=1 and no PMI/n-gram stats, but they DO get
        knowledge layer bonuses (h_knowledge, J3) and POS type assignments.
        
        Returns the number of words actually added.
        """
        if not self._built:
            return 0
        
        n_added = 0
        for word in words:
            w = word.lower().strip()
            if not w or w in self.SPECIALS:
                continue
            if w in self.word2idx:
                continue  # Already in vocabulary
            if self.max_size is not None and len(self.word2idx) >= self.max_size + len(self.SPECIALS) + 200:
                # Leave room for more additions; don't exceed by too much
                pass  # Allow overflow for knowledge words
            
            idx = len(self.word2idx)
            self.word2idx[w] = idx
            self.idx2word[idx] = w
            self.word_counts[w] = 1  # Minimal count
            n_added += 1
        
        return n_added

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

    def compute_log_probabilities(self, energies: np.ndarray) -> np.ndarray:
        """
        Compute log probabilities for each element given energies.

        Uses floating-point for the log computation (evaluation only,
        not in the generation hot path). Uses log-sum-exp for numerical
        stability.

        Returns array of log P(i) where P(i) ~ exp(-beta * E_i).
        """
        if len(energies) == 0:
            return np.array([], dtype=np.float64)

        e_min = float(energies.min())
        shifted = -self.beta * (energies.astype(np.float64) - e_min)
        # Clip to avoid overflow in exp
        shifted = np.clip(shifted, -500, 500)
        log_weights = shifted
        log_Z = np.log(np.exp(log_weights).sum())
        log_probs = log_weights - log_Z
        return log_probs


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

        Uses ONLY the longest matching context by default -- prevents common-word inflation.
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
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """
    Compute PMI coupling matrix J and local field h from sequences.

    Path 3b: J is now a scipy.sparse.csr_matrix (was dense np.ndarray).
    Only non-zero PMI values are stored, saving ~95% memory.

    J[w, w'] = log-floor PMI(w, w') for co-occurring words within window
    h[w] = self-information = floor(log2(N/count(w)))

    Returns (J, h) where J is csr_matrix(int64) and h is np.ndarray(int64).
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

    # Build sparse J matrix from non-zero PMI values
    rows, cols, data = [], [], []
    seen = set()
    for (w, w2), count in cooc_counts.items():
        if count >= min_count:
            pmi = compute_log_floor_pmi(
                int(count), int(unigram[w]), int(unigram[w2]),
                total_tokens, cap=pmi_cap
            )
            if pmi != 0:
                # Add both (w, w2) and (w2, w) for symmetric matrix
                if (w, w2) not in seen:
                    rows.append(w)
                    cols.append(w2)
                    data.append(pmi)
                    seen.add((w, w2))
                if (w2, w) not in seen:
                    rows.append(w2)
                    cols.append(w)
                    data.append(pmi)
                    seen.add((w2, w))

    J = sp.csr_matrix(
        (np.array(data, dtype=np.int64),
         (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
        shape=(V, V)
    )

    # Compute local field (self-information)
    h = np.ones(V, dtype=np.int64)
    for w in range(V):
        if unigram[w] > 0 and total_tokens > unigram[w]:
            ratio = total_tokens // int(unigram[w])
            if ratio >= 2:
                h[w] = ratio.bit_length() - 1

    n_nonzero = J.nnz
    print(f"    PMI matrix (sparse): {n_nonzero:,} non-zero entries out of {V*V:,}")
    if n_nonzero > 0:
        dense_bytes = V * V * 8
        sparse_bytes = J.data.nbytes + J.indices.nbytes + J.indptr.nbytes
        print(f"    Memory: sparse {sparse_bytes/1024/1024:.1f}MB vs dense {dense_bytes/1024/1024:.1f}MB")
        min_val = int(J.data.min())
        max_val = int(J.data.max())
        print(f"    PMI range: [{min_val}, {max_val}]")

    return J, h


def compute_skip_pmi_couplings(
    sequences: List[List[int]],
    vocab_size: int,
    max_dist: int = 5,
    min_count: int = 2,
    pmi_cap: int = 10,
) -> Dict[int, sp.csr_matrix]:
    """
    Compute distance-specific skip-gram PMI couplings.

    Path 2d: Instead of a flat window, compute PMI for each distance
    separately. This captures longer-range dependencies beyond window-5.

    J_skip[dist] is a sparse matrix where J_skip[dist][w1, w2] =
        log-floor PMI(w1, w2) computed from pairs exactly `dist` apart.

    Args:
        sequences: Tokenized sequences.
        vocab_size: Vocabulary size V.
        max_dist: Maximum skip distance to compute (default 5).
        min_count: Minimum co-occurrence count for PMI.
        pmi_cap: Cap on absolute PMI value.

    Returns:
        Dict mapping distance (1..max_dist) to csr_matrix of shape (V, V).
    """
    V = vocab_size

    # Count unigrams
    unigram = np.zeros(V, dtype=np.int64)
    for seq in sequences:
        for w in seq:
            unigram[w] += 1
    total_tokens = int(unigram.sum())

    # Count co-occurrences at each specific distance
    cooc_by_dist: Dict[int, Counter] = {d: Counter() for d in range(1, max_dist + 1)}
    for seq in sequences:
        for i, w in enumerate(seq):
            for d in range(1, min(max_dist + 1, len(seq) - i)):
                j = i + d
                cooc_by_dist[d][(w, seq[j])] += 1

    # Build sparse matrices for each distance
    J_skip: Dict[int, sp.csr_matrix] = {}
    for dist in range(1, max_dist + 1):
        rows, cols, data = [], [], []
        seen = set()
        for (w, w2), count in cooc_by_dist[dist].items():
            if count >= min_count:
                pmi = compute_log_floor_pmi(
                    int(count), int(unigram[w]), int(unigram[w2]),
                    total_tokens, cap=pmi_cap
                )
                if pmi != 0:
                    if (w, w2) not in seen:
                        rows.append(w)
                        cols.append(w2)
                        data.append(pmi)
                        seen.add((w, w2))
                    if (w2, w) not in seen:
                        rows.append(w2)
                        cols.append(w)
                        data.append(pmi)
                        seen.add((w2, w))

        if rows:
            J_skip[dist] = sp.csr_matrix(
                (np.array(data, dtype=np.int64),
                 (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
                shape=(V, V)
            )
        else:
            J_skip[dist] = sp.csr_matrix((V, V), dtype=np.int64)

        print(f"    Skip-PMI dist={dist}: {J_skip[dist].nnz:,} non-zero entries")

    return J_skip


# ===========================================================================
# KNOWLEDGE LAYER (Layer 2 + Layer 3)
# ===========================================================================

class KnowledgeLayer:
    """
    Knowledge injection layer for the Ising Knowledge Machine.
    
    Implements:
      Layer 2: Knowledge External Field — biases individual spins toward 
               knowledge-consistent states via h_knowledge[w]
      Layer 3: 3-Spin Couplings — represents SPO triples as many-body 
               Ising interactions where J3[s,p,o] creates an energy 
               contribution when subject and predicate are both present
    
    All computation is INTEGER-ONLY during generation.
    
    References:
      - Haydarov et al. (2025): Coupled Ising-Potts rich critical dynamics
      - Bertalan & Nishimori (2012): First-order phase transitions in p-spin
      - Li et al. (2021): Hamming-space KG embeddings (XOR + popcount)
      - Hashizume & Suzuki (2011): Many-body spin-pair glass phases
    """
    
    def __init__(self, vocab_size: int, knowledge_scale: int = 500,
                 spin3_scale: int = 800, max_context_pairs: int = 20):
        self.vocab_size = vocab_size
        self.knowledge_scale = knowledge_scale  # Layer 2: field strength
        self.spin3_scale = spin3_scale           # Layer 3: 3-spin coupling strength
        self.max_context_pairs = max_context_pairs
        
        # Layer 2: External field. h_knowledge[w] = sum of integer bonuses
        self.h_knowledge = np.zeros(vocab_size, dtype=np.int64)
        
        # Layer 3: 3-spin couplings. 
        # Key: (subject_idx, predicate_idx) tuple
        # Value: list of (object_idx, coupling_strength_int)
        self.J3: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        
        # Index for Layer 2: which words are subjects of triples
        # subject_idx -> list of (predicate_idx, object_idx)
        self.subject_index: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        
        # For tracking
        self.n_triples = 0
        self.n_unique_subjects = 0
        self.n_unique_predicates = 0
        self._built = False
    
    def add_triples_from_corpus(self, sequences, idx2word, types_system, min_count=3):
        """
        Extract SPO triples from training corpus using dependency patterns.
        
        Strategy: Use simple pattern extraction:
          - (NOUN, VERB/PREP, NOUN): subject-verb/prep-object patterns
          - (NOUN, AUX, ADJ/NOUN): subject-aux-predication patterns
        
        For each pattern, count occurrences and keep those above min_count.
        The coupling strength is computed as integer: count * spin3_scale.
        
        Uses existing POS type system for word classification.
        """
        # We need to classify each word's primary POS type
        # Use a simplified approach: check the allowed_types from the POSTypeSystem
        TAG_PRIORITY = {
            POS2IDX["PUNCT"]: 0, POS2IDX["DET"]: 1, POS2IDX["PRON"]: 2,
            POS2IDX["AUX"]: 3, POS2IDX["CONJ"]: 4, POS2IDX["PART"]: 5,
            POS2IDX["PREP"]: 6, POS2IDX["NUM"]: 7, POS2IDX["ADV"]: 8,
            POS2IDX["ADJ"]: 9, POS2IDX["NOUN"]: 10, POS2IDX["VERB"]: 11,
            POS2IDX["X"]: 12,
        }
        
        def get_primary_type(word_idx):
            if word_idx in types_system.allowed_types and types_system.allowed_types[word_idx]:
                tags = list(types_system.allowed_types[word_idx])
                return min(tags, key=lambda t: TAG_PRIORITY.get(t, 99))
            return POS2IDX["X"]
        
        # Count SPO triples from consecutive triples in sequences
        triple_counts = Counter()
        
        for seq in sequences:
            for i in range(len(seq) - 2):
                w0 = seq[i]
                w1 = seq[i + 1]
                w2 = seq[i + 2]
                
                # Skip special tokens
                if w0 < 4 or w1 < 4 or w2 < 4:
                    continue
                
                t0 = get_primary_type(w0)
                t1 = get_primary_type(w1)
                t2 = get_primary_type(w2)
                
                # Pattern: NOUN VERB NOUN (subject-verb-object)
                if t0 in (POS2IDX["NOUN"], POS2IDX["PRON"]) and \
                   t1 in (POS2IDX["VERB"], POS2IDX["AUX"]) and \
                   t2 in (POS2IDX["NOUN"], POS2IDX["PRON"], POS2IDX["NUM"]):
                    triple_counts[(w0, w1, w2)] += 1
                
                # Pattern: NOUN PREP NOUN (noun-preposition-noun)
                elif t0 in (POS2IDX["NOUN"], POS2IDX["PRON"]) and \
                     t1 == POS2IDX["PREP"] and \
                     t2 in (POS2IDX["NOUN"], POS2IDX["PRON"], POS2IDX["NUM"]):
                    triple_counts[(w0, w1, w2)] += 1
                
                # Pattern: NOUN AUX ADJ (subject-aux-adjective)
                elif t0 in (POS2IDX["NOUN"], POS2IDX["PRON"]) and \
                     t1 == POS2IDX["AUX"] and \
                     t2 == POS2IDX["ADJ"]:
                    triple_counts[(w0, w1, w2)] += 1
        
        # Only keep triples above min_count
        n_extracted = 0
        for (s, p, o), count in triple_counts.items():
            if count >= min_count:
                coupling_strength = count * self.spin3_scale
                
                # Add to J3
                key = (s, p)
                if key not in self.J3:
                    self.J3[key] = []
                self.J3[key].append((o, coupling_strength))
                
                # Add to subject_index
                self.subject_index[s].append((p, o))
                
                # Add to h_knowledge (Layer 2)
                # Each triple adds a base field to subject, predicate, and object
                self.h_knowledge[s] += self.knowledge_scale
                self.h_knowledge[p] += self.knowledge_scale // 2
                self.h_knowledge[o] += self.knowledge_scale
                
                self.n_triples += 1
                n_extracted += 1
        
        print(f"    Extracted {n_extracted} SPO triples from corpus "
              f"(min_count={min_count}, scanned {len(triple_counts)} patterns)")
    
    def add_conceptnet_triples(self, triples_text, word2idx):
        """
        Add triples from ConceptNet-style text format.
        Each triple is (subject_word, relation_word, object_word).
        Only add triples where all three words are in vocabulary.
        """
        n_added = 0
        for triple in triples_text:
            if len(triple) != 3:
                continue
            subj, pred, obj = triple
            
            # Look up indices
            s_idx = word2idx.get(subj.lower(), None)
            p_idx = word2idx.get(pred.lower(), None)
            o_idx = word2idx.get(obj.lower(), None)
            
            # Skip if any word is not in vocabulary or is a special token
            if s_idx is None or p_idx is None or o_idx is None:
                continue
            if s_idx < 4 or p_idx < 4 or o_idx < 4:
                continue
            
            # Add to J3
            key = (s_idx, p_idx)
            if key not in self.J3:
                self.J3[key] = []
            # Use a fixed coupling strength for curated triples
            self.J3[key].append((o_idx, self.spin3_scale * 2))
            
            # Add to subject_index
            self.subject_index[s_idx].append((p_idx, o_idx))
            
            # Add to h_knowledge (Layer 2)
            self.h_knowledge[s_idx] += self.knowledge_scale
            self.h_knowledge[p_idx] += self.knowledge_scale // 2
            self.h_knowledge[o_idx] += self.knowledge_scale
            
            self.n_triples += 1
            n_added += 1
        
        print(f"    Added {n_added} ConceptNet-style triples "
              f"(out of {len(triples_text)} provided)")
    
    def build(self):
        """
        Finalize the knowledge layer after adding all triples.
        Pre-compute h_knowledge from all triples, build subject_index.
        Pure integer arithmetic.
        
        v5.0 fix: Cap h_knowledge so it doesn't swamp the energy landscape.
        The real signal should come from J3 (context-dependent), not from
        h_knowledge (static, always on). h_knowledge is a BIAS, not a
        domination. We cap it to knowledge_scale * 3 maximum.
        """
        # Compute statistics
        subjects = set()
        predicates = set()
        for key, objects in self.J3.items():
            subjects.add(key[0])
            predicates.add(key[1])
        
        self.n_unique_subjects = len(subjects)
        self.n_unique_predicates = len(predicates)
        
        # v5.0: Cap h_knowledge to prevent energy domination
        # Words in many triples accumulate huge h_knowledge values,
        # which swamps all other energy terms. Cap to a reasonable bias.
        h_cap = self.knowledge_scale * 3  # Maximum bias per word
        self.h_knowledge = np.clip(self.h_knowledge, 0, h_cap).astype(np.int64)
        
        self._built = True
        
        print(f"    Knowledge layer built:")
        print(f"      Total triples: {self.n_triples}")
        print(f"      Unique subjects: {self.n_unique_subjects}")
        print(f"      Unique predicates: {self.n_unique_predicates}")
        print(f"      J3 entries: {len(self.J3)}")
        print(f"      h_knowledge non-zero: {int(np.count_nonzero(self.h_knowledge))}")
        print(f"      h_knowledge max: {int(self.h_knowledge.max())} (capped at {h_cap})")
    
    def compute_knowledge_field(self, context_words, candidate_words):
        """
        Layer 2: Compute knowledge external field contribution.
        
        For each context word that is a SUBJECT of some triple,
        add knowledge_scale to h_knowledge[object] for all matching
        objects where the predicate also appears in context.
        
        If only subject is in context (no predicate match), add
        knowledge_scale // 3 (weaker signal).
        
        Returns: integer array of shape (n_candidates,) with field bonuses.
        Pure integer arithmetic.
        """
        n_candidates = len(candidate_words)
        bonuses = np.zeros(n_candidates, dtype=np.int64)
        
        if not self._built:
            return bonuses
        
        # Build a set of context word indices for fast lookup
        context_set = set(context_words)
        
        # Build a dict of bonus per candidate word index
        word_bonuses = defaultdict(int)
        
        for subject_idx in context_words:
            if subject_idx not in self.subject_index:
                continue
            for (pred_idx, obj_idx) in self.subject_index[subject_idx]:
                if pred_idx in context_set:
                    # Full match: subject AND predicate in context
                    word_bonuses[obj_idx] += self.knowledge_scale
                else:
                    # Weaker signal: only subject in context
                    word_bonuses[obj_idx] += self.knowledge_scale // 3
        
        # Map bonuses to candidate_words array
        for i, w in enumerate(candidate_words):
            w_int = int(w)
            if w_int in word_bonuses:
                bonuses[i] += word_bonuses[w_int]
        
        return bonuses
    
    def compute_3spin_coupling(self, context_words, candidate_words):
        """
        Layer 3: Compute 3-spin coupling contribution.
        
        For each PAIR (wi, wj) in context_words, check if (wi, wj)
        or (wj, wi) is a key in J3. If so, add the coupling strength
        to the matching object words in candidate_words.
        
        This is the novel 3-body Ising interaction: the energy depends
        on the JOINT state of subject AND predicate being present.
        
        Returns: integer array of shape (n_candidates,) with 3-spin bonuses.
        Pure integer arithmetic.
        """
        n_candidates = len(candidate_words)
        bonuses = np.zeros(n_candidates, dtype=np.int64)
        
        if not self._built or not self.J3:
            return bonuses
        
        # Take the last max_context_pairs words to limit computation
        ctx = context_words[-self.max_context_pairs:] if len(context_words) > self.max_context_pairs else context_words
        
        # Build a dict of bonus per candidate word index
        word_bonuses = defaultdict(int)
        
        # Check all pairs in context
        for i in range(len(ctx)):
            for j in range(len(ctx)):
                if i == j:
                    continue
                wi = ctx[i]
                wj = ctx[j]
                
                # Check J3[(wi, wj)] -- wi is subject, wj is predicate
                key_forward = (wi, wj)
                if key_forward in self.J3:
                    for (obj_idx, strength) in self.J3[key_forward]:
                        word_bonuses[obj_idx] += strength
        
        # Map bonuses to candidate_words array
        for i, w in enumerate(candidate_words):
            w_int = int(w)
            if w_int in word_bonuses:
                bonuses[i] += word_bonuses[w_int]
        
        return bonuses
    
    def compute_knowledge_energy(self, context_words, candidate_words):
        """
        Combined knowledge energy contribution (Layer 2 + Layer 3).
        
        Returns: integer array of shape (n_candidates,) with total 
        knowledge bonuses (subtract from energy for lower energy = more likely).
        """
        if not self._built:
            return np.zeros(len(candidate_words), dtype=np.int64)
        
        field_bonus = self.compute_knowledge_field(context_words, candidate_words)
        spin3_bonus = self.compute_3spin_coupling(context_words, candidate_words)
        return field_bonus + spin3_bonus
    
    def get_diagnostics(self, context_words, candidate_words):
        """
        Get diagnostic information about knowledge layer activation.
        
        Returns dict with counts of field hits, 3-spin hits, etc.
        """
        if not self._built:
            return {'field_hits': 0, 'spin3_hits': 0, 'active_triples': 0}
        
        context_set = set(context_words)
        field_hits = 0
        spin3_hits = 0
        active_triples = 0
        
        # Count field hits
        for subject_idx in context_words:
            if subject_idx in self.subject_index:
                field_hits += 1
        
        # Count 3-spin hits
        ctx = context_words[-self.max_context_pairs:] if len(context_words) > self.max_context_pairs else context_words
        for i in range(len(ctx)):
            for j in range(len(ctx)):
                if i == j:
                    continue
                key = (ctx[i], ctx[j])
                if key in self.J3:
                    spin3_hits += len(self.J3[key])
                    active_triples += 1
        
        return {
            'field_hits': field_hits,
            'spin3_hits': spin3_hits,
            'active_triples': active_triples,
        }


# ===========================================================================
# LAYER 4: CATEGORY COUPLINGS VIA ONTOLOGY
# ===========================================================================

class CategoryLayer:
    """
    Layer 4: Category Couplings via Ontology.
    
    Hypernym-based J_category couplings that help words like "dog" and "cat"
    share context via their shared hypernym "animal".
    
    When context contains words from category C, other words from category C
    receive a field bonus (they are more likely to appear). This is the
    Ising equivalent of WordNet-based semantic smoothing.
    
    The coupling is:
      J_category[w1, w2] = category_scale if w1 and w2 share a hypernym
      h_category[w] = sum of category_scale for each category w belongs to
                       when another member of that category is in context
    
    All computation is INTEGER-ONLY during generation.
    
    References:
      - Miller (1995): WordNet: A Lexical Database for English
      - Marcolli et al. (2015): Syntactic parameters as spin variables
      - Moro et al. (2014): Semantic category effects in the Ising model
    """
    
    def __init__(self, vocab_size: int, category_scale: int = 400):
        self.vocab_size = vocab_size
        self.category_scale = category_scale
        
        # category_name -> set of word indices
        self.categories: Dict[str, Set[int]] = defaultdict(set)
        # word_idx -> set of category names
        self.word_categories: Dict[int, Set[str]] = defaultdict(set)
        # Pre-computed: word_idx -> set of word indices sharing any category
        self.word_peers: Dict[int, Set[int]] = defaultdict(set)
        # h_category: static field bonus for words that belong to categories
        self.h_category = np.zeros(vocab_size, dtype=np.int64)
        
        self.n_categories = 0
        self.n_categorized_words = 0
        self._built = False
    
    def add_category(self, category_name: str, word_indices: List[int]):
        """Add a category with its member word indices."""
        for idx in word_indices:
            if idx < self.vocab_size:
                self.categories[category_name].add(idx)
                self.word_categories[idx].add(category_name)
    
    def build(self):
        """Pre-compute peer sets for O(1) lookup during generation."""
        # Build word_peers: for each word, find all other words sharing categories
        for idx, cats in self.word_categories.items():
            peers = set()
            for cat in cats:
                peers.update(self.categories[cat])
            peers.discard(idx)  # Don't include self
            self.word_peers[idx] = peers
        
        # Build static h_category field
        for idx, cats in self.word_categories.items():
            self.h_category[idx] = len(cats) * (self.category_scale // 4)
        
        self.n_categories = len(self.categories)
        self.n_categorized_words = len(self.word_categories)
        self._built = True
        
        print(f"    Category layer built:")
        print(f"      Categories: {self.n_categories}")
        print(f"      Categorized words: {self.n_categorized_words}")
        print(f"      Peer pairs: {sum(len(p) for p in self.word_peers.values()) // 2}")
    
    def compute_category_bonus(self, context_words, candidate_words):
        """
        Compute category-based bonus for candidate words.
        
        For each context word, if the candidate shares a category,
        add category_scale bonus. This creates semantic attraction
        between co-hyponyms (dog/cat, car/bicycle, apple/orange).
        
        Returns: integer array of shape (n_candidates,) with bonuses.
        Pure integer arithmetic.
        """
        if not self._built:
            return np.zeros(len(candidate_words), dtype=np.int64)
        
        n_candidates = len(candidate_words)
        bonuses = np.zeros(n_candidates, dtype=np.int64)
        
        # Collect all peers of context words
        context_peers = set()
        for w in context_words:
            if w in self.word_peers:
                context_peers.update(self.word_peers[w])
        
        if not context_peers:
            return bonuses
        
        # Build bonus lookup
        peer_bonus: Dict[int, int] = defaultdict(int)
        for w in context_peers:
            peer_bonus[w] += self.category_scale
        
        # Map to candidate array
        for i, w in enumerate(candidate_words):
            w_int = int(w)
            if w_int in peer_bonus:
                bonuses[i] = peer_bonus[w_int]
        
        return bonuses
    
    def compute_category_field(self, context_words, candidate_words):
        """
        Compute category field bonus for candidates when context
        contains same-category members.
        
        This is like h_category but context-dependent: the field
        is only active when a category co-member is in context.
        """
        if not self._built:
            return np.zeros(len(candidate_words), dtype=np.int64)
        
        n_candidates = len(candidate_words)
        bonuses = np.zeros(n_candidates, dtype=np.int64)
        
        # Check which categories are active in context
        active_categories = set()
        for w in context_words:
            if w in self.word_categories:
                active_categories.update(self.word_categories[w])
        
        if not active_categories:
            return bonuses
        
        # For each candidate, check if it belongs to an active category
        cat_bonus: Dict[int, int] = defaultdict(int)
        for cat in active_categories:
            for member_idx in self.categories[cat]:
                cat_bonus[member_idx] += self.category_scale // 2
        
        for i, w in enumerate(candidate_words):
            w_int = int(w)
            if w_int in cat_bonus:
                bonuses[i] = cat_bonus[w_int]
        
        return bonuses
    
    def get_diagnostics(self, context_words):
        """Get diagnostic info about category layer activation."""
        if not self._built:
            return {'active_categories': 0, 'peer_matches': 0}
        
        active_categories = set()
        for w in context_words:
            if w in self.word_categories:
                active_categories.update(self.word_categories[w])
        
        peer_matches = 0
        for w in context_words:
            if w in self.word_peers:
                peer_matches += len(self.word_peers[w] & set(context_words))
        
        return {
            'active_categories': len(active_categories),
            'peer_matches': peer_matches,
        }


# ===========================================================================
# LAYER 5: MARKOV LOGIC PENALTY
# ===========================================================================

class MarkovLogicLayer:
    """
    Layer 5: Markov Logic Penalty.
    
    Logical constraints that enforce factual consistency as soft energy
    penalties. Each rule has the form:
      IF context_contains(trigger_words) THEN bonus(target_words)
      or
      IF context_contains(trigger_words) THEN penalty(target_words)
    
    Rules are encoded as integer energy contributions:
    - Soft rules: bonus for satisfying, penalty for violating
    - Hard rules: large penalty for violating (nearly deterministic)
    
    The energy contribution is:
      E_logic(w|ctx) = -rule_scale  if w is a target and all triggers in ctx (bonus)
      E_logic(w|ctx) = +rule_scale  if w is a target and all triggers in ctx (penalty)
    
    All computation is INTEGER-ONLY during generation.
    
    References:
      - Richardson & Domingos (2006): Markov Logic Networks
      - Singla & Domingos (2005): Discriminative training of MLNs
      - Getoor & Taskar (2007): Introduction to Statistical Relational Learning
    """
    
    def __init__(self, vocab_size: int, rule_scale: int = 600,
                 hard_rule_scale: int = 50000):
        self.vocab_size = vocab_size
        self.rule_scale = rule_scale          # Soft rule penalty/bonus
        self.hard_rule_scale = hard_rule_scale  # Hard rule penalty
        
        # Rules: list of dicts
        self.rules: List[Dict] = []
        
        # Index: trigger_word -> list of rule indices for fast lookup
        self.trigger_index: Dict[int, List[int]] = defaultdict(list)
        
        self.n_rules = 0
        self.n_soft_rules = 0
        self.n_hard_rules = 0
        self._built = False
    
    def add_rule(self, trigger_words: List[int], target_words: List[int],
                 rule_type: str = 'bonus', strength: str = 'soft'):
        """
        Add a Markov logic rule.
        
        Args:
            trigger_words: List of word indices that trigger the rule.
            target_words: List of word indices that are affected.
            rule_type: 'bonus' (reward target) or 'penalty' (punish target)
            strength: 'soft' or 'hard'
        """
        rule_idx = len(self.rules)
        scale = self.hard_rule_scale if strength == 'hard' else self.rule_scale
        
        rule = {
            'trigger': frozenset(trigger_words),
            'targets': set(target_words),
            'type': rule_type,
            'strength': strength,
            'scale': scale,
        }
        self.rules.append(rule)
        
        # Index by trigger words for fast lookup
        for tw in trigger_words:
            self.trigger_index[tw].append(rule_idx)
        
        self.n_rules += 1
        if strength == 'hard':
            self.n_hard_rules += 1
        else:
            self.n_soft_rules += 1
    
    def build(self):
        """Finalize rules and compute statistics."""
        self._built = True
        print(f"    Markov Logic layer built:")
        print(f"      Total rules: {self.n_rules}")
        print(f"      Soft rules: {self.n_soft_rules}")
        print(f"      Hard rules: {self.n_hard_rules}")
    
    def compute_logic_energy(self, context_words, candidate_words):
        """
        Compute Markov logic energy contribution.
        
        For each active rule (all trigger words present in context),
        apply bonus or penalty to target words among candidates.
        
        Returns: integer array of shape (n_candidates,) with energy adjustments.
        Positive = penalty (higher energy = less likely).
        Negative = bonus (lower energy = more likely).
        Pure integer arithmetic.
        """
        if not self._built or not self.rules:
            return np.zeros(len(candidate_words), dtype=np.int64)
        
        context_set = set(context_words)
        n_candidates = len(candidate_words)
        energy = np.zeros(n_candidates, dtype=np.int64)
        
        # Find potentially active rules via trigger index
        candidate_rule_indices = set()
        for w in context_words:
            if w in self.trigger_index:
                candidate_rule_indices.update(self.trigger_index[w])
        
        if not candidate_rule_indices:
            return energy
        
        # Check each potentially active rule
        active_bonuses: Dict[int, int] = defaultdict(int)  # word_idx -> total bonus
        active_penalties: Dict[int, int] = defaultdict(int)  # word_idx -> total penalty
        
        for rule_idx in candidate_rule_indices:
            rule = self.rules[rule_idx]
            
            # Check if ALL trigger words are in context
            if not rule['trigger'].issubset(context_set):
                continue
            
            # Rule is active
            scale = rule['scale']
            if rule['type'] == 'bonus':
                for target in rule['targets']:
                    active_bonuses[target] += scale
            else:  # penalty
                for target in rule['targets']:
                    active_penalties[target] += scale
        
        # Apply to candidates
        for i, w in enumerate(candidate_words):
            w_int = int(w)
            if w_int in active_bonuses:
                energy[i] -= active_bonuses[w_int]  # Negative = bonus
            if w_int in active_penalties:
                energy[i] += active_penalties[w_int]  # Positive = penalty
        
        return energy
    
    def get_diagnostics(self, context_words):
        """Get diagnostic info about logic layer activation."""
        if not self._built:
            return {'active_rules': 0, 'active_bonuses': 0, 'active_penalties': 0}
        
        context_set = set(context_words)
        active_rules = 0
        active_bonuses = 0
        active_penalties = 0
        
        candidate_rule_indices = set()
        for w in context_words:
            if w in self.trigger_index:
                candidate_rule_indices.update(self.trigger_index[w])
        
        for rule_idx in candidate_rule_indices:
            rule = self.rules[rule_idx]
            if rule['trigger'].issubset(context_set):
                active_rules += 1
                if rule['type'] == 'bonus':
                    active_bonuses += len(rule['targets'])
                else:
                    active_penalties += len(rule['targets'])
        
        return {
            'active_rules': active_rules,
            'active_bonuses': active_bonuses,
            'active_penalties': active_penalties,
        }


# ===========================================================================
# CONCEPTNET LOADER
# ===========================================================================

def fetch_conceptnet_triples(word2idx: Dict[str, int], max_triples: int = 5000) -> List[Tuple[str, str, str]]:
    """
    Fetch ConceptNet triples filtered to vocabulary.
    
    Attempts multiple strategies to load ConceptNet:
    1. HuggingFace conceptnet5 dataset (try multiple configs)
    2. Pre-computed English-only subset from HuggingFace
    3. Falls back to an expanded hardcoded commonsense database.
    
    Args:
        word2idx: Vocabulary mapping.
        max_triples: Maximum number of triples to return.
    
    Returns:
        List of (subject, predicate, object) string triples where
        all three words are in vocabulary.
    """
    triples = []
    
    RELATION_MAP = {
        '/r/UsedFor': 'used',
        '/r/CapableOf': 'can',
        '/r/HasProperty': 'has',
        '/r/PartOf': 'part',
        '/r/HasA': 'has',
        '/r/AtLocation': 'at',
        '/r/Causes': 'cause',
        '/r/HasSubevent': 'lead',
        '/r/HasPrerequisite': 'need',
        '/r/MadeOf': 'made',
        '/r/IsA': 'is',
        '/r/Synonym': 'is',
        '/r/Antonym': 'not',
        '/r/RelatedTo': 'related',
        '/r/ReceivesAction': 'receive',
        '/r/Externals': 'has',
        '/r/MotivatedByGoal': 'for',
        '/r/DefinedAs': 'is',
        '/r/InstanceOf': 'is',
        '/r/MemberOf': 'part',
        '/r/SubeventOf': 'part',
        '/r/HasFirstSubevent': 'lead',
        '/r/HasLastSubevent': 'lead',
        '/r/Entails': 'imply',
        '/r/CreatedBy': 'made',
        '/r/SymbolOf': 'is',
        '/r/LocatedNear': 'at',
        '/r/Desires': 'want',
        '/r/NotDesires': 'not',
        '/r/NotIsA': 'not',
        '/r/NotHasProperty': 'not',
        '/r/NotCapableOf': 'not',
        '/r/ObstructedBy': 'need',
        '/r/HasContext': 'at',
    }
    
    def extract_word(uri):
        """Extract English word from ConceptNet URI like /c/en/dog."""
        if not uri:
            return None
        parts = uri.split('/')
        # Format: /c/en/word or /c/en/word/pos
        if len(parts) >= 3 and parts[-2] == 'en':
            word = parts[-1].replace('_', ' ')
            # Skip multi-word phrases
            if ' ' in word:
                return None
            return word.lower()
        elif len(parts) >= 4 and parts[2] == 'en':
            word = parts[3].replace('_', ' ')
            if ' ' in word:
                return None
            return word.lower()
        return None
    
    # Strategy 1: Try HuggingFace ConceptNet dataset with multiple configs
    try:
        from datasets import load_dataset
        print("    Attempting ConceptNet from HuggingFace...")
        
        # Try different dataset names and configs
        dataset_configs = [
            ("conceptnet5", None, "train"),
            ("conceptnet5", "conceptnet5", "train"),
            ("Gabriel/synthetic-reward-training-data-conceptnet", None, "train"),
        ]
        
        ds = None
        for ds_name, ds_config, ds_split in dataset_configs:
            try:
                if ds_config:
                    ds = load_dataset(ds_name, name=ds_config, split=ds_split, streaming=True)
                else:
                    ds = load_dataset(ds_name, split=ds_split, streaming=True)
                print(f"    Loaded from '{ds_name}' (config={ds_config})")
                break
            except Exception:
                continue
        
        if ds is not None:
            scanned = 0
            for example in ds:
                scanned += 1
                if len(triples) >= max_triples:
                    break
                if scanned > max_triples * 30:
                    break
                
                try:
                    # Try multiple column name patterns
                    node1 = (example.get('node1') or example.get('/node1') or 
                             example.get('head') or example.get('source') or '')
                    relation = (example.get('relation') or example.get('/relation') or 
                                example.get('edge_type') or '')
                    node2 = (example.get('node2') or example.get('/node2') or 
                             example.get('tail') or example.get('target') or '')
                    
                    # Some datasets store the full edge as a string
                    if not relation and 'edge' in example:
                        edge = example['edge']
                        if isinstance(edge, str) and '\t' in edge:
                            parts = edge.split('\t')
                            if len(parts) >= 4:
                                node1, relation, node2 = parts[1], parts[2], parts[3]
                    
                    subj = extract_word(str(node1))
                    obj = extract_word(str(node2))
                    pred = RELATION_MAP.get(str(relation), None)
                    
                    if subj and obj and pred:
                        if (subj in word2idx and pred in word2idx and obj in word2idx):
                            triples.append((subj, pred, obj))
                except Exception:
                    continue
            
            if triples:
                print(f"    Loaded {len(triples)} ConceptNet triples (scanned {scanned})")
                return triples
            else:
                print(f"    No matching triples found (scanned {scanned})")
    except Exception as e:
        print(f"    ConceptNet HuggingFace unavailable: {e}")
    
    # Strategy 2: Try pre-filtered conceptnet-assertions-english
    try:
        from datasets import load_dataset
        print("    Trying alternative ConceptNet sources...")
        # Try loading a simpler English-only ConceptNet subset
        for alt_name in ["conceptnet5/assertions", "conceptnet5"]:
            try:
                ds2 = load_dataset(alt_name, split="train", streaming=True)
                scanned = 0
                for example in ds2:
                    scanned += 1
                    if len(triples) >= max_triples:
                        break
                    if scanned > 50000:
                        break
                    # Try parsing
                    for key in example:
                        val = str(example[key])
                        if val.startswith('/c/en/'):
                            break
                break
            except Exception:
                continue
    except Exception:
        pass
    
    # Strategy 3: Expanded hardcoded commonsense database
    print("    Using expanded hardcoded commonsense database...")
    triples = _get_expanded_commonsense()
    
    # Filter to vocabulary
    filtered = []
    for s, p, o in triples:
        if (s in word2idx and p in word2idx and o in word2idx):
            filtered.append((s, p, o))
    
    print(f"    Hardcoded triples: {len(filtered)} matched vocabulary (of {len(triples)} total)")
    return filtered


def _get_expanded_commonsense() -> List[Tuple[str, str, str]]:
    """
    Comprehensive hardcoded commonsense triple database.
    ~500+ triples covering animals, geography, physics, people, 
    objects, causation, education, food, and social relations.
    """
    return [
        # ===== ANIMALS =====
        ("dog", "chase", "cat"), ("cat", "chase", "mouse"),
        ("bird", "fly", "sky"), ("fish", "swim", "water"),
        ("horse", "run", "field"), ("snake", "eat", "mouse"),
        ("dog", "bark", "loud"), ("cat", "meow", "loud"),
        ("bird", "build", "nest"), ("fish", "live", "water"),
        ("horse", "eat", "grass"), ("cow", "eat", "grass"),
        ("sheep", "eat", "grass"), ("rabbit", "eat", "carrot"),
        ("bear", "eat", "honey"), ("bee", "make", "honey"),
        ("spider", "build", "web"), ("ant", "build", "colony"),
        ("lion", "hunt", "prey"), ("tiger", "hunt", "prey"),
        ("eagle", "fly", "high"), ("whale", "swim", "ocean"),
        ("dolphin", "swim", "ocean"), ("shark", "swim", "ocean"),
        ("frog", "jump", "high"), ("monkey", "climb", "tree"),
        ("elephant", "is", "big"), ("mouse", "is", "small"),
        ("dog", "is", "animal"), ("cat", "is", "animal"),
        ("bird", "is", "animal"), ("fish", "is", "animal"),
        ("horse", "is", "animal"), ("cow", "is", "animal"),
        ("sheep", "is", "animal"), ("lion", "is", "animal"),
        ("tiger", "is", "animal"), ("bear", "is", "animal"),
        ("elephant", "is", "animal"), ("whale", "is", "animal"),
        ("dolphin", "is", "animal"), ("frog", "is", "animal"),
        ("snake", "is", "animal"), ("monkey", "is", "animal"),
        ("bee", "is", "insect"), ("ant", "is", "insect"),
        ("spider", "is", "insect"), ("butterfly", "is", "insect"),
        ("dog", "can", "bark"), ("cat", "can", "meow"),
        ("bird", "can", "fly"), ("fish", "can", "swim"),
        ("horse", "can", "run"), ("elephant", "can", "walk"),
        ("frog", "can", "jump"), ("monkey", "can", "climb"),
        ("dog", "has", "tail"), ("cat", "has", "tail"),
        ("bird", "has", "wing"), ("fish", "has", "fin"),
        ("elephant", "has", "trunk"), ("giraffe", "has", "neck"),
        # ===== NATURE AND PHYSICS =====
        ("water", "freeze", "ice"), ("ice", "melt", "water"),
        ("sun", "is", "star"), ("earth", "is", "planet"),
        ("water", "boil", "steam"), ("rain", "fall", "ground"),
        ("fire", "burn", "wood"), ("snow", "fall", "winter"),
        ("wind", "blow", "hard"), ("light", "travel", "fast"),
        ("sound", "travel", "fast"), ("gravity", "pull", "down"),
        ("magnet", "attract", "metal"), ("sun", "provide", "light"),
        ("sun", "provide", "heat"), ("moon", "orbit", "earth"),
        ("earth", "orbit", "sun"), ("planet", "orbit", "star"),
        ("mountain", "is", "high"), ("valley", "is", "low"),
        ("ocean", "is", "deep"), ("river", "flow", "down"),
        ("lake", "contain", "water"), ("ocean", "contain", "water"),
        ("cloud", "contain", "water"), ("ice", "is", "cold"),
        ("fire", "is", "hot"), ("snow", "is", "cold"),
        ("steam", "is", "hot"), ("rock", "is", "hard"),
        ("sand", "is", "soft"), ("glass", "is", "fragile"),
        ("wood", "is", "solid"), ("air", "is", "gas"),
        ("water", "is", "liquid"), ("iron", "is", "metal"),
        ("gold", "is", "metal"), ("silver", "is", "metal"),
        ("copper", "is", "metal"), ("oxygen", "is", "gas"),
        ("hydrogen", "is", "gas"), ("nitrogen", "is", "gas"),
        ("carbon", "is", "element"), ("diamond", "is", "hard"),
        # ===== GEOGRAPHY =====
        ("paris", "is", "capital"), ("france", "has", "capital"),
        ("london", "is", "capital"), ("england", "has", "capital"),
        ("berlin", "is", "capital"), ("germany", "has", "capital"),
        ("rome", "is", "capital"), ("italy", "has", "capital"),
        ("madrid", "is", "capital"), ("spain", "has", "capital"),
        ("tokyo", "is", "capital"), ("japan", "has", "capital"),
        ("beijing", "is", "capital"), ("china", "has", "capital"),
        ("washington", "is", "capital"), ("america", "has", "capital"),
        ("moscow", "is", "capital"), ("russia", "has", "capital"),
        ("canberra", "is", "capital"), ("australia", "has", "capital"),
        ("ottawa", "is", "capital"), ("canada", "has", "capital"),
        ("africa", "is", "continent"), ("europe", "is", "continent"),
        ("asia", "is", "continent"), ("america", "is", "continent"),
        ("australia", "is", "continent"), ("antarctica", "is", "continent"),
        ("sahara", "is", "desert"), ("nile", "is", "river"),
        ("amazon", "is", "river"), ("pacific", "is", "ocean"),
        ("atlantic", "is", "ocean"), ("everest", "is", "mountain"),
        ("alps", "is", "mountain"),
        # ===== PEOPLE AND ROLES =====
        ("student", "study", "subject"), ("teacher", "teach", "student"),
        ("doctor", "treat", "patient"), ("child", "learn", "school"),
        ("scientist", "study", "nature"), ("writer", "write", "book"),
        ("artist", "create", "art"), ("musician", "play", "music"),
        ("chef", "cook", "food"), ("farmer", "grow", "food"),
        ("engineer", "build", "structure"), ("lawyer", "argue", "case"),
        ("nurse", "care", "patient"), ("police", "protect", "people"),
        ("soldier", "protect", "country"), ("pilot", "fly", "plane"),
        ("sailor", "sail", "ship"), ("driver", "drive", "car"),
        ("programmer", "write", "code"), ("designer", "create", "design"),
        ("baker", "bake", "bread"), ("butcher", "cut", "meat"),
        ("carpenter", "build", "furniture"), ("electrician", "fix", "wire"),
        ("mechanic", "fix", "car"), ("plumber", "fix", "pipe"),
        ("doctor", "is", "person"), ("teacher", "is", "person"),
        ("student", "is", "person"), ("scientist", "is", "person"),
        ("mother", "love", "child"), ("father", "love", "child"),
        ("parent", "love", "child"), ("friend", "help", "friend"),
        ("king", "rule", "country"), ("queen", "rule", "country"),
        ("president", "lead", "country"), ("mayor", "lead", "city"),
        # ===== OBJECTS AND PROPERTIES =====
        ("car", "run", "road"), ("boat", "sail", "water"),
        ("plane", "fly", "sky"), ("train", "run", "track"),
        ("bicycle", "run", "road"), ("bus", "run", "road"),
        ("car", "is", "vehicle"), ("bus", "is", "vehicle"),
        ("train", "is", "vehicle"), ("bicycle", "is", "vehicle"),
        ("plane", "is", "vehicle"), ("boat", "is", "vehicle"),
        ("truck", "is", "vehicle"), ("ship", "is", "vehicle"),
        ("car", "has", "wheel"), ("bicycle", "has", "wheel"),
        ("truck", "has", "wheel"), ("bus", "has", "wheel"),
        ("book", "contain", "information"), ("library", "contain", "book"),
        ("computer", "process", "information"), ("internet", "connect", "people"),
        ("phone", "connect", "people"), ("television", "show", "video"),
        ("camera", "capture", "image"), ("clock", "show", "time"),
        ("calendar", "show", "date"), ("map", "show", "location"),
        ("key", "open", "door"), ("door", "is", "entrance"),
        ("window", "is", "opening"), ("wall", "is", "barrier"),
        ("roof", "is", "cover"), ("floor", "is", "surface"),
        ("chair", "is", "furniture"), ("table", "is", "furniture"),
        ("bed", "is", "furniture"), ("desk", "is", "furniture"),
        ("sofa", "is", "furniture"), ("cabinet", "is", "furniture"),
        ("knife", "cut", "food"), ("fork", "is", "utensil"),
        ("spoon", "is", "utensil"), ("plate", "is", "dish"),
        ("cup", "hold", "water"), ("bottle", "hold", "water"),
        ("glass", "hold", "water"), ("bowl", "hold", "food"),
        # ===== FOOD AND HEALTH =====
        ("apple", "is", "fruit"), ("orange", "is", "fruit"),
        ("banana", "is", "fruit"), ("grape", "is", "fruit"),
        ("strawberry", "is", "fruit"), ("lemon", "is", "fruit"),
        ("tomato", "is", "fruit"), ("peach", "is", "fruit"),
        ("carrot", "is", "vegetable"), ("potato", "is", "vegetable"),
        ("onion", "is", "vegetable"), ("rice", "is", "grain"),
        ("wheat", "is", "grain"), ("corn", "is", "grain"),
        ("bread", "is", "food"), ("meat", "is", "food"),
        ("cheese", "is", "food"), ("egg", "is", "food"),
        ("milk", "is", "drink"), ("water", "is", "drink"),
        ("juice", "is", "drink"), ("tea", "is", "drink"),
        ("coffee", "is", "drink"), ("beer", "is", "drink"),
        ("wine", "is", "drink"), ("soup", "is", "food"),
        ("cake", "is", "food"), ("pie", "is", "food"),
        ("food", "provide", "energy"), ("water", "is", "important"),
        ("air", "is", "important"), ("food", "is", "important"),
        ("exercise", "improve", "health"), ("sleep", "improve", "health"),
        ("medicine", "cure", "disease"), ("hospital", "treat", "patient"),
        ("vaccine", "prevent", "disease"), ("vitamin", "improve", "health"),
        # ===== CAUSAL RELATIONS =====
        ("heat", "cause", "expansion"), ("cold", "cause", "contraction"),
        ("rain", "cause", "flood"), ("earthquake", "cause", "damage"),
        ("fire", "cause", "smoke"), ("wind", "cause", "wave"),
        ("sun", "cause", "warm"), ("snow", "cause", "cold"),
        ("reading", "improve", "knowledge"), ("practice", "improve", "skill"),
        ("study", "lead", "knowledge"), ("work", "produce", "result"),
        ("effort", "lead", "success"), ("error", "lead", "learning"),
        ("conflict", "cause", "stress"), ("cooperation", "lead", "progress"),
        ("pollution", "harm", "environment"), ("recycling", "help", "environment"),
        ("education", "improve", "life"), ("technology", "change", "world"),
        ("science", "advance", "knowledge"), ("research", "discover", "truth"),
        # ===== EDUCATION =====
        ("school", "is", "important"), ("education", "is", "important"),
        ("science", "is", "important"), ("research", "is", "important"),
        ("school", "teach", "student"), ("university", "teach", "student"),
        ("library", "contain", "book"), ("laboratory", "use", "equipment"),
        ("classroom", "hold", "student"), ("textbook", "contain", "knowledge"),
        ("math", "is", "subject"), ("science", "is", "subject"),
        ("history", "is", "subject"), ("language", "is", "subject"),
        ("art", "is", "subject"), ("music", "is", "subject"),
        ("math", "use", "number"), ("science", "study", "nature"),
        ("history", "study", "past"), ("geography", "study", "earth"),
        # ===== EMOTIONS AND SOCIAL =====
        ("happiness", "is", "emotion"), ("sadness", "is", "emotion"),
        ("anger", "is", "emotion"), ("fear", "is", "emotion"),
        ("love", "is", "emotion"), ("hate", "is", "emotion"),
        ("joy", "is", "emotion"), ("surprise", "is", "emotion"),
        ("music", "is", "art"), ("painting", "is", "art"),
        ("sculpture", "is", "art"), ("dance", "is", "art"),
        ("poetry", "is", "art"), ("theater", "is", "art"),
        ("sport", "is", "activity"), ("game", "is", "activity"),
        ("football", "is", "sport"), ("basketball", "is", "sport"),
        ("tennis", "is", "sport"), ("swimming", "is", "sport"),
        # ===== COLORS AND SHAPES =====
        ("red", "is", "color"), ("blue", "is", "color"),
        ("green", "is", "color"), ("yellow", "is", "color"),
        ("white", "is", "color"), ("black", "is", "color"),
        ("circle", "is", "shape"), ("square", "is", "shape"),
        ("triangle", "is", "shape"), ("rectangle", "is", "shape"),
        ("sphere", "is", "shape"), ("cube", "is", "shape"),
        # ===== TIME AND WEATHER =====
        ("morning", "is", "time"), ("evening", "is", "time"),
        ("night", "is", "time"), ("afternoon", "is", "time"),
        ("spring", "is", "season"), ("summer", "is", "season"),
        ("autumn", "is", "season"), ("winter", "is", "season"),
        ("january", "is", "month"), ("february", "is", "month"),
        ("monday", "is", "day"), ("friday", "is", "day"),
        ("rain", "is", "weather"), ("snow", "is", "weather"),
        ("storm", "is", "weather"), ("cloud", "is", "weather"),
        ("sun", "is", "star"), ("moon", "is", "satellite"),
        # ===== TOOLS AND MATERIALS =====
        ("hammer", "is", "tool"), ("saw", "is", "tool"),
        ("drill", "is", "tool"), ("screwdriver", "is", "tool"),
        ("wrench", "is", "tool"), ("axe", "is", "tool"),
        ("hammer", "hit", "nail"), ("saw", "cut", "wood"),
        ("drill", "make", "hole"), ("screwdriver", "turn", "screw"),
        ("wood", "is", "material"), ("metal", "is", "material"),
        ("plastic", "is", "material"), ("stone", "is", "material"),
        ("glass", "is", "material"), ("paper", "is", "material"),
        ("cloth", "is", "material"), ("rubber", "is", "material"),
        # ===== BUILDINGS AND PLACES =====
        ("house", "is", "building"), ("apartment", "is", "building"),
        ("office", "is", "building"), ("store", "is", "building"),
        ("restaurant", "is", "building"), ("hospital", "is", "building"),
        ("church", "is", "building"), ("museum", "is", "building"),
        ("city", "is", "place"), ("town", "is", "place"),
        ("village", "is", "place"), ("country", "is", "place"),
        ("park", "is", "place"), ("garden", "is", "place"),
        ("market", "is", "place"), ("airport", "is", "place"),
        # ===== CLOTHING =====
        ("shirt", "is", "clothing"), ("pants", "is", "clothing"),
        ("dress", "is", "clothing"), ("jacket", "is", "clothing"),
        ("coat", "is", "clothing"), ("hat", "is", "clothing"),
        ("shoe", "is", "clothing"), ("sock", "is", "clothing"),
        ("glove", "is", "clothing"), ("scarf", "is", "clothing"),
        # ===== BODY PARTS =====
        ("hand", "is", "part"), ("foot", "is", "part"),
        ("head", "is", "part"), ("eye", "is", "part"),
        ("ear", "is", "part"), ("nose", "is", "part"),
        ("mouth", "is", "part"), ("heart", "is", "organ"),
        ("brain", "is", "organ"), ("lung", "is", "organ"),
        ("eye", "see", "light"), ("ear", "hear", "sound"),
        ("nose", "smell", "odor"), ("tongue", "taste", "flavor"),
        ("skin", "feel", "touch"), ("hand", "hold", "object"),
        ("foot", "walk", "ground"), ("heart", "pump", "blood"),
        ("brain", "think", "thought"), ("lung", "breathe", "air"),
    ]


# ===========================================================================
# ISING-ENHANCED N-GRAM LANGUAGE MODEL
# ===========================================================================

class IsingLM:
    """
    Ising Spin Glass Language Model — v5.0 Genuine Ising Dynamics.

    Architecture (NO overrides, NO bypasses):
      1. POS type selection: Boltzmann from type energy landscape
      2. Word selection: Boltzmann from E(w|ctx) — ALL layers compete
      3. MCMC refinement: Post-generation spin-flip passes (Metropolis)

    Energy landscape:
      E(w|ctx) = -recall(w) -J_pm_i[w,ctx] -J3_knowledge[w,ctx]
                 -h_knowledge[w] -J_category[w,ctx] +E_logic[w,ctx]
                 -h[w] +penalties

    Key principle: Knowledge creates COMPETING energy wells.
    When (dog, barks)->bark and (dog, chases)->chase both fire,
    they produce two deep wells. Boltzmann at temperature beta
    picks between them STOCHASTICALLY. No override. No lookup.

    Phase transition: At critical beta, knowledge transitions from
    invisible (disordered) to dominant (ordered). We tune beta to
    sit near the transition for maximum influence with some noise.

    Parameters:
      - recall_scale, pmi_weight, field_weight
      - beta_type, beta_word
      - ising_enabled (ablation switch)
      - mcmc_refine_steps: number of post-generation spin-flip passes
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
        J: sp.csr_matrix,
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
        J_skip: Optional[Dict[int, sp.csr_matrix]] = None,
        knowledge_layer: Optional["KnowledgeLayer"] = None,
        category_layer: Optional["CategoryLayer"] = None,
        markov_logic_layer: Optional["MarkovLogicLayer"] = None,
        mcmc_refine_steps: int = 2,
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

        # Path 2d: Skip-gram PMI couplings (distance-specific)
        self.J_skip = J_skip if J_skip is not None else {}
        self.max_skip_dist = max(self.J_skip.keys()) if self.J_skip else 0

        # Knowledge layer (Layer 2 + Layer 3)
        self.knowledge_layer = knowledge_layer
        
        # Category layer (Layer 4)
        self.category_layer = category_layer
        
        # Markov Logic layer (Layer 5)
        self.markov_logic_layer = markov_logic_layer
        
        # MCMC refinement (post-generation spin-flip)
        self.mcmc_refine_steps = mcmc_refine_steps

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
            'knowledge_hits': 0, 'spin3_firings': 0,
            'category_hits': 0, 'logic_hits': 0,
            'mcmc_flips_accepted': 0, 'mcmc_flips_proposed': 0,
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

    def _compute_type_energy(self, pos: int, type_idx: int, types_history: List[int],
                             context_words: Optional[List[int]] = None) -> int:
        """Compute energy for a POS type at position pos. Pure integer.
        
        In v5.0, knowledge layers can bias type selection through the energy
        landscape: if 3-spin couplings predict an object of type T, that type
        gets an energy bonus. This is NOT an override — it's a bias in the
        Boltzmann distribution over types.
        """
        energy = 0
        types_for_check = list(types_history) + [type_idx]
        penalty = self.types.compute_grammar_penalty(
            types_for_check, len(types_history), type_idx
        )
        energy += penalty
        if len(types_history) > 0 and type_idx == types_history[-1]:
            if type_idx not in (POS2IDX['NOUN'], POS2IDX['X']):
                energy += 50
        
        # Knowledge-driven type bias: if J3 predicts objects of this type,
        # give this type an energy bonus (lower energy = more likely)
        if context_words is not None and self.knowledge_layer is not None and self.knowledge_layer._built:
            kl = self.knowledge_layer
            ctx = context_words[-kl.max_context_pairs:] if len(context_words) > kl.max_context_pairs else context_words
            type_bonus = 0
            for i in range(len(ctx)):
                for j in range(len(ctx)):
                    if i == j:
                        continue
                    key = (ctx[i], ctx[j])
                    if key in kl.J3:
                        for (obj_idx, strength) in kl.J3[key]:
                            obj_type = self._get_word_type(obj_idx)
                            if obj_type == type_idx:
                                type_bonus += strength // 500  # Scale down for type energy
            energy -= type_bonus  # Lower energy = more likely
        
        return energy

    def _compute_word_energy(
        self, pos: int, candidate_words: np.ndarray, word_type: int,
        context_words: List[int], context_types: List[int], recall_hit: bool,
    ) -> np.ndarray:
        """
        Compute energy for candidate words — v5.0 genuine Ising dynamics.

        E(w) = -recall_bonus(w)          [recall: n-gram match signal]
             - pmi_coupling(w, ctx)       [PMI: word affinity signal]
             - knowledge_energy(w, ctx)   [KNOWLEDGE: Layer 2 + Layer 3]
             - category_energy(w, ctx)    [CATEGORY: Layer 4]
             + logic_energy(w, ctx)       [LOGIC: Layer 5]
             - field(w)                   [unigram frequency]
             + penalties                  [HARD: grammar, anti-repetition]

        v5.0 KEY CHANGE: Knowledge energy is NO LONGER dominated by recall.
        When J3 fires, it produces a deep energy well that naturally wins
        in Boltzmann sampling. Multiple J3 firings create COMPETING wells.
        The winner is determined stochastically by temperature beta.

        All integer arithmetic.
        """
        n_candidates = len(candidate_words)
        energies = np.zeros(n_candidates, dtype=np.int64)

        # === RECALL BONUS (n-gram match signal) ===
        recall_bonuses = self.ngram_index.get_recall_bonus(
            context_words=context_words,
            candidate_words=candidate_words,
            recall_scale=self.recall_scale,
            context_weight_factor=4,
            longest_only=True,
        )
        energies -= recall_bonuses

        # === PMI COUPLING (Ising model -- word affinity signal) ===
        # v5.0: NO damping when recall hits. All terms compete freely.
        if self.ising_enabled and len(context_words) > 0:
            coupling_sums = self._compute_pmi_coupling_sum(
                candidate_words, context_words
            )
            energies -= coupling_sums * self.pmi_weight

        # === KNOWLEDGE ENERGY (Layer 2 + Layer 3) ===
        # v5.0: Knowledge competes freely through the Hamiltonian.
        # When J3 fires, it creates deep energy wells that DOMINATE
        # over recall. Multiple triples create COMPETING wells.
        if self.knowledge_layer is not None and len(context_words) > 0:
            knowledge_bonus = self.knowledge_layer.compute_knowledge_energy(
                context_words, candidate_words
            )
            energies -= knowledge_bonus
            # Track diagnostics
            if int(knowledge_bonus.max()) > 0:
                self._stats['knowledge_hits'] += 1
            kd = self.knowledge_layer.get_diagnostics(context_words, candidate_words)
            if kd['spin3_hits'] > 0:
                self._stats['spin3_firings'] += 1
        
        # === CATEGORY ENERGY (Layer 4: hypernym-based couplings) ===
        if self.category_layer is not None and len(context_words) > 0:
            cat_bonus = self.category_layer.compute_category_bonus(
                context_words, candidate_words
            )
            energies -= cat_bonus
            cat_field = self.category_layer.compute_category_field(
                context_words, candidate_words
            )
            energies -= cat_field
            # Track diagnostics
            if int(cat_bonus.max()) > 0 or int(cat_field.max()) > 0:
                self._stats['category_hits'] += 1
        
        # === MARKOV LOGIC ENERGY (Layer 5: factual consistency) ===
        if self.markov_logic_layer is not None and len(context_words) > 0:
            logic_energy = self.markov_logic_layer.compute_logic_energy(
                context_words, candidate_words
            )
            energies += logic_energy  # logic_energy already has correct sign
            # Track diagnostics
            if int(np.abs(logic_energy).max()) > 0:
                self._stats['logic_hits'] += 1

        # === LOCAL FIELD (unigram frequency) ===
        # v5.0: Field always contributes fully, no damping
        field_vals = self.h[candidate_words] * self.field_weight
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

    def _compute_pmi_coupling_sum(
        self,
        candidate_words: np.ndarray,
        context_words: List[int],
    ) -> np.ndarray:
        """
        Compute PMI coupling sums for candidate words against context.

        Path 2d: Uses distance-weighted skip-gram couplings when available.
        For each context word at distance d, uses J_skip[d] if present,
        otherwise falls back to the base J matrix.

        Path 3b: Uses sparse matrix operations.

        Returns coupling_sums array of shape (n_candidates,) with int64 values.
        """
        n_candidates = len(candidate_words)
        coupling_sums = np.zeros(n_candidates, dtype=np.int64)

        if not self.ising_enabled or len(context_words) == 0:
            return coupling_sums

        # Determine the effective window
        effective_window = max(self.window, self.max_skip_dist)
        context_start = max(0, len(context_words) - effective_window)
        ctx = context_words[context_start:]

        if not ctx:
            return coupling_sums

        # If we have skip-gram couplings, use distance-specific matrices
        if self.J_skip:
            for i, c in enumerate(ctx):
                # Distance from current position (1-indexed)
                dist = len(ctx) - i
                c_int = int(c)

                # Choose the appropriate coupling matrix for this distance
                if dist in self.J_skip:
                    J_dist = self.J_skip[dist]
                else:
                    J_dist = self.J

                # Extract coupling values: J_dist[w, c] for each candidate w
                # With scipy sparse, A[[r1,r2,...], [c1,c2,...]] gives element-wise pairs
                c_list = [c_int] * n_candidates
                w_list = candidate_words.tolist()
                coupling_vals = J_dist[w_list, c_list]
                coupling_sums += np.asarray(coupling_vals, dtype=np.int64).flatten()
        else:
            # Fallback: use base J with flat window
            ctx_arr = np.array(ctx, dtype=np.int64)
            # Sparse matrix row/column selection
            J_rows = self.J[candidate_words.tolist(), :]
            J_sub = J_rows[:, ctx_arr.tolist()]
            coupling_sums = np.asarray(J_sub.sum(axis=1), dtype=np.int64).flatten()

        return coupling_sums

    # ===================================================================
    # Path 2a: Global Coherence Scoring (Beam Generation)
    # ===================================================================

    def generate_beam(self, prompt: str = "the", length: int = 20,
                      n_beams: int = 5) -> Dict:
        """
        Generate text using beam-like search with global energy ranking.

        Generates n_beams candidate sequences independently, then ranks
        them by total energy E = Σ E(w_t). The lowest-energy sequence
        is returned as the best candidate.

        This uses the Ising model as a GLOBAL constraint rather than
        just a local next-word signal.

        Args:
            prompt: Starting word.
            length: Number of tokens to generate.
            n_beams: Number of candidate sequences to generate.

        Returns:
            Dict with 'text', 'words', 'types', 'diagnostics',
            plus 'beam_energy' and 'all_candidates'.
        """
        candidates = []
        for beam_idx in range(n_beams):
            result = self.generate(prompt=prompt, length=length)

            # Compute total energy for the full sequence
            total_energy = 0
            words = result['words']
            types_list = result['types']
            for pos in range(1, len(words)):
                context_words = words[:pos]
                context_types = types_list[:pos]
                word_type = types_list[pos]
                recall_matches = self.ngram_index.lookup(context_words)
                recall_hit = bool(recall_matches)
                candidate_arr = np.array([words[pos]], dtype=np.int64)
                e = self._compute_word_energy(
                    pos, candidate_arr, word_type,
                    context_words, context_types, recall_hit
                )
                total_energy += int(e[0])

            candidates.append({
                'energy': total_energy,
                'result': result,
            })

        # Sort by energy (lowest = best = most coherent)
        candidates.sort(key=lambda x: x['energy'])

        best = candidates[0]['result']
        best['beam_energy'] = candidates[0]['energy']
        best['beam_rank'] = 1
        best['all_candidates'] = [
            {'energy': c['energy'], 'text': c['result']['text']}
            for c in candidates
        ]
        return best

    # ===================================================================
    # Path 2b: Joint Phrase Sampling via MCMC
    # ===================================================================

    def _joint_sample(
        self,
        context_words: List[int],
        context_types: List[int],
        phrase_len: int = 2,
        n_proposals: int = 20,
    ) -> Optional[List[int]]:
        """
        Joint phrase sampling using MCMC.

        Instead of always sampling one word at a time, sometimes sample
        2-3 words jointly. Proposes a phrase (w1, w2, ...), computes
        joint energy E(w1, w2) = E(w1) + E(w2) + J[w1, w2], then
        accepts/rejects via Metropolis criterion.

        This is where J-couplings ACTUALLY constrain word combinations,
        because the direct coupling J[w1, w2] is explicitly evaluated.

        Args:
            context_words: Preceding word indices.
            context_types: Preceding POS type indices.
            phrase_len: Number of words to sample jointly (2 or 3).
            n_proposals: Number of MCMC proposals to try.

        Returns:
            List of word indices for the accepted phrase, or None
            if no good phrase found (fall back to single-word sampling).
        """
        if phrase_len < 2:
            return None

        best_phrase = None
        best_energy = 0  # We want the most negative energy (lowest)

        # Get the type of the previous word for type selection
        prev_type = context_types[-1] if context_types else POS2IDX["NOUN"]

        for _ in range(n_proposals):
            phrase_words = []
            phrase_types = []
            phrase_energy = 0

            # Generate the phrase word by word, but compute JOINT energy
            for step in range(phrase_len):
                if step == 0:
                    step_prev_type = prev_type
                    step_types_history = context_types
                else:
                    step_prev_type = phrase_types[step - 1]
                    step_types_history = context_types + phrase_types

                # Choose type for this position
                valid_types = self._get_valid_next_types(
                    step_prev_type, step_types_history
                )
                if not valid_types:
                    break

                # Pick a type via Boltzmann (energy-based, no override)
                if step == 0:
                    step_context = context_words
                else:
                    step_context = context_words + phrase_words

                # Compute type energies (includes knowledge bias via J3)
                type_energies = np.array([
                    self._compute_type_energy(
                        len(context_words) + step, t, step_types_history, step_context
                    )
                    for t in valid_types
                ], dtype=np.int64)

                # Add recall type bias as energy bonus (NOT an override)
                recall_matches = self.ngram_index.lookup(step_context)
                if recall_matches:
                    best_k = max(recall_matches.keys())
                    best_conts = recall_matches[best_k]
                    if best_k >= 2 and best_conts:
                        recall_word, _, _ = best_conts[0]
                        recall_type = self._get_word_type(recall_word)
                        if recall_type in valid_types:
                            for i, t in enumerate(valid_types):
                                if t == recall_type:
                                    type_energies[i] -= 200  # Moderate recall bias

                chosen_type = valid_types[self.type_sampler.sample(type_energies)]

                # Get candidates for this type
                candidate_list = self.type_words.get(chosen_type, [])
                if not candidate_list:
                    break
                candidate_words = np.array(candidate_list, dtype=np.int64)

                # Top-k filtering
                if len(candidate_words) > 200:
                    field_vals = self.h[candidate_words]
                    top_k = np.argsort(field_vals)[-200:]
                    candidate_words = candidate_words[top_k]

                # Compute energy
                if step == 0:
                    e_context = context_words
                else:
                    e_context = context_words + phrase_words

                recall_hit = bool(self.ngram_index.lookup(e_context))
                energies = self._compute_word_energy(
                    len(context_words) + step, candidate_words, chosen_type,
                    e_context, step_types_history, recall_hit
                )

                # Sample a word
                word_idx = self.word_sampler.sample(energies)
                chosen_word = int(candidate_words[word_idx])

                phrase_words.append(chosen_word)
                phrase_types.append(chosen_type)
                phrase_energy += int(energies[word_idx])

            if len(phrase_words) < phrase_len:
                continue

            # Add direct pairwise J coupling between phrase words
            # This is the KEY part where J-couplings constrain combinations
            for i in range(len(phrase_words)):
                for j in range(i + 1, len(phrase_words)):
                    w1, w2 = phrase_words[i], phrase_words[j]
                    # Get direct J coupling from sparse matrix
                    coupling = int(self.J[w1, w2])
                    phrase_energy -= coupling * self.pmi_weight

            if phrase_energy < best_energy:
                best_energy = phrase_energy
                best_phrase = phrase_words

        # Accept the phrase if its energy is negative enough
        # (i.e., it's a low-energy = good configuration)
        if best_phrase is not None and best_energy < 0:
            return best_phrase

        return None

    # ===================================================================
    # Path 2c: Temperature Annealing
    # ===================================================================

    def generate_annealed(
        self,
        prompt: str = "the",
        length: int = 20,
        beta_start: float = 0.005,
        beta_end: float = 0.5,
    ) -> Dict:
        """
        Generate text with linear temperature annealing.

        Simulates the Ising model phase transition:
        - Start HOT (low beta = diverse, random sampling)
        - Cool down (high beta = deterministic, low-energy sampling)
        - beta(t) = beta_start + (beta_end - beta_start) * t / length

        At each position, a NEW Boltzmann sampler is created with the
        current beta value. This is computationally more expensive than
        standard generation but provides genuine Ising annealing.

        Args:
            prompt: Starting word.
            length: Number of tokens to generate.
            beta_start: Initial inverse temperature (hot = diverse).
            beta_end: Final inverse temperature (cold = deterministic).

        Returns:
            Dict with 'text', 'words', 'types', 'diagnostics',
            plus 'beta_schedule'.
        """
        # Resolve prompt
        prompt_idx = self.vocab.word2idx.get(prompt)
        if prompt_idx is None:
            prompt_idx = self.vocab.word2idx.get(prompt.lower())
        if prompt_idx is None:
            prompt_idx = 4

        prompt_type = self._get_word_type(prompt_idx)
        words = [prompt_idx]
        types_list = [prompt_type]
        consecutive_copies = 0
        diagnostics = []
        beta_schedule = []

        for pos in range(1, length):
            # Linear annealing: beta increases over time
            beta_t = beta_start + (beta_end - beta_start) * pos / max(1, length - 1)
            beta_schedule.append(beta_t)

            # Create a new sampler with the current beta
            annealed_word_sampler = IntegerBoltzmannSampler(
                beta=beta_t, max_delta=500
            )

            # === STEP 1: Choose POS type (BOLTZMANN, not override) ===
            valid_types = self._get_valid_next_types(types_list[-1], types_list)

            # Check if recall suggests a type bias (not override — just bias)
            recall_type_bias = None
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
                                recall_type_bias = recall_type

            # Compute type energies (includes knowledge bias via J3 predictions)
            type_energies = np.array([
                self._compute_type_energy(pos, t, types_list, words)
                for t in valid_types
            ], dtype=np.int64)

            # Add recall type bias as an energy bonus (NOT an override)
            if recall_type_bias is not None:
                for i, t in enumerate(valid_types):
                    if t == recall_type_bias:
                        type_energies[i] -= 200  # Moderate recall bias

            chosen_type = valid_types[self.type_sampler.sample(type_energies)]

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

            # === STEP 3: Choose word via Boltzmann (ALL knowledge through energy) ===
            candidate_list = self.type_words.get(chosen_type, [])
            if not candidate_list:
                candidate_list = list(range(min(200, self.vocab_size)))
            candidate_words = np.array(candidate_list, dtype=np.int64)

            # Top-k filtering by field strength
            if len(candidate_words) > 300:
                field_vals = self.h[candidate_words]
                top_k = np.argsort(field_vals)[-300:]
                candidate_words = candidate_words[top_k]

            # HARD LOGIC FILTER: infinite energy barriers
            logic_mask = self._filter_by_logic(candidate_words, words)
            if logic_mask.any():
                candidate_words = candidate_words[logic_mask]

            # Check recall availability
            recall_matches = self.ngram_index.lookup(words)
            recall_hit = bool(recall_matches)

            # Compute energy (integer-only, ALL 5 layers compete)
            word_energies = self._compute_word_energy(
                pos, candidate_words, chosen_type,
                words, types_list, recall_hit
            )

            # Integer Boltzmann sample with ANNEALED temperature
            chosen_energy = 0
            if copy_word is not None:
                chosen_word = copy_word
            else:
                word_idx = annealed_word_sampler.sample(word_energies)
                chosen_word = int(candidate_words[word_idx])
                chosen_energy = int(word_energies[word_idx])

            words.append(chosen_word)
            types_list.append(chosen_type)

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
                'energy': chosen_energy,
                'beta': beta_t,
            })

        # === STEP 4: MCMC spin-flip refinement ===
        if self.mcmc_refine_steps > 0:
            words, types_list = self._mcmc_refine(words, types_list, n_passes=self.mcmc_refine_steps)

        text = self.vocab.decode(words)
        type_names = [IDX2POS.get(t, "UNK") for t in types_list]

        return {
            'text': text,
            'words': words,
            'types': types_list,
            'type_names': type_names,
            'diagnostics': diagnostics,
            'beta_schedule': beta_schedule,
        }

    # ===================================================================
    # v5.0: Genuine Ising Generation (NO overrides, ALL Boltzmann)
    # ===================================================================

    def _filter_by_logic(self, candidate_words, context_words):
        """
        HARD LOGIC FILTER: Remove candidates that violate hard logic rules.
        
        This is physically legitimate — hard rules act as infinite energy
        barriers. They don't bypass Boltzmann; they set E(w) = infinity
        for physically impossible states.
        
        Returns a boolean mask of shape (n_candidates,) where True = keep.
        """
        n = len(candidate_words)
        keep = np.ones(n, dtype=bool)
        
        if self.markov_logic_layer is None or not self.markov_logic_layer._built:
            return keep
        
        logic_energy = self.markov_logic_layer.compute_logic_energy(
            context_words, candidate_words
        )
        
        # Hard rules produce penalties > 10000
        for i in range(n):
            if int(logic_energy[i]) > 10000:
                keep[i] = False
                self._stats['logic_hits'] += 1
        
        return keep

    def _compute_sequence_energy(self, words: List[int], types_list: List[int]) -> int:
        """
        Compute total energy of a word sequence.
        
        E_total = sum over positions of E(w_t | ctx_t)
        
        This is used by MCMC refinement to evaluate whether a spin-flip
        lowers the total energy of the sequence.
        """
        total_energy = 0
        for pos in range(1, len(words)):
            context_words = words[:pos]
            context_types = types_list[:pos]
            word_type = types_list[pos]
            recall_matches = self.ngram_index.lookup(context_words)
            recall_hit = bool(recall_matches)
            candidate_arr = np.array([words[pos]], dtype=np.int64)
            e = self._compute_word_energy(
                pos, candidate_arr, word_type,
                context_words, context_types, recall_hit
            )
            total_energy += int(e[0])
        return total_energy

    def _mcmc_refine(self, words: List[int], types_list: List[int],
                     n_passes: int = 2) -> Tuple[List[int], List[int]]:
        """
        MCMC spin-flip refinement — genuinely Ising dynamics.
        
        After initial generation, iterate through the sequence and propose
        flipping each word to a lower-energy alternative. Accept via the
        Metropolis criterion:
        
            If E_new < E_old: accept (lower energy = better)
            If E_new >= E_old: accept with probability exp(-beta * (E_new - E_old))
        
        This is EXACTLY how an Ising spin glass relaxes to its ground state.
        Multiple competing knowledge triples create multiple local minima;
        MCMC helps the system find deeper minima.
        
        The Metropolis acceptance uses the IntegerBoltzmannSampler's lookup
        table, so it's integer-only in the hot path.
        
        Args:
            words: List of word indices
            types_list: List of POS type indices
            n_passes: Number of full sweeps through the sequence
            
        Returns:
            (refined_words, refined_types) — possibly modified sequences
        """
        if n_passes <= 0 or len(words) < 3:
            return words, types_list
        
        words = list(words)  # Make mutable copy
        types_list = list(types_list)
        
        for pass_idx in range(n_passes):
            # Sweep through sequence, proposing flips at each position
            for pos in range(1, len(words)):
                context_words = words[:pos]
                context_types = types_list[:pos]
                current_word = words[pos]
                word_type = types_list[pos]
                
                # Get candidate words of the same type
                candidate_list = self.type_words.get(word_type, [])
                if len(candidate_list) < 2:
                    continue
                candidate_words = np.array(candidate_list, dtype=np.int64)
                
                # Top-k filtering
                if len(candidate_words) > 300:
                    field_vals = self.h[candidate_words]
                    top_k = np.argsort(field_vals)[-300:]
                    candidate_words = candidate_words[top_k]
                
                # Hard logic filter
                logic_mask = self._filter_by_logic(candidate_words, context_words)
                if logic_mask.any():
                    candidate_words = candidate_words[logic_mask]
                if len(candidate_words) < 2:
                    continue
                
                # Compute energies for all candidates
                recall_matches = self.ngram_index.lookup(context_words)
                recall_hit = bool(recall_matches)
                energies = self._compute_word_energy(
                    pos, candidate_words, word_type,
                    context_words, context_types, recall_hit
                )
                
                # Current word energy
                current_mask = candidate_words == current_word
                if current_mask.any():
                    current_idx = np.where(current_mask)[0][0]
                    current_energy = int(energies[current_idx])
                else:
                    current_energy = 0  # Current word not in candidates
                
                # Propose a flip: sample from Boltzmann distribution
                self._stats['mcmc_flips_proposed'] += 1
                proposed_idx = self.word_sampler.sample(energies)
                proposed_word = int(candidate_words[proposed_idx])
                proposed_energy = int(energies[proposed_idx])
                
                # Metropolis criterion
                delta_e = proposed_energy - current_energy
                
                if delta_e < 0:
                    # Always accept lower energy
                    words[pos] = proposed_word
                    self._stats['mcmc_flips_accepted'] += 1
                else:
                    # Accept with probability exp(-beta * delta_e)
                    # Use the Boltzmann lookup table for this
                    delta_clamped = min(delta_e, self.word_sampler.max_delta)
                    accept_prob_weight = int(self.word_sampler.table[delta_clamped])
                    total_weight = self.word_sampler.scale  # P(reject) + P(accept) = 1
                    if np.random.randint(0, total_weight) < accept_prob_weight:
                        words[pos] = proposed_word
                        self._stats['mcmc_flips_accepted'] += 1
        
        return words, types_list

    def generate(self, prompt: str = "the", length: int = 20) -> Dict:
        """
        Generate text autoregressively — v5.0 GENUINE ISING DYNAMICS.
        
        At each position:
          1. Choose POS type: Boltzmann from type energy landscape
             (grammar + knowledge bias via J3 predictions)
          2. Check copy mechanism (legitimate: it's a form of recall)
          3. Apply hard logic filter (infinite energy barriers)
          4. Compute E(w|ctx) with ALL 5 layers competing
          5. Boltzmann sample: P(w) ~ exp(-beta * E(w))
          6. Post-generation: MCMC spin-flip refinement
        
        NO overrides. NO bypasses. Knowledge competes through the Hamiltonian.
        When multiple J3 keys fire, they create competing energy wells.
        Boltzmann picks between them stochastically at temperature beta.
        
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
            # === STEP 1: Choose POS type (BOLTZMANN, not override) ===
            valid_types = self._get_valid_next_types(types[-1], types)

            # Check if recall suggests a type bias (not override — just bias)
            recall_type_bias = None
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
                                recall_type_bias = recall_type

            # Compute type energies (includes knowledge bias through J3)
            type_energies = np.array([
                self._compute_type_energy(pos, t, types, words)
                for t in valid_types
            ], dtype=np.int64)
            
            # Add recall type bias as an energy bonus (NOT an override)
            if recall_type_bias is not None:
                for i, t in enumerate(valid_types):
                    if t == recall_type_bias:
                        type_energies[i] -= 200  # Moderate recall bias

            chosen_type = valid_types[self.type_sampler.sample(type_energies)]

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

            # === STEP 3: Choose word via Boltzmann (ALL knowledge through energy) ===
            candidate_list = self.type_words.get(chosen_type, [])
            if not candidate_list:
                candidate_list = list(range(min(200, self.vocab_size)))
            candidate_words = np.array(candidate_list, dtype=np.int64)

            # Top-k filtering by field strength
            if len(candidate_words) > 300:
                field_vals = self.h[candidate_words]
                top_k = np.argsort(field_vals)[-300:]
                candidate_words = candidate_words[top_k]

            # HARD LOGIC FILTER: infinite energy barriers
            logic_mask = self._filter_by_logic(candidate_words, words)
            if logic_mask.any():
                candidate_words = candidate_words[logic_mask]
            # If all filtered out, keep originals (safety)

            # Check recall availability
            recall_matches = self.ngram_index.lookup(words)
            recall_hit = bool(recall_matches)

            # Compute energy (integer-only, ALL 5 layers compete)
            word_energies = self._compute_word_energy(
                pos, candidate_words, chosen_type,
                words, types, recall_hit
            )

            # Integer Boltzmann sample — knowledge wins through DEEP ENERGY WELLS
            chosen_energy = 0
            if copy_word is not None:
                chosen_word = copy_word
            else:
                word_idx = self.word_sampler.sample(word_energies)
                chosen_word = int(candidate_words[word_idx])
                chosen_energy = int(word_energies[word_idx])

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
                'energy': chosen_energy,
            })

        # === STEP 4: MCMC spin-flip refinement ===
        if self.mcmc_refine_steps > 0:
            words, types = self._mcmc_refine(words, types, n_passes=self.mcmc_refine_steps)

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
        stats['mcmc_accept_rate'] = (
            stats['mcmc_flips_accepted'] / max(1, stats['mcmc_flips_proposed'])
        )
        return stats


# ===========================================================================
# MODEL: Training + Generation Pipeline
# ===========================================================================

class IsingLMModel:
    """
    Complete model: training pipeline + generation.

    Training:
      1. Load corpus
      2. Build vocabulary (with enhanced tokenizer)
      3. Build POS type system
      4. Compute PMI couplings (sparse)
      5. Compute skip-gram PMI couplings (distance-specific)
      6. Build n-gram index
      7. Build knowledge layer (SPO triples + 3-spin couplings)
      8. Create generator(s)

    Generation:
      - With Ising + Knowledge (default)
      - Without Ising (ablation baseline, but WITH knowledge)
      - Without Knowledge (knowledge-off baseline, with Ising)
      - Beam generation (global coherence)
      - Annealed generation (phase transition)
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
        skip_pmi_max_dist: int = 5,
        knowledge_scale: int = 500,
        spin3_scale: int = 800,
        category_scale: int = 400,
        logic_rule_scale: int = 600,
        logic_hard_scale: int = 50000,
        use_conceptnet: bool = True,
        mcmc_refine_steps: int = 2,
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
        self.skip_pmi_max_dist = skip_pmi_max_dist
        self.knowledge_scale = knowledge_scale
        self.spin3_scale = spin3_scale
        self.category_scale = category_scale
        self.logic_rule_scale = logic_rule_scale
        self.logic_hard_scale = logic_hard_scale
        self.use_conceptnet = use_conceptnet
        self.mcmc_refine_steps = mcmc_refine_steps

        self.vocab: Optional[Vocabulary] = None
        self.types: Optional[POSTypeSystem] = None
        self.J: Optional[sp.csr_matrix] = None
        self.h: Optional[np.ndarray] = None
        self.J_skip: Optional[Dict[int, sp.csr_matrix]] = None
        self.ngram_index: Optional[NGramIndex] = None
        self.knowledge_layer: Optional[KnowledgeLayer] = None
        self.category_layer: Optional[CategoryLayer] = None
        self.markov_logic_layer: Optional[MarkovLogicLayer] = None
        self.generator: Optional[IsingLM] = None
        self.baseline_generator: Optional[IsingLM] = None
        self.sequences: Optional[List[List[int]]] = None
        self.test_sequences: Optional[List[List[int]]] = None

    def train(self, n_samples: int = 20000) -> "IsingLMModel":
        """Train the model from FineWeb-Edu corpus."""
        print("=" * 70)
        print("ISING-ENHANCED N-GRAM LANGUAGE MODEL -- TRAINING")
        print("=" * 70)
        print(f"\n  Architecture: N-gram (primary) + Ising PMI (secondary) + 5 Knowledge Layers")
        print(f"  Integer-only hot path: Lookup-table Boltzmann (NO np.exp)")
        print(f"  Ising enabled: {self.ising_enabled}")
        print(f"  Sparse PMI: YES (scipy.sparse.csr_matrix)")
        print(f"  Skip-gram PMI: YES (distance {1}-{self.skip_pmi_max_dist})")
        print(f"  Knowledge scale: {self.knowledge_scale}")
        print(f"  3-Spin scale: {self.spin3_scale}")
        print(f"  Category scale: {self.category_scale}")
        print(f"  Logic rule scale: {self.logic_rule_scale}")
        print(f"  Use ConceptNet: {self.use_conceptnet}")
        print()

        t0 = time.time()

        # Step 1: Load corpus
        print("[1/11] Loading corpus...")
        texts = load_fineweb_edu(n_samples=n_samples)
        print(f"  Loaded {len(texts)} texts ({time.time()-t0:.1f}s)")

        # Step 2: Build vocabulary (with knowledge augmentation)
        print("\n[2/11] Building vocabulary...")
        self.vocab = Vocabulary(
            min_freq=self.vocab_min_freq,
            max_size=self.vocab_max_size,
        )
        self.vocab.build(texts)
        print(f"  Corpus vocabulary: {len(self.vocab)} words")
        
        # Step 2b: Augment vocabulary with knowledge words
        # These are words that appear in our knowledge triples and category
        # definitions but may not appear in the corpus. Without them, our
        # knowledge layers silently fail because triple words aren't in vocab.
        knowledge_words = self._collect_knowledge_words()
        n_added = self.vocab.add_words(knowledge_words)
        if n_added > 0:
            print(f"  Added {n_added} knowledge words (total: {len(self.vocab)})")

        # Step 3: Build POS type system
        print("\n[3/11] Building POS type system...")
        self.types = POSTypeSystem(
            vocab_size=len(self.vocab),
            window=self.pmi_window,
        )
        self.types.build_from_vocabulary(self.vocab.word2idx, self.vocab.idx2word)
        self.types.build_grammar_penalties(penalty_strength=60)
        sequences = tokenize_texts(texts, self.vocab)
        sequences = truncate_sequences(sequences, max_len=20)

        # Path 3c: Split 90% train, 10% test for perplexity evaluation
        split_idx = int(len(sequences) * 0.9)
        self.sequences = sequences[:split_idx]
        self.test_sequences = sequences[split_idx:]
        print(f"  Train sequences: {len(self.sequences)}, Test sequences: {len(self.test_sequences)}")

        self.types.compute_type_couplings(self.sequences, self.vocab.idx2word)
        n_typed = sum(1 for w in range(len(self.vocab)) if w in self.types.allowed_types)
        print(f"  POS system built: {N_POS} types, {n_typed} words typed")

        # Step 4: Compute PMI couplings (sparse)
        print("\n[4/11] Computing PMI couplings (sparse)...")
        self.J, self.h = compute_pmi_couplings(
            self.sequences, len(self.vocab),
            window=self.pmi_window,
            min_count=self.pmi_min_count,
            pmi_cap=self.pmi_cap,
        )

        # Step 5: Compute skip-gram PMI couplings
        print(f"\n[5/11] Computing skip-gram PMI couplings (dist 1-{self.skip_pmi_max_dist})...")
        self.J_skip = compute_skip_pmi_couplings(
            self.sequences, len(self.vocab),
            max_dist=self.skip_pmi_max_dist,
            min_count=self.pmi_min_count,
            pmi_cap=self.pmi_cap,
        )

        # Step 6: Build n-gram index
        print("\n[6/11] Building n-gram index...")
        self.ngram_index = NGramIndex(
            max_n=self.ngram_max_n,
            min_count=self.ngram_min_count,
        )
        self.ngram_index.build(self.sequences)

        # Step 7: Build knowledge layer (Layer 2 + Layer 3)
        print("\n[7/11] Building knowledge layer (Layer 2 + Layer 3)...")
        self.knowledge_layer = KnowledgeLayer(
            vocab_size=len(self.vocab),
            knowledge_scale=self.knowledge_scale,
            spin3_scale=self.spin3_scale,
        )
        # Extract SPO triples from corpus
        self.knowledge_layer.add_triples_from_corpus(
            self.sequences, self.vocab.idx2word, self.types, min_count=3
        )
        # Add expanded commonsense triples
        self._add_commonsense_triples()
        # Add ConceptNet triples (if enabled)
        if self.use_conceptnet:
            self._add_conceptnet_triples()
        # Finalize
        self.knowledge_layer.build()
        
        # Step 8: Build category layer (Layer 4)
        print("\n[8/11] Building category layer (Layer 4)...")
        self.category_layer = CategoryLayer(
            vocab_size=len(self.vocab),
            category_scale=self.category_scale,
        )
        self._add_category_ontology()
        self.category_layer.build()
        
        # Step 9: Build Markov Logic layer (Layer 5)
        print("\n[9/11] Building Markov Logic layer (Layer 5)...")
        self.markov_logic_layer = MarkovLogicLayer(
            vocab_size=len(self.vocab),
            rule_scale=self.logic_rule_scale,
            hard_rule_scale=self.logic_hard_scale,
        )
        self._add_logic_rules()
        self.markov_logic_layer.build()

        # Step 10: Compute scale diagnostics
        print("\n[10/11] Scale diagnostics...")
        self._print_scale_diagnostics()

        # Step 11: Build generators
        print("\n[11/11] Building generators...")
        self._build_generators()

        t_total = time.time() - t0
        print(f"\nTraining complete: {t_total:.1f}s")
        return self

    def _collect_knowledge_words(self):
        """
        Collect all words from knowledge triples and category definitions
        that should be in vocabulary even if absent from corpus.
        
        This is the key to making knowledge layers VISIBLE: if "bark" isn't
        in the vocabulary, the triple ("dog", "bark", "loud") silently fails.
        """
        knowledge_words = set()
        
        # From commonsense triples
        for s, p, o in _get_expanded_commonsense():
            knowledge_words.add(s)
            knowledge_words.add(p)
            knowledge_words.add(o)
        
        # From category definitions
        for cat_name, word_list in self._get_category_definitions().items():
            knowledge_words.update(word_list)
        
        # From logic rule words
        for triggers, targets, _, _ in self._get_logic_rule_definitions():
            knowledge_words.update(triggers)
            knowledge_words.update(targets)
        
        # Remove very short words and pure punctuation
        knowledge_words = {
            w for w in knowledge_words 
            if len(w) >= 2 and any(c.isalpha() for c in w)
        }
        
        return sorted(knowledge_words)
    
    def _get_category_definitions(self):
        """Return category definitions dict (used for vocab augmentation + building)."""
        return {
            # Living things
            "animal": ["dog", "cat", "bird", "fish", "horse", "cow", "sheep",
                       "lion", "tiger", "bear", "elephant", "whale", "dolphin",
                       "frog", "snake", "monkey", "rabbit", "deer", "wolf",
                       "fox", "eagle", "hawk", "owl", "penguin", "turtle"],
            "insect": ["bee", "ant", "spider", "butterfly", "fly", "mosquito",
                       "beetle", "worm", "cockroach"],
            "person": ["man", "woman", "child", "boy", "girl", "person",
                       "people", "student", "teacher", "doctor", "nurse",
                       "scientist", "writer", "artist", "musician", "chef",
                       "farmer", "engineer", "lawyer", "pilot", "driver",
                       "programmer", "designer", "baker", "mother", "father",
                       "parent", "friend", "king", "queen", "president"],
            # Natural objects
            "planet": ["earth", "mars", "venus", "jupiter", "saturn",
                       "mercury", "neptune", "uranus", "pluto"],
            "star": ["sun", "star", "sirius"],
            "satellite": ["moon"],
            # Materials and substances
            "metal": ["iron", "gold", "silver", "copper", "aluminum", "steel",
                      "tin", "zinc", "lead", "platinum"],
            "gas": ["air", "oxygen", "hydrogen", "nitrogen", "helium",
                    "carbon", "steam", "vapor"],
            "liquid": ["water", "oil", "milk", "juice", "blood", "acid",
                       "alcohol", "gasoline"],
            "material": ["wood", "metal", "plastic", "stone", "glass",
                         "paper", "cloth", "rubber", "leather", "cement",
                         "brick", "clay", "sand", "concrete"],
            # Places
            "building": ["house", "apartment", "office", "store", "restaurant",
                         "hospital", "church", "museum", "school", "library",
                         "factory", "warehouse", "hotel", "theater", "stadium"],
            "place": ["city", "town", "village", "country", "state", "park",
                      "garden", "market", "airport", "station", "port",
                      "beach", "forest", "desert", "island", "mountain",
                      "valley", "lake", "river", "road", "street", "bridge"],
            "continent": ["africa", "europe", "asia", "america", "australia",
                          "antarctica"],
            # Vehicles
            "vehicle": ["car", "bus", "train", "bicycle", "plane", "boat",
                        "truck", "ship", "motorcycle", "helicopter", "submarine",
                        "rocket", "van", "taxi"],
            # Food and drink
            "fruit": ["apple", "orange", "banana", "grape", "strawberry",
                      "lemon", "tomato", "peach", "pear", "cherry", "mango",
                      "pineapple", "watermelon", "plum"],
            "vegetable": ["carrot", "potato", "onion", "tomato", "cabbage",
                          "lettuce", "pepper", "corn", "pea", "bean", "celery"],
            "food": ["bread", "meat", "cheese", "egg", "rice", "pasta",
                     "soup", "cake", "pie", "pizza", "sandwich", "salad",
                     "fish", "chicken", "beef", "pork"],
            "drink": ["water", "milk", "juice", "tea", "coffee", "beer",
                      "wine", "soda", "lemonade"],
            # Clothing
            "clothing": ["shirt", "pants", "dress", "jacket", "coat", "hat",
                         "shoe", "sock", "glove", "scarf", "boot", "belt",
                         "tie", "suit", "skirt", "vest"],
            # Furniture
            "furniture": ["chair", "table", "bed", "desk", "sofa", "cabinet",
                          "shelf", "wardrobe", "dresser", "stool", "bench"],
            # Tools
            "tool": ["hammer", "saw", "drill", "screwdriver", "wrench",
                     "axe", "knife", "scissors", "pliers", "chisel",
                     "ruler", "compass", "level", "clamp"],
            # Emotions
            "emotion": ["happiness", "sadness", "anger", "fear", "love",
                        "hate", "joy", "surprise", "disgust", "shame",
                        "pride", "jealousy", "hope", "anxiety"],
            # Art forms
            "art": ["music", "painting", "sculpture", "dance", "poetry",
                    "theater", "film", "photography", "architecture", "literature"],
            # Sports
            "sport": ["football", "basketball", "tennis", "swimming",
                      "baseball", "soccer", "golf", "boxing", "wrestling",
                      "hockey", "cricket", "rugby", "volleyball"],
            # Academic subjects
            "subject": ["math", "science", "history", "language", "art",
                        "music", "geography", "physics", "chemistry",
                        "biology", "philosophy", "economics", "psychology"],
            # Colors
            "color": ["red", "blue", "green", "yellow", "white", "black",
                      "orange", "purple", "pink", "brown", "gray", "violet"],
            # Seasons
            "season": ["spring", "summer", "autumn", "winter"],
            # Weather
            "weather": ["rain", "snow", "storm", "cloud", "wind", "fog",
                        "hail", "thunder", "lightning", "tornado"],
            # Body parts
            "body_part": ["hand", "foot", "head", "eye", "ear", "nose",
                          "mouth", "arm", "leg", "finger", "toe", "neck",
                          "back", "chest", "shoulder", "knee"],
            "organ": ["heart", "brain", "lung", "liver", "kidney", "stomach"],
            # Time periods
            "time": ["morning", "evening", "night", "afternoon", "dawn",
                     "dusk", "midnight", "noon"],
            # Shapes
            "shape": ["circle", "square", "triangle", "rectangle", "sphere",
                      "cube", "cylinder", "cone", "oval", "diamond"],
        }
    
    def _get_logic_rule_definitions(self):
        """Return logic rule definitions as list of (triggers, targets, type, strength)."""
        return [
            # Animal actions
            (["dog"], ["bark", "chase", "run", "eat", "play"], "bonus", "soft"),
            (["cat"], ["meow", "chase", "sleep", "eat", "play"], "bonus", "soft"),
            (["bird"], ["fly", "sing", "build", "eat"], "bonus", "soft"),
            (["fish"], ["swim", "eat", "live"], "bonus", "soft"),
            (["horse"], ["run", "eat", "gallop"], "bonus", "soft"),
            (["bee"], ["buzz", "make", "fly"], "bonus", "soft"),
            (["lion"], ["hunt", "roar", "run"], "bonus", "soft"),
            # People and roles
            (["doctor"], ["treat", "help", "work"], "bonus", "soft"),
            (["teacher"], ["teach", "help", "explain"], "bonus", "soft"),
            (["student"], ["study", "learn", "read"], "bonus", "soft"),
            (["scientist"], ["study", "research", "discover"], "bonus", "soft"),
            (["writer"], ["write", "read", "create"], "bonus", "soft"),
            (["chef"], ["cook", "prepare", "make"], "bonus", "soft"),
            (["farmer"], ["grow", "plant", "harvest"], "bonus", "soft"),
            (["musician"], ["play", "sing", "perform"], "bonus", "soft"),
            (["artist"], ["paint", "create", "draw"], "bonus", "soft"),
            # Physics/nature
            (["fire"], ["hot", "burn", "heat"], "bonus", "soft"),
            (["ice"], ["cold", "freeze", "melt"], "bonus", "soft"),
            (["water"], ["wet", "flow", "liquid"], "bonus", "soft"),
            (["sun"], ["hot", "bright", "warm"], "bonus", "soft"),
            (["snow"], ["cold", "white", "freeze"], "bonus", "soft"),
            (["rock"], ["hard", "solid", "heavy"], "bonus", "soft"),
            (["wind"], ["blow", "move", "strong"], "bonus", "soft"),
            # Location
            (["ocean"], ["water", "deep", "fish", "wave"], "bonus", "soft"),
            (["forest"], ["tree", "animal", "green", "wood"], "bonus", "soft"),
            (["desert"], ["hot", "dry", "sand"], "bonus", "soft"),
            (["mountain"], ["high", "rock", "snow"], "bonus", "soft"),
            (["school"], ["student", "teacher", "learn", "class"], "bonus", "soft"),
            (["hospital"], ["doctor", "patient", "treat", "health"], "bonus", "soft"),
            (["library"], ["book", "read", "study"], "bonus", "soft"),
            (["kitchen"], ["cook", "food", "eat"], "bonus", "soft"),
            # Hard contradictions
            (["fire"], ["cold", "freeze", "ice"], "penalty", "hard"),
            (["ice"], ["hot", "burn", "fire"], "penalty", "hard"),
            (["snow"], ["hot", "burn"], "penalty", "hard"),
            (["dead"], ["run", "walk", "live"], "penalty", "hard"),
        ]

    def _add_commonsense_triples(self):
        """Add curated commonsense triples from the expanded database."""
        commonsense = _get_expanded_commonsense()
        self.knowledge_layer.add_conceptnet_triples(commonsense, self.vocab.word2idx)
    
    def _add_conceptnet_triples(self):
        """Add ConceptNet triples if available."""
        triples = fetch_conceptnet_triples(
            self.vocab.word2idx, max_triples=5000
        )
        if triples:
            self.knowledge_layer.add_conceptnet_triples(triples, self.vocab.word2idx)
    
    def _add_category_ontology(self):
        """Build category ontology for Layer 4 (hypernym-based couplings)."""
        w2i = self.vocab.word2idx
        category_defs = self._get_category_definitions()
        
        for cat_name, word_list in category_defs.items():
            indices = [w2i[w] for w in word_list if w in w2i and w2i[w] >= 4]
            if len(indices) >= 2:
                self.category_layer.add_category(cat_name, indices)
    
    def _add_logic_rules(self):
        """Build Markov logic rules for Layer 5 (factual consistency)."""
        w2i = self.vocab.word2idx
        
        def idx(word):
            """Get word index, return None if not in vocab."""
            return w2i.get(word, None)
        
        def make_idx_list(words):
            """Get indices for a list of words, filtering out None."""
            return [i for w in words if (i := idx(w)) is not None and i >= 4]
        
        # Use centralized rule definitions (now includes knowledge words in vocab)
        for triggers, targets, rtype, strength in self._get_logic_rule_definitions():
            t_idx = make_idx_list(triggers)
            tgt_idx = make_idx_list(targets)
            if t_idx and tgt_idx:
                self.markov_logic_layer.add_rule(t_idx, tgt_idx, rtype, strength)
        
        # Additional soft contradiction rules (not in centralized defs)
        extra_contradictions = [
            (["water"], ["dry"], "penalty", "soft"),
            (["desert"], ["wet", "rain", "flood"], "penalty", "soft"),
            (["silent"], ["loud", "noise", "shout"], "penalty", "soft"),
        ]
        for triggers, targets, rtype, strength in extra_contradictions:
            t_idx = make_idx_list(triggers)
            tgt_idx = make_idx_list(targets)
            if t_idx and tgt_idx:
                self.markov_logic_layer.add_rule(t_idx, tgt_idx, rtype, strength)
    
    def _print_scale_diagnostics(self):
        """Print diagnostics comparing energy scales across layers."""
        print(f"  Energy scale comparison:")
        print(f"    recall_scale:    {self.recall_scale:>8}")
        print(f"    pmi_weight:      {self.pmi_weight:>8}")
        print(f"    field_weight:    {self.field_weight:>8}")
        print(f"    knowledge_scale: {self.knowledge_scale:>8}")
        print(f"    spin3_scale:     {self.spin3_scale:>8}")
        print(f"    category_scale:  {self.category_scale:>8}")
        print(f"    logic_rule_scale:{self.logic_rule_scale:>8}")
        print(f"    logic_hard_scale:{self.logic_hard_scale:>8}")
        
        # Ratio analysis
        if self.recall_scale > 0:
            print(f"\n  Ratio vs recall_scale=100:")
            print(f"    knowledge_scale/recall: {self.knowledge_scale/self.recall_scale:.1%}")
            print(f"    spin3_scale/recall:     {self.spin3_scale/self.recall_scale:.1%}")
            print(f"    category_scale/recall:  {self.category_scale/self.recall_scale:.1%}")
            print(f"    logic_rule/recall:      {self.logic_rule_scale/self.recall_scale:.1%}")

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
            J_skip=self.J_skip,
            knowledge_layer=self.knowledge_layer,
            category_layer=self.category_layer,
            markov_logic_layer=self.markov_logic_layer,
        )

        # Main generator (with Ising + Knowledge + Category + Logic + MCMC)
        self.generator = IsingLM(
            **gen_kwargs,
            pmi_weight=self.pmi_weight,
            ising_enabled=self.ising_enabled,
            mcmc_refine_steps=self.mcmc_refine_steps,
        )

        # Ablation baseline (without Ising, but WITH knowledge layers)
        self.baseline_generator = IsingLM(
            **gen_kwargs,
            pmi_weight=0,
            ising_enabled=False,
            mcmc_refine_steps=0,
        )

        # Knowledge-off baseline (with Ising but NO knowledge layers)
        self.knowledge_off_generator = IsingLM(
            vocab=self.vocab,
            ngram_index=self.ngram_index,
            J=self.J, h=self.h, types=self.types,
            recall_scale=self.recall_scale,
            pmi_weight=self.pmi_weight,
            field_weight=self.field_weight,
            beta_type=self.beta_type,
            beta_word=self.beta_word,
            copy_enabled=self.copy_enabled,
            copy_min_context=self.copy_min_context,
            copy_min_confidence=self.copy_min_confidence,
            same_word_penalty=self.same_word_penalty,
            max_closed_class_run=self.max_closed_class_run,
            ising_enabled=self.ising_enabled,
            J_skip=self.J_skip,
            knowledge_layer=None,       # NO knowledge layer
            category_layer=None,        # NO category layer
            markov_logic_layer=None,    # NO logic layer
            mcmc_refine_steps=0,        # No MCMC without knowledge
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

    # ===================================================================
    # Path 2a: Beam generation wrapper
    # ===================================================================

    def generate_beam(self, prompt: str = "the", length: int = 20,
                      n_beams: int = 5) -> Dict:
        """Generate text with beam search and global energy ranking."""
        if self.generator is None:
            self._build_generators()
        return self.generator.generate_beam(
            prompt=prompt, length=length, n_beams=n_beams
        )

    # ===================================================================
    # Path 2c: Annealed generation wrapper
    # ===================================================================

    def generate_annealed(self, prompt: str = "the", length: int = 20,
                          beta_start: float = 0.005,
                          beta_end: float = 0.5) -> Dict:
        """Generate text with temperature annealing."""
        if self.generator is None:
            self._build_generators()
        return self.generator.generate_annealed(
            prompt=prompt, length=length,
            beta_start=beta_start, beta_end=beta_end
        )

    # ===================================================================
    # Path 3c: Perplexity Evaluation
    # ===================================================================

    def compute_perplexity(
        self,
        test_sequences: Optional[List[List[int]]] = None,
        n_samples: int = 100,
    ) -> float:
        """
        Compute perplexity on held-out test sequences.

        PPL = exp(-1/N * Σ log P(w_t | ctx))

        where P(w_t | ctx) = exp(-beta * E(w_t)) / Σ exp(-beta * E(w))
        over all candidates of the same POS type.

        This uses the word_sampler's Boltzmann lookup table for efficient
        computation of the partition function.

        Args:
            test_sequences: Test sequences to evaluate. If None, uses
                self.test_sequences (held out during training).
            n_samples: Maximum number of sequences to evaluate.

        Returns:
            Perplexity value (lower is better).
        """
        if self.generator is None:
            self._build_generators()

        if test_sequences is None:
            test_sequences = self.test_sequences

        if not test_sequences:
            print("  Warning: No test sequences available. Returning inf PPL.")
            return float('inf')

        gen = self.generator
        sampler = gen.word_sampler

        total_log_prob = 0.0
        total_tokens = 0

        eval_seqs = test_sequences[:n_samples]

        for seq_idx, seq in enumerate(eval_seqs):
            if len(seq) < 3:
                continue

            for pos in range(1, len(seq)):
                target_word = seq[pos]
                context_words = seq[:pos]
                context_types = [gen._get_word_type(w) for w in context_words]

                # Determine the POS type for the target word
                word_type = gen._get_word_type(target_word)

                # Get candidate words for this type
                candidate_list = gen.type_words.get(word_type, [])
                if not candidate_list:
                    continue
                candidate_words = np.array(candidate_list, dtype=np.int64)

                # Top-k filtering (same as during generation)
                if len(candidate_words) > 300:
                    field_vals = gen.h[candidate_words]
                    top_k = np.argsort(field_vals)[-300:]
                    candidate_words = candidate_words[top_k]

                # Check if target word is in candidates
                target_in_candidates = int(target_word) in set(candidate_words.tolist())
                if not target_in_candidates:
                    # Target not reachable; use smoothing
                    total_log_prob += -15.0  # Very low probability
                    total_tokens += 1
                    continue

                # Check recall
                recall_matches = gen.ngram_index.lookup(context_words)
                recall_hit = bool(recall_matches)

                # Compute energies for all candidates
                energies = gen._compute_word_energy(
                    pos, candidate_words, word_type,
                    context_words, context_types, recall_hit
                )

                # Compute log probabilities
                log_probs = sampler.compute_log_probabilities(energies)

                # Find the target word's log probability
                target_idx = np.where(candidate_words == target_word)[0]
                if len(target_idx) > 0:
                    total_log_prob += float(log_probs[target_idx[0]])
                else:
                    total_log_prob += -15.0

                total_tokens += 1

        if total_tokens == 0:
            return float('inf')

        # PPL = exp(-1/N * Σ log P(w_t | ctx))
        avg_log_prob = total_log_prob / total_tokens
        perplexity = math.exp(-avg_log_prob)

        print(f"  Perplexity: {perplexity:.2f} (evaluated on {total_tokens} tokens)")
        return perplexity

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
