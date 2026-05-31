"""
Vocabulary builder and POS type system for the Integer Language Model.

Combines word-frequency counting, POS tag assignment, and tokenization
into a single module. No external NLP dependencies — pure rule-based
POS assignment using English morphological heuristics.
"""

import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional


# ===========================================================================
# COARSE POS TAGS (13 categories)
# ===========================================================================

COARSE_POS = [
    "NOUN",     # nouns
    "VERB",     # verbs
    "ADJ",      # adjectives
    "ADV",      # adverbs
    "DET",      # determiners
    "PREP",     # prepositions
    "PRON",     # pronouns
    "AUX",      # auxiliaries / modals
    "CONJ",     # conjunctions
    "PART",     # particles
    "NUM",      # numbers
    "PUNCT",    # punctuation
    "X",        # other / unknown
]

POS2IDX = {tag: i for i, tag in enumerate(COARSE_POS)}
IDX2POS = {i: tag for i, tag in enumerate(COARSE_POS)}
N_POS = len(COARSE_POS)

# Priority for choosing the "primary" POS when a word has multiple tags
# (Closed-class tags get higher priority = lower number)
TAG_PRIORITY = {
    POS2IDX["PUNCT"]: 0, POS2IDX["DET"]: 1, POS2IDX["PRON"]: 2,
    POS2IDX["AUX"]: 3, POS2IDX["CONJ"]: 4, POS2IDX["PART"]: 5,
    POS2IDX["PREP"]: 6, POS2IDX["NUM"]: 7, POS2IDX["ADV"]: 8,
    POS2IDX["ADJ"]: 9, POS2IDX["NOUN"]: 10, POS2IDX["VERB"]: 11,
    POS2IDX["X"]: 12,
}


# ===========================================================================
# WORD SETS FOR CLOSED-CLASS CATEGORIES
# ===========================================================================

_DET_WORDS = frozenset({
    "the", "a", "an", "this", "that", "these", "those",
    "some", "any", "all", "each", "every", "no", "both",
    "either", "neither", "my", "your", "his", "her", "its",
    "our", "their",
})

_PRON_WORDS = frozenset({
    "i", "me", "you", "he", "him", "she", "her", "it",
    "we", "us", "they", "them", "myself", "yourself",
    "himself", "herself", "itself", "ourselves",
    "themselves", "who", "whom", "which", "what",
})

_AUX_WORDS = frozenset({
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did",
    "can", "could", "will", "would", "shall", "should",
    "may", "might", "must",
})

_CONJ_WORDS = frozenset({
    "and", "or", "but", "nor", "for", "yet", "so",
    "although", "because", "since", "unless", "while",
    "if", "then", "when", "where", "whether",
})

_PART_WORDS = frozenset({
    "to", "not", "up", "down", "out", "off", "on",
    "in", "away", "over",
})

_PREP_WORDS = frozenset({
    "of", "in", "to", "for", "with", "on", "at", "from",
    "by", "about", "as", "into", "through", "during",
    "before", "after", "above", "below", "between",
    "under", "over", "against", "within", "without",
    "among", "upon", "toward", "towards",
})


# ===========================================================================
# VOCABULARY CLASS
# ===========================================================================

class Vocabulary:
    """
    Word vocabulary with POS type assignments.

    Builds a fixed-size vocabulary from text data, assigns POS types
    using morphological heuristics, and provides tokenization.

    Attributes:
        words: List of vocabulary words (index = word ID).
        word2idx: Dict mapping word -> ID.
        idx2word: Dict mapping ID -> word.
        word_freq: Array of word frequencies, shape (V,).
        word_pos: Array of primary POS type per word, shape (V,).
        word_pos_set: Dict mapping word ID -> set of possible POS types.
    """

    SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

    def __init__(
        self,
        max_size: int = 2000,
        min_freq: int = 5,
        max_seq_len: int = 30,
    ):
        self.max_size = max_size
        self.min_freq = min_freq
        self.max_seq_len = max_seq_len

        self.words: List[str] = []
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freq: np.ndarray = np.array([], dtype=np.int32)
        self.word_pos: np.ndarray = np.array([], dtype=np.int32)
        self.word_pos_set: Dict[int, set] = {}
        self.V: int = 0

    def build(self, texts: List[str]) -> "Vocabulary":
        """Build vocabulary from raw texts."""
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)

        # Filter by minimum frequency
        filtered = {w: c for w, c in word_counts.items() if c >= self.min_freq}
        sorted_words = sorted(filtered.items(), key=lambda x: -x[1])

        # Build word lists with special tokens
        self.words = list(self.SPECIAL_TOKENS) + [w for w, _ in sorted_words[:self.max_size - 4]]
        self.word2idx = {w: i for i, w in enumerate(self.words)}
        self.idx2word = {i: w for i, w in enumerate(self.words)}
        self.V = len(self.words)

        # Build frequency array
        self.word_freq = np.zeros(self.V, dtype=np.int32)
        for w, c in sorted_words[:self.max_size - 4]:
            if w in self.word2idx:
                self.word_freq[self.word2idx[w]] = c

        # Assign POS types
        self._assign_pos_types()

        return self

    def _assign_pos_types(self):
        """Assign POS types to all vocabulary words using rule-based heuristics."""
        self.word_pos = np.zeros(self.V, dtype=np.int32)
        self.word_pos_set = {}

        for idx in range(self.V):
            word = self.words[idx]

            # Special tokens
            if idx < 4:
                self.word_pos[idx] = POS2IDX["X"]
                self.word_pos_set[idx] = {POS2IDX["X"]}
                continue

            tags = self._classify_pos(word)
            self.word_pos_set[idx] = set(tags)

            # Pick primary POS (most specific / highest priority)
            primary = min(tags, key=lambda t: TAG_PRIORITY.get(t, 99))
            self.word_pos[idx] = primary

    @staticmethod
    def _classify_pos(word: str) -> List[int]:
        """Rule-based POS classification for an English word."""
        w = word.lower()
        tags = []

        # Punctuation
        if not any(c.isalnum() for c in w):
            return [POS2IDX["PUNCT"]]

        # Numbers
        if w.replace(".", "").replace(",", "").replace("-", "").isdigit():
            tags.append(POS2IDX["NUM"])

        # Closed-class sets (exact match)
        if w in _DET_WORDS:
            tags.append(POS2IDX["DET"])
        if w in _PRON_WORDS:
            tags.append(POS2IDX["PRON"])
        if w in _AUX_WORDS:
            tags.append(POS2IDX["AUX"])
        if w in _CONJ_WORDS:
            tags.append(POS2IDX["CONJ"])
        if w in _PART_WORDS:
            tags.append(POS2IDX["PART"])
        if w in _PREP_WORDS:
            tags.append(POS2IDX["PREP"])

        # Morphological heuristics for open-class words
        if w.endswith("ly"):
            tags.append(POS2IDX["ADV"])
        if (w.endswith("ful") or w.endswith("less") or w.endswith("ous") or
            w.endswith("ive") or w.endswith("able") or w.endswith("ible") or
            w.endswith("al") or w.endswith("ial") or w.endswith("ent") or
            w.endswith("ant") or w.endswith("ic") or w.endswith("ical")):
            tags.append(POS2IDX["ADJ"])
        if (w.endswith("ing") or w.endswith("ed") or w.endswith("ize") or
            w.endswith("ify") or w.endswith("ate") or w.endswith("en") or
            w.endswith("es") or w.endswith("ied")):
            tags.append(POS2IDX["VERB"])
        if (w.endswith("tion") or w.endswith("sion") or w.endswith("ment") or
            w.endswith("ness") or w.endswith("ity") or w.endswith("ism") or
            w.endswith("ist") or w.endswith("ence") or w.endswith("ance") or
            w.endswith("er") or w.endswith("or")):
            tags.append(POS2IDX["NOUN"])

        # Defaults: most English words can be nouns or verbs
        if POS2IDX["NOUN"] not in tags and len(w) >= 2 and w[0].isalpha():
            tags.append(POS2IDX["NOUN"])
        if POS2IDX["VERB"] not in tags and len(w) >= 3 and w[0].isalpha() and POS2IDX["AUX"] not in tags:
            tags.append(POS2IDX["VERB"])

        return tags if tags else [POS2IDX["X"]]

    def tokenize(self, texts: List[str]) -> List[List[int]]:
        """Tokenize texts into word ID sequences."""
        sequences = []
        for text in texts:
            words = text.lower().split()
            ids = [self.word2idx.get(w, 1) for w in words]
            ids = [i for i in ids if i >= 4]  # Skip special tokens
            if len(ids) >= 2:
                sequences.append(ids[:self.max_seq_len])
        return sequences

    def decode(self, ids: List[int]) -> str:
        """Decode word IDs back to text."""
        return " ".join(self.idx2word.get(i, "<unk>") for i in ids)

    def pos_distribution(self) -> Dict[str, int]:
        """Return count of words per POS category."""
        counts = {}
        for idx in range(self.V):
            name = IDX2POS.get(int(self.word_pos[idx]), "X")
            counts[name] = counts.get(name, 0) + 1
        return counts
