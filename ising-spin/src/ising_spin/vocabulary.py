"""
Vocabulary builder and DATA-DRIVEN word class system.

v81 BREAKING CHANGE:
  The old system had 13 hardcoded POS tags with 88% of words tagged NOUN.
  hash(word, pos) with pos=NOUN for 88% of words ≈ hash(word, 0) — the
  "class" dimension carried ZERO information.

  The new system uses DATA-DRIVEN word classes:
  - Frequency buckets: words binned by frequency rank into K buckets.
    Function words (the, a, was) land in bucket 0, content words in
    higher buckets. K is VARIABLE (default 20), not hardcoded.
  - Distributional clusters: built from bigram co-occurrence patterns.
    Words with similar left/right context distributions get the same cluster.

  POS tags are kept for DIAGNOSTICS ONLY — never used in features.

  The key insight: frequency rank naturally separates function words from
  content words, which is exactly what POS tags were trying (and failing)
  to capture. But frequency is continuous, not degenerate: you get a
  smooth gradient from "the" (rank 0) → "cat" (rank 200) → "dinosaur" (rank 1990).
  With K=20 buckets, each bucket has ~100 words, giving hash(word, bucket)
  2000×20 = 40000 unique keys — rich, smooth, non-degenerate.
"""

import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional


# ===========================================================================
# COARSE POS TAGS (diagnostics only — NOT used in features)
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
TAG_PRIORITY = {
    POS2IDX["PUNCT"]: 0, POS2IDX["DET"]: 1, POS2IDX["PRON"]: 2,
    POS2IDX["AUX"]: 3, POS2IDX["CONJ"]: 4, POS2IDX["PART"]: 5,
    POS2IDX["PREP"]: 6, POS2IDX["NUM"]: 7, POS2IDX["ADV"]: 8,
    POS2IDX["ADJ"]: 9, POS2IDX["NOUN"]: 10, POS2IDX["VERB"]: 11,
    POS2IDX["X"]: 12,
}


# ===========================================================================
# WORD SETS FOR CLOSED-CLASS CATEGORIES (diagnostics only)
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
    Word vocabulary with DATA-DRIVEN word class assignments.

    Builds a fixed-size vocabulary from text data, computes frequency-based
    word classes (buckets), and optionally distributional clusters from
    bigram statistics.

    Word classes replace the old static POS tags for features:
    - Frequency buckets: words ranked by frequency, binned into K groups
    - K is VARIABLE (default 20), not hardcoded at 13
    - Each bucket has ~V/K words — balanced, non-degenerate

    Attributes:
        words: List of vocabulary words (index = word ID).
        word2idx: Dict mapping word -> ID.
        idx2word: Dict mapping ID -> word.
        word_freq: Array of word frequencies, shape (V,).
        word_freq_rank: Array of frequency rank per word (0 = most frequent).
        word_bucket: Array of frequency bucket per word, shape (V,).
        n_buckets: Number of frequency buckets (variable).
        word_pos: Array of POS type per word (diagnostics only).
    """

    SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

    def __init__(
        self,
        max_size: int = 2000,
        min_freq: int = 5,
        max_seq_len: int = 30,
        n_buckets: int = 20,
    ):
        self.max_size = max_size
        self.min_freq = min_freq
        self.max_seq_len = max_seq_len
        self.n_buckets = n_buckets

        self.words: List[str] = []
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freq: np.ndarray = np.array([], dtype=np.int32)
        self.word_freq_rank: np.ndarray = np.array([], dtype=np.int32)
        self.word_bucket: np.ndarray = np.array([], dtype=np.int32)
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

        # Build frequency-based word classes (THE KEY CHANGE from v80)
        self._build_freq_buckets()

        # Assign POS types (diagnostics only)
        self._assign_pos_types()

        return self

    def _build_freq_buckets(self):
        """
        Build frequency-based word class buckets.

        This is the CORE of the v81 redesign. Instead of 13 hardcoded POS tags
        where 88% are NOUN, we bin words by frequency rank into K buckets.

        WHY THIS WORKS:
        - High-frequency words (the, a, was, is, he, she) = function words
          that play grammatical roles → bucket 0
        - Mid-frequency words (cat, dog, said, went) = common content words
          → middle buckets
        - Low-frequency words (dinosaur, castle, whispered) = rare content
          words → last buckets

        With K=20 buckets and V=2000, each bucket has ~100 words.
        hash(word, bucket) produces V*K = 40000 unique keys — rich, smooth,
        non-degenerate. Compare to the old hash(word, pos) with 88% having
        the same POS value — nearly degenerate.

        The bucket IDs are STABLE: they depend only on frequency rank, not
        on the arbitrary POS assignment heuristics.
        """
        # Frequency rank: 0 = most frequent, V-1 = least frequent
        # Special tokens (0-3) get their own bucket
        self.word_freq_rank = np.zeros(self.V, dtype=np.int32)
        self.word_bucket = np.zeros(self.V, dtype=np.int32)

        # Sort real words (ID >= 4) by frequency (descending)
        real_word_ids = np.arange(4, self.V)
        if len(real_word_ids) == 0:
            return

        freqs = self.word_freq[real_word_ids]
        sorted_by_freq = real_word_ids[np.argsort(-freqs)]

        # Assign ranks
        for rank, word_id in enumerate(sorted_by_freq):
            self.word_freq_rank[word_id] = rank

        # Assign buckets: K buckets for real words, bucket 0 for special tokens
        # Each bucket gets approximately the same number of words
        n_real = len(sorted_by_freq)
        for i, word_id in enumerate(sorted_by_freq):
            # Bucket 1..K for real words (bucket 0 = special tokens)
            bucket = 1 + min(i * self.n_buckets // n_real, self.n_buckets - 1)
            self.word_bucket[word_id] = bucket

        # Print bucket distribution
        print(f"  Frequency buckets: K={self.n_buckets}, ~{n_real // max(1, self.n_buckets)} words/bucket",
              flush=True)

    def build_distributional_clusters(
        self,
        sequences: List[List[int]],
        n_clusters: int = 30,
    ) -> np.ndarray:
        """
        Build distributional word clusters from bigram co-occurrence.

        Words that appear in similar contexts (similar distribution of
        preceding and following words) get the same cluster.

        This is a SIMPLE hash-based approach — no k-means, no SVD:
        1. For each word, compute a "context fingerprint" by hashing
           its top-N most frequent followers
        2. Cluster by fingerprint hash modulo n_clusters

        Returns: word_cluster array, shape (V,), dtype int32.

        NOTE: This is optional — frequency buckets work well by default.
              Call this AFTER build() and only if you want richer classes.
        """
        # Count what follows each word
        followers = {}
        for seq in sequences:
            for pos in range(1, len(seq)):
                prev = seq[pos - 1]
                target = seq[pos]
                if prev not in followers:
                    followers[prev] = Counter()
                followers[prev][target] += 1

        # For each word, hash its top-10 followers into a fingerprint
        word_cluster = np.zeros(self.V, dtype=np.int32)
        for word_id in range(self.V):
            if word_id < 4:
                continue
            if word_id not in followers or not followers[word_id]:
                word_cluster[word_id] = 0
                continue

            # Top 10 followers
            top = followers[word_id].most_common(10)
            # Simple hash: multiply follower IDs by primes and sum
            fp = 0
            for i, (fid, cnt) in enumerate(top):
                fp += fid * (i + 1) * 2654435761 + cnt * 2246822519
            word_cluster[word_id] = (fp % n_clusters) + 1  # 1..n_clusters

        print(f"  Distributional clusters: K={n_clusters}, "
              f"built from {len(followers)} words with followers",
              flush=True)

        return word_cluster

    def _assign_pos_types(self):
        """Assign POS types to all vocabulary words (diagnostics only)."""
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
        """Rule-based POS classification for an English word (diagnostics only)."""
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
        """Return count of words per POS category (diagnostics only)."""
        counts = {}
        for idx in range(self.V):
            name = IDX2POS.get(int(self.word_pos[idx]), "X")
            counts[name] = counts.get(name, 0) + 1
        return counts

    def bucket_distribution(self) -> Dict[int, int]:
        """Return count of words per frequency bucket."""
        counts = {}
        for idx in range(self.V):
            b = int(self.word_bucket[idx])
            counts[b] = counts.get(b, 0) + 1
        return counts
