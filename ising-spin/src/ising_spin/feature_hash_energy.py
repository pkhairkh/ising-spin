"""
Dynamic Feature-Hashed Integer Energy Table — v81 DATA-DRIVEN CLASSES.

ARCHITECTURE CHANGE (v81):
  The v80 "fix" replaced hash(pos, pos) with hash(word, pos), but this was
  STILL BROKEN because 88% of words were tagged NOUN. When nearly every word
  has the same class label, hash(word, class) ≈ hash(word, 0) — the class
  dimension carries ZERO discriminative information.

  v81 ELIMINATES ALL STATIC POS DEPENDENCY:
  - Word classes are DATA-DRIVEN (frequency buckets), not rule-based POS
  - Number of classes K is VARIABLE (default 20), not hardcoded at 13
  - Each bucket has ~V/K words — balanced, non-degenerate
  - hash(word, bucket) has V*K = 40000+ unique keys (vs V*1 ≈ 2000 with POS)

  WHY FREQUENCY BUCKETS WORK:
  Bucket 0 (special tokens): <pad>, <unk>, <bos>, <eos>
  Bucket 1 (highest freq): the, a, was, is, he, she, it, they...
    → These ARE the function words (DET, PRON, AUX) that POS was trying to tag
  Bucket 2-5: and, but, not, to, in, on, with, for, at, from...
    → Prepositions, conjunctions, particles
  Bucket 6-10: said, went, came, had, could, would, little, good...
    → Common verbs, adjectives
  Bucket 11-20: cat, dog, house, tree, play, run, eat, walk, happy...
    → Content words: nouns, verbs, adjectives by frequency tier

  The frequency gradient naturally captures the functional/content distinction
  that POS was designed for — but WITHOUT the degenerate 88%-NOUN problem.

AVAILABLE FEATURES:
  Lexical (pure word ID, no class dependency):
    LexBigramFeature    — hash(prev_word, cand_word)     token-specific pairs
    LexSkipFeature      — hash(prev2_word, cand_word)    skip-gram lexical
    LexTrigramFeature   — hash(prev2_word, prev_word, cand_word) lexical 3-gram

  Class-word mixed (DATA-DRIVEN, replaces all POS features):
    ClassWordBigramFeature — hash(prev_class, cand_word)  class→word transitions
    WordClassBigramFeature — hash(prev_word, cand_class)  word→class transitions
    ClassWordSkipFeature   — hash(prev2_class, cand_word) class→word at distance 2
    WordClassSkipFeature   — hash(prev2_word, cand_class) word→class at distance 2
    ClassTrigramFeature    — hash(prev2_class, prev_class, cand_class) class 3-gram

DEFAULT FEATURE SET:
  LexBigramFeature, WordClassBigramFeature, ClassWordBigramFeature,
  LexSkipFeature, ClassTrigramFeature, LexTrigramFeature

  These 6 features use VARIABLE data-driven classes instead of static POS.

ADD YOUR OWN FEATURE:
  1. Subclass FeatureSpec
  2. Implement get_hash_args_batch() and get_hash_args_nce()
  3. Call energy_table.add_feature(your_feature)

All tables trained via integer NCE with CLASS-balanced negatives.
O(1) per candidate. Pure integer arithmetic. No neural nets.
"""

import numpy as np
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Hash functions — deterministic, collision-resistant, integer-only
# ---------------------------------------------------------------------------

_P1 = np.int64(2654435761)
_P2 = np.int64(2246822519)
_P3 = np.int64(3266489917)
_P4 = np.int64(3367900313)
_MASK32 = np.int64(0xFFFFFFFF)


def _hash2_vec(a: np.ndarray, b: np.ndarray, h_idx: int, P: int) -> np.ndarray:
    """Vectorized double-hash for (a, b) pairs."""
    val = (a.astype(np.int64) * _P1 + b.astype(np.int64) * _P2
           + np.int64(h_idx) * _P3) & _MASK32
    return val % np.int64(P)


def _hash3_vec(a: np.ndarray, b: np.ndarray, c: np.ndarray,
               h_idx: int, P: int) -> np.ndarray:
    """Vectorized triple-hash for (a, b, c) triples."""
    val = (a.astype(np.int64) * _P1 + b.astype(np.int64) * _P2
           + c.astype(np.int64) * _P4 + np.int64(h_idx) * _P3) & _MASK32
    return val % np.int64(P)


def _hash2(a: int, b: int, h_idx: int, P: int) -> int:
    """Scalar double-hash."""
    val = (a * 2654435761 + b * 2246822519 + h_idx * 3266489917) & 0xFFFFFFFF
    return int(val % P)


def _hash3(a: int, b: int, c: int, h_idx: int, P: int) -> int:
    """Scalar triple-hash."""
    val = (a * 2654435761 + b * 2246822519 + c * 3367900313
           + h_idx * 3266489917) & 0xFFFFFFFF
    return int(val % P)


# ---------------------------------------------------------------------------
# FeatureSpec — base class for energy features
# ---------------------------------------------------------------------------

class FeatureSpec:
    """
    Base class for energy features. Each feature is self-contained:
    its own hash tables, learning rate (eta), clipping range, and weight.

    The feature's contribution to the total energy is:
        E_feature = weight * sum_h(table[h][hash(inputs, h)])

    IMPORTANT v81 CHANGE: get_hash_args_batch/nce now receive `word_class`
    instead of `word_pos`. The class array is DATA-DRIVEN (frequency buckets),
    not static POS. The number of classes is VARIABLE, not hardcoded.
    """

    def __init__(
        self,
        name: str,
        n_hashes: int = 2,
        table_size: int = 65537,
        eta: int = 1,
        clip: int = 100,
        weight: float = 1.0,
    ):
        self.name = name
        self.n_hashes = n_hashes
        self.table_size = table_size
        self.eta = eta
        self.clip = clip
        self.weight = weight
        # Initialize hash tables (all zeros = no prior)
        self.tables = [np.zeros(table_size, dtype=np.int32) for _ in range(n_hashes)]

    # -------------------------------------------------------------------
    # Subclasses MUST implement these two methods
    # -------------------------------------------------------------------

    def get_hash_args_batch(
        self,
        context: List[int],
        candidates: np.ndarray,
        word_class: np.ndarray,
    ) -> Optional[Tuple]:
        """
        Extract hash input arrays from context + candidates (generation-time).

        Args:
            context: List of context word IDs.
            candidates: Array of candidate word IDs, shape (K,).
            word_class: Array of class ID per word, shape (V,).
                        In v81 this is frequency bucket, NOT POS.

        Returns: tuple of (a, b) or (a, b, c) as np.int64 arrays, shape (K,)
                 Or None if this feature doesn't apply (e.g., context too short).
        """
        raise NotImplementedError(
            f"{self.name}: get_hash_args_batch() not implemented"
        )

    def get_hash_args_nce(
        self,
        prev_words: np.ndarray,
        prev_class: np.ndarray,
        right_words: np.ndarray,
        right_class: np.ndarray,
        prev2_words: Optional[np.ndarray] = None,
        prev2_class: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> Optional[Tuple]:
        """
        Extract hash input arrays from NCE training batch.

        Args:
            prev_words:   shape (N,) — previous word IDs
            prev_class:   shape (N,) — previous word class IDs
            right_words:  shape (N,) — target word IDs (positive or negative)
            right_class:  shape (N,) — target/negative class IDs
            prev2_words:  shape (N,) — word at t-2 (0 if none), or None
            prev2_class:  shape (N,) — class at t-2 (0 if none), or None
            mask:         shape (N,) — bool, True where prev2 is valid

        Returns: tuple of arrays for hashing, or None if inapplicable.
        """
        raise NotImplementedError(
            f"{self.name}: get_hash_args_nce() not implemented"
        )

    # -------------------------------------------------------------------
    # Provided methods — energy computation
    # -------------------------------------------------------------------

    def energy_batch(
        self,
        context: List[int],
        candidates: np.ndarray,
        word_class: np.ndarray,
    ) -> np.ndarray:
        """Compute energy contribution for all candidates."""
        args = self.get_hash_args_batch(context, candidates, word_class)
        if args is None:
            return np.zeros(len(candidates), dtype=np.int64)
        K = len(candidates)
        e = np.zeros(K, dtype=np.int64)
        for h in range(self.n_hashes):
            if len(args) == 2:
                slots = _hash2_vec(args[0], args[1], h, self.table_size)
            else:
                slots = _hash3_vec(args[0], args[1], args[2], h, self.table_size)
            e += self.tables[h][slots]
        return e

    def energy_scalar(
        self,
        context: List[int],
        candidate: int,
        word_class: np.ndarray,
    ) -> int:
        """Compute energy for a single candidate."""
        args = self.get_hash_args_batch(context, np.array([candidate], dtype=np.int64), word_class)
        if args is None:
            return 0
        e = 0
        for h in range(self.n_hashes):
            if len(args) == 2:
                slot = _hash2(int(args[0][0]), int(args[1][0]), h, self.table_size)
            else:
                slot = _hash3(int(args[0][0]), int(args[1][0]), int(args[2][0]), h, self.table_size)
            e += int(self.tables[h][slot])
        return e

    # -------------------------------------------------------------------
    # Provided methods — NCE training
    # -------------------------------------------------------------------

    def nce_positive(
        self,
        prev_words: np.ndarray,
        prev_class: np.ndarray,
        targets: np.ndarray,
        target_class: np.ndarray,
        prev2_words: Optional[np.ndarray] = None,
        prev2_class: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ):
        """Apply positive NCE update (real pairs): table[hash] -= eta."""
        args = self.get_hash_args_nce(
            prev_words, prev_class, targets, target_class,
            prev2_words, prev2_class, mask,
        )
        if args is None:
            return
        for h in range(self.n_hashes):
            if len(args) == 2:
                slots = _hash2_vec(args[0], args[1], h, self.table_size)
            else:
                slots = _hash3_vec(args[0], args[1], args[2], h, self.table_size)
            np.add.at(self.tables[h], slots, -self.eta)

    def nce_negative(
        self,
        prev_words: np.ndarray,
        prev_class: np.ndarray,
        neg_words: np.ndarray,
        neg_class: np.ndarray,
        prev2_words: Optional[np.ndarray] = None,
        prev2_class: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ):
        """Apply negative NCE update (corrupt pairs): table[hash] += eta."""
        args = self.get_hash_args_nce(
            prev_words, prev_class, neg_words, neg_class,
            prev2_words, prev2_class, mask,
        )
        if args is None:
            return
        for h in range(self.n_hashes):
            if len(args) == 2:
                slots = _hash2_vec(args[0], args[1], h, self.table_size)
            else:
                slots = _hash3_vec(args[0], args[1], args[2], h, self.table_size)
            np.add.at(self.tables[h], slots, self.eta)

    def clip_tables(self):
        """Clip all table values to prevent energy explosion."""
        for h in range(self.n_hashes):
            np.clip(self.tables[h], -self.clip, self.clip, out=self.tables[h])

    def statistics(self) -> Dict:
        """Return feature statistics."""
        all_vals = np.concatenate([t.ravel() for t in self.tables])
        return {
            'name': self.name,
            'weight': self.weight,
            'eta': self.eta,
            'clip': self.clip,
            'table_size': self.table_size,
            'n_hashes': self.n_hashes,
            'nnz': int(np.count_nonzero(all_vals)),
            'range': (int(all_vals.min()), int(all_vals.max())),
            'mean': float(all_vals.mean()),
            'std': float(all_vals.std()),
            'memory_kb': sum(t.nbytes for t in self.tables) / 1024,
        }

    def __repr__(self):
        return (f"{self.__class__.__name__}(name={self.name!r}, "
                f"n_hashes={self.n_hashes}, table_size={self.table_size}, "
                f"eta={self.eta}, clip={self.clip}, weight={self.weight})")


# ===========================================================================
# CONCRETE FEATURE IMPLEMENTATIONS
# ===========================================================================

# ---------------------------------------------------------------------------
# Lexical features (pure word IDs, NO class dependency)
# ---------------------------------------------------------------------------

class LexBigramFeature(FeatureSpec):
    """
    hash(prev_word, cand_word) — lexical bigram transitions.

    The main workhorse: token-specific pairs like ("the", "cat"), ("a", "dog").
    With V=2000 and table_size=65537, about 6% collision rate — enough for
    smooth generalization without losing too much specificity.

    This feature has NO class dependency — it's pure lexical.
    """

    def __init__(self, n_hashes=3, table_size=65537, eta=1, clip=100, weight=1.0):
        super().__init__("lex_bi", n_hashes, table_size, eta, clip, weight)

    def get_hash_args_batch(self, context, candidates, word_class):
        if not context:
            return None
        K = len(candidates)
        prev = np.full(K, context[-1], dtype=np.int64)
        return (prev, candidates.astype(np.int64))

    def get_hash_args_nce(self, prev_words, prev_class, right_words, right_class,
                          prev2_words=None, prev2_class=None, mask=None):
        return (prev_words, right_words)


class LexSkipFeature(FeatureSpec):
    """
    hash(prev2_word, cand_word) — skip-gram lexical.

    Captures patterns where a word at distance 2 predicts the current word.
    Example: "the cat sat" → hash("the", "sat") captures DET→VERB at distance 2.
    Pure lexical — no class dependency.
    """

    def __init__(self, n_hashes=2, table_size=65537, eta=1, clip=80, weight=0.3):
        super().__init__("lex_skip", n_hashes, table_size, eta, clip, weight)

    def get_hash_args_batch(self, context, candidates, word_class):
        if len(context) < 2:
            return None
        K = len(candidates)
        prev2 = np.full(K, context[-2], dtype=np.int64)
        return (prev2, candidates.astype(np.int64))

    def get_hash_args_nce(self, prev_words, prev_class, right_words, right_class,
                          prev2_words=None, prev2_class=None, mask=None):
        if prev2_words is None:
            return None
        if mask is None:
            return (prev2_words, right_words)
        if not np.any(mask):
            return None
        return (prev2_words[mask], right_words[mask])


class LexTrigramFeature(FeatureSpec):
    """
    hash(prev2_word, prev_word, cand_word) — lexical trigram.

    Three-word token patterns: "once upon a", "there was a", etc.
    Much more specific than bigrams — captures local collocations.
    Pure lexical — no class dependency.
    """

    def __init__(self, n_hashes=2, table_size=65537, eta=1, clip=80, weight=0.3):
        super().__init__("lex_tri", n_hashes, table_size, eta, clip, weight)

    def get_hash_args_batch(self, context, candidates, word_class):
        if len(context) < 2:
            return None
        K = len(candidates)
        prev2 = np.full(K, context[-2], dtype=np.int64)
        prev1 = np.full(K, context[-1], dtype=np.int64)
        return (prev2, prev1, candidates.astype(np.int64))

    def get_hash_args_nce(self, prev_words, prev_class, right_words, right_class,
                          prev2_words=None, prev2_class=None, mask=None):
        if prev2_words is None:
            return None
        if mask is None:
            return (prev2_words, prev_words, right_words)
        if not np.any(mask):
            return None
        return (prev2_words[mask], prev_words[mask], right_words[mask])


# ---------------------------------------------------------------------------
# Class-word mixed features (DATA-DRIVEN, replaces ALL POS features)
#
# Key difference from v80: `class` is frequency bucket (K=20, balanced)
# NOT POS tag (K=13, 88% NOUN = degenerate).
# ---------------------------------------------------------------------------

class ClassWordBigramFeature(FeatureSpec):
    """
    hash(prev_class, cand_word) — class→word transitions.

    REPLACES PosWordBigramFeature. Instead of hash(pos, word) where pos
    was 88% NOUN (degenerate), this uses hash(bucket, word) where bucket
    is a DATA-DRIVEN frequency class with ~V/K words per class.

    With K=20 buckets: 20 distinct class labels, each with ~100 words.
    hash(bucket, word) has 20*V = 40000 unique keys — 20x richer than
    the degenerate hash(POS_NOUN, word) that collapsed 88% of words.

    Learns: "after function words (bucket 1), 'the' is common"
            "after content words (bucket 10), specific nouns follow"
    """

    def __init__(self, n_hashes=2, table_size=65537, eta=1, clip=50, weight=0.5):
        super().__init__("cls_word_bi", n_hashes, table_size, eta, clip, weight)

    def get_hash_args_batch(self, context, candidates, word_class):
        if not context:
            return None
        K = len(candidates)
        prev_class = np.full(K, int(word_class[context[-1]]), dtype=np.int64)
        return (prev_class, candidates.astype(np.int64))

    def get_hash_args_nce(self, prev_words, prev_class, right_words, right_class,
                          prev2_words=None, prev2_class=None, mask=None):
        return (prev_class, right_words)


class WordClassBigramFeature(FeatureSpec):
    """
    hash(prev_word, cand_class) — word→class transitions.

    REPLACES WordPosBigramFeature. Instead of hash(word, pos) where pos
    was 88% NOUN, this uses hash(word, bucket) where bucket is balanced.

    With K=20 buckets and V=2000, hash(word, bucket) produces 2000*20 =
    40000 unique keys. The frequency bucket naturally captures:
    - "the" → bucket 1 → predicts function-word followers
    - "cat" → bucket 8 → predicts content-word followers

    The collisions in a 65537-slot table create SMOOTH generalization:
    words that share a frequency tier share similar transition patterns.
    """

    def __init__(self, n_hashes=2, table_size=65537, eta=1, clip=50, weight=0.5):
        super().__init__("word_cls_bi", n_hashes, table_size, eta, clip, weight)

    def get_hash_args_batch(self, context, candidates, word_class):
        if not context:
            return None
        K = len(candidates)
        prev = np.full(K, context[-1], dtype=np.int64)
        cand_class = word_class[candidates].astype(np.int64)
        return (prev, cand_class)

    def get_hash_args_nce(self, prev_words, prev_class, right_words, right_class,
                          prev2_words=None, prev2_class=None, mask=None):
        return (prev_words, right_class)


class ClassWordSkipFeature(FeatureSpec):
    """
    hash(prev2_class, cand_word) — class→word at distance 2.

    Like ClassWordBigramFeature but at skip distance.
    With balanced K=20 classes, this is much richer than the old
    PosWordSkipFeature where 88% of prev2_pos was NOUN.
    """

    def __init__(self, n_hashes=2, table_size=65537, eta=1, clip=50, weight=0.3):
        super().__init__("cls_word_skip", n_hashes, table_size, eta, clip, weight)

    def get_hash_args_batch(self, context, candidates, word_class):
        if len(context) < 2:
            return None
        K = len(candidates)
        prev2_class = np.full(K, int(word_class[context[-2]]), dtype=np.int64)
        return (prev2_class, candidates.astype(np.int64))

    def get_hash_args_nce(self, prev_words, prev_class, right_words, right_class,
                          prev2_words=None, prev2_class=None, mask=None):
        if prev2_class is None:
            return None
        if mask is None:
            return (prev2_class, right_words)
        if not np.any(mask):
            return None
        return (prev2_class[mask], right_words[mask])


class WordClassSkipFeature(FeatureSpec):
    """
    hash(prev2_word, cand_class) — word→class at distance 2.

    Like WordClassBigramFeature but at skip distance.
    Replaces the degenerate WordPosSkipFeature.
    """

    def __init__(self, n_hashes=2, table_size=65537, eta=1, clip=50, weight=0.3):
        super().__init__("word_cls_skip", n_hashes, table_size, eta, clip, weight)

    def get_hash_args_batch(self, context, candidates, word_class):
        if len(context) < 2:
            return None
        K = len(candidates)
        prev2 = np.full(K, context[-2], dtype=np.int64)
        cand_class = word_class[candidates].astype(np.int64)
        return (prev2, cand_class)

    def get_hash_args_nce(self, prev_words, prev_class, right_words, right_class,
                          prev2_words=None, prev2_class=None, mask=None):
        if prev2_words is None:
            return None
        if mask is None:
            return (prev2_words, right_class)
        if not np.any(mask):
            return None
        return (prev2_words[mask], right_class[mask])


class ClassTrigramFeature(FeatureSpec):
    """
    hash(prev2_class, prev_class, cand_class) — class trigram.

    REPLACES PosTrigramFeature. With K=20 balanced classes, there are
    20^3 = 8000 unique class triple patterns (vs 13^3=2197 with the
    degenerate POS system). More patterns, better distributed.

    Learns patterns like: "function-word → content-word → function-word"
    (the cat was...) which maps to bucket transitions like 1→6→1.
    """

    def __init__(self, n_hashes=2, table_size=65537, eta=1, clip=50, weight=0.5):
        super().__init__("cls_tri", n_hashes, table_size, eta, clip, weight)

    def get_hash_args_batch(self, context, candidates, word_class):
        if len(context) < 2:
            return None
        K = len(candidates)
        prev2_class = np.full(K, int(word_class[context[-2]]), dtype=np.int64)
        prev1_class = np.full(K, int(word_class[context[-1]]), dtype=np.int64)
        cand_class = word_class[candidates].astype(np.int64)
        return (prev2_class, prev1_class, cand_class)

    def get_hash_args_nce(self, prev_words, prev_class, right_words, right_class,
                          prev2_words=None, prev2_class=None, mask=None):
        if prev2_class is None:
            return None
        if mask is None:
            return (prev2_class, prev_class, right_class)
        if not np.any(mask):
            return None
        return (prev2_class[mask], prev_class[mask], right_class[mask])


# ---------------------------------------------------------------------------
# Default feature set factory
# ---------------------------------------------------------------------------

def default_features(
    vocab_size: int = 2000,
    n_classes: int = 20,
    lex_table_size: int = 65537,
    class_table_size: int = 65537,
    tri_table_size: int = 65537,
    class_tri_table_size: int = 65537,
) -> List[FeatureSpec]:
    """
    Create the recommended default feature set.

    These 6 features use DATA-DRIVEN classes instead of static POS:
      - LexBigramFeature: main workhorse (pure lexical pairs)
      - WordClassBigramFeature: word→class (replaces WordPosBigramFeature!)
      - ClassWordBigramFeature: class→word (replaces PosWordBigramFeature!)
      - LexSkipFeature: skip-gram lexical (long-range dependencies)
      - ClassTrigramFeature: class trigram (replaces PosTrigramFeature!)
      - LexTrigramFeature: lexical trigram patterns (collocations)

    With K=20 frequency buckets, class features have 20*V = 40000 keys
    instead of the degenerate 1*V ≈ 2000 keys from the old POS system.

    Total memory: ~3.5 MB for V=2000, K=20.
    """
    return [
        LexBigramFeature(
            n_hashes=3, table_size=lex_table_size,
            eta=1, clip=100, weight=1.0,
        ),
        WordClassBigramFeature(
            n_hashes=2, table_size=class_table_size,
            eta=1, clip=50, weight=0.5,
        ),
        ClassWordBigramFeature(
            n_hashes=2, table_size=class_table_size,
            eta=1, clip=50, weight=0.5,
        ),
        LexSkipFeature(
            n_hashes=2, table_size=lex_table_size,
            eta=1, clip=80, weight=0.3,
        ),
        ClassTrigramFeature(
            n_hashes=2, table_size=class_tri_table_size,
            eta=1, clip=50, weight=0.5,
        ),
        LexTrigramFeature(
            n_hashes=2, table_size=tri_table_size,
            eta=1, clip=80, weight=0.3,
        ),
    ]


# ===========================================================================
# FeatureHashEnergyTable — dynamic feature registry
# ===========================================================================

class FeatureHashEnergyTable:
    """
    Feature-Hashed Integer Energy Table with DATA-DRIVEN word classes.

    v81 BREAKING CHANGE: Uses `word_class` (frequency buckets) instead of
    `word_pos` (static POS tags). The class system is:
    - DATA-DRIVEN: frequency buckets computed from corpus statistics
    - VARIABLE: K classes (default 20), not hardcoded at 13
    - BALANCED: ~V/K words per class, not 88% in one class
    - COMPOSABLE: any word→class mapping can be plugged in

    Usage:
        energy = FeatureHashEnergyTable(vocab_size=2000, word_class=word_bucket)
        for feat in default_features():
            energy.add_feature(feat)
        energy.train_nce(sequences)
        E = energy.compute_local_energy_batch(context, candidates)
    """

    def __init__(
        self,
        vocab_size: int,
        word_class: np.ndarray,
        n_classes: int = 20,
        seed: int = 42,
    ):
        self.V = vocab_size
        self.word_class = word_class.astype(np.int32)
        self.n_classes = n_classes
        self.seed = seed
        self.features: OrderedDict[str, FeatureSpec] = OrderedDict()

        # Build class-indexed word lists for BALANCED negative sampling
        # This replaces the old POS-balanced sampling with class-balanced sampling
        self._class_word_indices: Dict[int, np.ndarray] = {}
        for cls in range(n_classes + 1):  # +1 because bucket 0 = special tokens
            indices = np.where(self.word_class == cls)[0]
            if len(indices) > 0:
                self._class_word_indices[cls] = indices.astype(np.int64)

    def add_feature(self, feature: FeatureSpec):
        """Register a feature. Can be called any time before training."""
        self.features[feature.name] = feature

    def remove_feature(self, name: str):
        """Remove a feature by name."""
        if name in self.features:
            del self.features[name]

    def get_feature(self, name: str) -> Optional[FeatureSpec]:
        """Get a feature by name."""
        return self.features.get(name)

    # -------------------------------------------------------------------
    # Energy computation — the hot path
    # -------------------------------------------------------------------

    def compute_local_energy_batch(
        self,
        context_word_ids: List[int],
        candidates: np.ndarray,
    ) -> np.ndarray:
        """
        Compute total local energy for all candidates given context.

        E = sum_features(weight_f * E_f)

        Returns integer energy array. Lower = more likely = better.
        """
        K = len(candidates)
        if not context_word_ids:
            return np.zeros(K, dtype=np.int64)

        total = np.zeros(K, dtype=np.float64)
        for feat in self.features.values():
            e = feat.energy_batch(context_word_ids, candidates, self.word_class)
            total += feat.weight * e.astype(np.float64)

        return total.astype(np.int64)

    def compute_local_energy(
        self,
        context_word_ids: List[int],
        candidate: int,
    ) -> int:
        """Scalar version for single candidate."""
        if not context_word_ids:
            return 0

        total = 0.0
        for feat in self.features.values():
            total += feat.weight * feat.energy_scalar(
                context_word_ids, candidate, self.word_class
            )
        return int(total)

    # -------------------------------------------------------------------
    # NCE Training — vectorized batch integer updates
    # -------------------------------------------------------------------

    def _sample_balanced_negatives(
        self, rng: np.random.RandomState, size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample negative words with BALANCED class distribution.

        Instead of random words (which would be dominated by high-freq buckets),
        we pick a random class uniformly, then a random word from that class.

        This ensures ALL class transitions get negative training signal,
        not just the dominant class. This is the v81 equivalent of the old
        POS-balanced sampling, but with balanced classes instead of degenerate ones.

        Returns (neg_word_ids, neg_class_ids) as int64 arrays.
        """
        # All class IDs that have words (skip empty classes)
        valid_classes = list(self._class_word_indices.keys())
        if not valid_classes:
            # Fallback: random words
            neg_words = rng.randint(4, self.V, size=size)
            neg_class = self.word_class[neg_words].astype(np.int64)
            return neg_words, neg_class

        neg_class = np.array(
            [rng.choice(valid_classes) for _ in range(size)],
            dtype=np.int64,
        )
        neg_words = np.empty(size, dtype=np.int64)

        for cls, indices in self._class_word_indices.items():
            mask = neg_class == cls
            n = int(mask.sum())
            if n > 0:
                neg_words[mask] = rng.choice(indices, size=n)

        return neg_words, neg_class

    def train_nce(
        self,
        sequences: List[List[int]],
        n_epochs: int = 3,
        n_negatives: int = 3,
        callback=None,
    ) -> Dict:
        """
        Train all feature tables via vectorized integer NCE.

        For each (prev, target) pair:
          Positive: feature.tables[h][hash(real_pair)] -= eta
          Negative: feature.tables[h][hash(fake_pair)] += eta

        KEY v81: Negative samples are CLASS-balanced (not POS-balanced).
        With K=20 balanced classes, every class gets proper negative signal.
        """
        import time as _time

        if not self.features:
            print("    WARNING: No features registered — nothing to train!", flush=True)
            return {'epochs': []}

        rng = np.random.RandomState(self.seed)

        # Precompute all pairs
        all_prev, all_target, all_prev2 = [], [], []
        for seq in sequences:
            if len(seq) < 2:
                continue
            for pos in range(1, len(seq)):
                all_prev.append(seq[pos - 1])
                all_target.append(seq[pos])
                all_prev2.append(seq[pos - 2] if pos >= 2 else 0)

        all_prev = np.array(all_prev, dtype=np.int64)
        all_target = np.array(all_target, dtype=np.int64)
        all_prev2 = np.array(all_prev2, dtype=np.int64)
        N = len(all_prev)

        # Class arrays (v81: word_class instead of word_pos)
        all_prev_class = self.word_class[all_prev].astype(np.int64)
        all_target_class = self.word_class[all_target].astype(np.int64)
        all_prev2_class = self.word_class[all_prev2].astype(np.int64)
        has_prev2 = all_prev2 > 0

        feat_names = [f.name for f in self.features.values()]
        print(f"    {N:,} training pairs", flush=True)
        print(f"    Features: {', '.join(feat_names)}", flush=True)
        print(f"    Class-balanced negatives: {len(self._class_word_indices)} classes", flush=True)

        all_stats = []

        for epoch in range(n_epochs):
            t0 = _time.time()

            order = rng.permutation(N)
            sp = all_prev[order]
            st = all_target[order]
            sp2 = all_prev2[order]
            sp_cls = all_prev_class[order]
            st_cls = all_target_class[order]
            sp2_cls = all_prev2_class[order]
            hp2 = has_prev2[order]

            chunk = 100000
            for c0 in range(0, N, chunk):
                c1 = min(c0 + chunk, N)
                cp = sp[c0:c1]; ct = st[c0:c1]; cp2 = sp2[c0:c1]
                cpc = sp_cls[c0:c1]; ctc = st_cls[c0:c1]; cp2c = sp2_cls[c0:c1]
                chp2 = hp2[c0:c1]
                C = len(cp)

                # Positive updates — all features
                for feat in self.features.values():
                    feat.nce_positive(
                        cp, cpc, ct, ctc,
                        prev2_words=cp2, prev2_class=cp2c, mask=chp2,
                    )

                # Negative updates — class-balanced negatives
                for _ in range(n_negatives):
                    neg, neg_cls = self._sample_balanced_negatives(rng, C)

                    for feat in self.features.values():
                        feat.nce_negative(
                            cp, cpc, neg, neg_cls,
                            prev2_words=cp2, prev2_class=cp2c, mask=chp2,
                        )

            t_elapsed = _time.time() - t0

            # Clip all features
            for feat in self.features.values():
                feat.clip_tables()

            # Discriminative accuracy per feature + combined
            n_check = min(2000, N)
            ci = rng.choice(N, n_check, replace=False)
            cp_chk = all_prev[ci]; ct_chk = all_target[ci]
            cpc_chk = all_prev_class[ci]; ctc_chk = all_target_class[ci]
            cp2_chk = all_prev2[ci]; cp2c_chk = all_prev2_class[ci]
            hp2_chk = has_prev2[ci]

            neg_chk, neg_cls_chk = self._sample_balanced_negatives(rng, n_check)

            # Per-feature discrimination
            feat_discs = {}
            for feat in self.features.values():
                pos_args = feat.get_hash_args_nce(
                    cp_chk, cpc_chk, ct_chk, ctc_chk,
                    cp2_chk, cp2c_chk, hp2_chk,
                )
                neg_args = feat.get_hash_args_nce(
                    cp_chk, cpc_chk, neg_chk, neg_cls_chk,
                    cp2_chk, cp2c_chk, hp2_chk,
                )
                if pos_args is not None and neg_args is not None:
                    n_eval = len(pos_args[0])
                    pe = np.zeros(n_eval, dtype=np.int64)
                    ne = np.zeros(n_eval, dtype=np.int64)
                    for h in range(feat.n_hashes):
                        if len(pos_args) == 2:
                            pe += feat.tables[h][_hash2_vec(pos_args[0], pos_args[1], h, feat.table_size)]
                            ne += feat.tables[h][_hash2_vec(neg_args[0], neg_args[1], h, feat.table_size)]
                        else:
                            pe += feat.tables[h][_hash3_vec(pos_args[0], pos_args[1], pos_args[2], h, feat.table_size)]
                            ne += feat.tables[h][_hash3_vec(neg_args[0], neg_args[1], neg_args[2], h, feat.table_size)]
                    feat_discs[feat.name] = float(np.sum(pe < ne)) / max(1, n_eval)
                else:
                    feat_discs[feat.name] = 0.5

            # Combined discrimination
            comb_real = np.zeros(n_check, dtype=np.float64)
            comb_neg = np.zeros(n_check, dtype=np.float64)
            for feat in self.features.values():
                pos_args = feat.get_hash_args_nce(
                    cp_chk, cpc_chk, ct_chk, ctc_chk,
                    cp2_chk, cp2c_chk, None,
                )
                neg_args = feat.get_hash_args_nce(
                    cp_chk, cpc_chk, neg_chk, neg_cls_chk,
                    cp2_chk, cp2c_chk, None,
                )
                if pos_args is not None and neg_args is not None and len(pos_args[0]) == n_check:
                    pe = np.zeros(n_check, dtype=np.int64)
                    ne = np.zeros(n_check, dtype=np.int64)
                    for h in range(feat.n_hashes):
                        if len(pos_args) == 2:
                            pe += feat.tables[h][_hash2_vec(pos_args[0], pos_args[1], h, feat.table_size)]
                            ne += feat.tables[h][_hash2_vec(neg_args[0], neg_args[1], h, feat.table_size)]
                        else:
                            pe += feat.tables[h][_hash3_vec(pos_args[0], pos_args[1], pos_args[2], h, feat.table_size)]
                            ne += feat.tables[h][_hash3_vec(neg_args[0], neg_args[1], neg_args[2], h, feat.table_size)]
                    comb_real += feat.weight * pe.astype(np.float64)
                    comb_neg += feat.weight * ne.astype(np.float64)

            comb_d = float(np.sum(comb_real < comb_neg)) / max(1, n_check)

            stats = {
                'epoch': epoch,
                'combined_disc': comb_d,
                'feature_disc': feat_discs,
                'time_s': t_elapsed,
            }
            all_stats.append(stats)

            disc_str = ", ".join(f"{k}={v:.3f}" for k, v in feat_discs.items())
            print(f"    Epoch {epoch+1}/{n_epochs}: "
                  f"combined={comb_d:.3f} | {disc_str} | "
                  f"time={t_elapsed:.1f}s", flush=True)

            # Show per-feature table stats
            for feat in self.features.values():
                fs = feat.statistics()
                print(f"      {feat.name}: range=[{fs['range'][0]},{fs['range'][1]}], "
                      f"mean={fs['mean']:.1f}, std={fs['std']:.1f}, "
                      f"nnz={fs['nnz']}", flush=True)

            if callback:
                callback(epoch, stats)

        return {'epochs': all_stats}

    # -------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------

    def get_class_matrix(self) -> np.ndarray:
        """
        Return K×K class transition energy matrix for visualization.

        This is computed from the ClassWordBigramFeature by marginalizing
        over prev_class words. If not present, returns zeros.
        """
        K = self.n_classes + 1  # +1 for bucket 0 (special tokens)
        matrix = np.zeros((K, K), dtype=np.float64)

        # Try ClassWordBigramFeature
        cls_word_feat = self.features.get("cls_word_bi")
        if cls_word_feat is not None:
            for c1 in range(K):
                for c2 in range(K):
                    total = sum(
                        int(cls_word_feat.tables[h][_hash2(c1, c2, h, cls_word_feat.table_size)])
                        for h in range(cls_word_feat.n_hashes)
                    )
                    matrix[c1, c2] = total / max(1, cls_word_feat.n_hashes)
            return matrix

        # Try ClassTrigramFeature — marginalize to get bigram
        cls_tri_feat = self.features.get("cls_tri")
        if cls_tri_feat is not None:
            for c1 in range(K):
                for c2 in range(K):
                    # Average over all prev2 classes
                    total = 0.0
                    for c0 in range(K):
                        t = sum(
                            int(cls_tri_feat.tables[h][_hash3(c0, c1, c2, h, cls_tri_feat.table_size)])
                            for h in range(cls_tri_feat.n_hashes)
                        )
                        total += t
                    matrix[c1, c2] = total / max(1, K * cls_tri_feat.n_hashes)
            return matrix

        return matrix

    def memory_mb(self) -> float:
        """Estimate total memory usage in MB."""
        total = sum(
            sum(t.nbytes for t in feat.tables)
            for feat in self.features.values()
        )
        return total / (1024 * 1024)

    def statistics(self) -> Dict:
        """Return combined table statistics."""
        feat_stats = {feat.name: feat.statistics() for feat in self.features.values()}
        return {
            'n_features': len(self.features),
            'feature_names': list(self.features.keys()),
            'features': feat_stats,
            'memory_mb': self.memory_mb(),
            'n_classes': self.n_classes,
        }
