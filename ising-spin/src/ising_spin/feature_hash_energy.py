"""
Dynamic Feature-Hashed Integer Energy Table with Variable Features.

ARCHITECTURE CHANGE (v80):
  The old system hardcoded 4 table types (pos_bi, lex_bi, skip_lex, skip_pos_bi)
  with a static 13x13 POS matrix. Only 169 unique POS pair keys → instant
  saturation at ±500, no gradient signal, POS disc_acc below random.

  The new system uses a FeatureSpec base class with dynamic registration.
  Any number of features can be added via add_feature(). Each feature is
  self-contained: its own hash tables, eta, clip, weight.

KEY INSIGHT — Why the old POS table was broken:
  With 13 POS types, hash(prev_pos, cand_pos) produces only 13x13=169 unique
  keys. Each key independently accumulates NCE updates and saturates at the
  clip limit. Once saturated, there's ZERO gradient — the table can't learn.

  The fix: mixed word-POS features like hash(prev_word, cand_pos) produce
  Vx13 = 26000+ unique keys. Hash collisions in a 65537-slot table create
  SMOOTH generalization: "the"->NOUN and "a"->NOUN hash to nearby slots,
  learning that DET->NOUN is good. But "the"->VERB hashes differently,
  allowing fine-grained distinctions.

AVAILABLE FEATURES:
  Bigram (2-token context):
    LexBigramFeature    — hash(prev_word, cand_word)   token-specific pairs
    WordPosBigramFeature — hash(prev_word, cand_pos)   word→POS transitions
    PosWordBigramFeature — hash(prev_pos, cand_word)   POS→word transitions
    PosBigramFeature    — hash(prev_pos, cand_pos)     POS pair (optional)

  Skip-gram (skip-1 context):
    LexSkipFeature      — hash(prev2_word, cand_word)  long-range lexical
    WordPosSkipFeature  — hash(prev2_word, cand_pos)   word→POS at distance 2
    PosWordSkipFeature  — hash(prev2_pos, cand_word)   POS→word at distance 2
    PosSkipFeature      — hash(prev2_pos, cand_pos)    POS skip (optional)

  Trigram (3-token context):
    LexTrigramFeature   — hash(prev2_word, prev_word, cand_word) lexical 3-gram
    PosTrigramFeature   — hash(prev2_pos, prev_pos, cand_pos)    POS 3-gram

DEFAULT FEATURE SET (recommended):
  LexBigramFeature, WordPosBigramFeature, PosWordBigramFeature,
  LexSkipFeature, PosTrigramFeature, LexTrigramFeature

  Note: PosBigramFeature is EXCLUDED by default — the 169-key static
  POS matrix is the disaster we're fixing. WordPosBigramFeature and
  PosWordBigramFeature provide the same POS generalization but with
  Vx13 richness instead of 13x13 poverty.

ADD YOUR OWN FEATURE:
  1. Subclass FeatureSpec
  2. Implement get_hash_args_batch() and get_hash_args_nce()
  3. Call energy_table.add_feature(your_feature)

All tables trained via integer NCE with balanced POS negatives.
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


def _next_prime(n: int) -> int:
    """Find the next prime >= n."""
    if n <= 2:
        return 2
    if n % 2 == 0:
        n += 1
    while True:
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                break
        else:
            return n
        n += 2


# ---------------------------------------------------------------------------
# FeatureSpec — base class for energy features
# ---------------------------------------------------------------------------

class FeatureSpec:
    """
    Base class for energy features. Each feature is self-contained:
    its own hash tables, learning rate (eta), clipping range, and weight.

    To add a new feature:
      1. Subclass FeatureSpec
      2. Implement get_hash_args_batch() and get_hash_args_nce()
      3. Call energy_table.add_feature(your_feature)

    The feature's contribution to the total energy is:
        E_feature = weight * sum_h(table[h][hash(inputs, h)])
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
        word_pos: np.ndarray,
    ) -> Optional[Tuple]:
        """
        Extract hash input arrays from context + candidates (generation-time).

        Returns: tuple of (a, b) or (a, b, c) as np.int64 arrays, shape (K,)
                 Or None if this feature doesn't apply (e.g., context too short).
        """
        raise NotImplementedError(
            f"{self.name}: get_hash_args_batch() not implemented"
        )

    def get_hash_args_nce(
        self,
        prev_words: np.ndarray,
        prev_pos: np.ndarray,
        right_words: np.ndarray,
        right_pos: np.ndarray,
        prev2_words: Optional[np.ndarray] = None,
        prev2_pos: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> Optional[Tuple]:
        """
        Extract hash input arrays from NCE training batch.

        Args:
            prev_words:  shape (N,) — previous word IDs
            prev_pos:    shape (N,) — previous POS IDs
            right_words: shape (N,) — target word IDs (positive) or negative word IDs
            right_pos:   shape (N,) — target/negative POS IDs
            prev2_words: shape (N,) — word at t-2 (0 if none), or None
            prev2_pos:   shape (N,) — POS at t-2 (0 if none), or None
            mask:        shape (N,) — bool, True where prev2 is valid

        Returns: tuple of arrays for hashing, or None if inapplicable.
                 For features using prev2, apply mask and return masked arrays.
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
        word_pos: np.ndarray,
    ) -> np.ndarray:
        """Compute energy contribution for all candidates."""
        args = self.get_hash_args_batch(context, candidates, word_pos)
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
        word_pos: np.ndarray,
    ) -> int:
        """Compute energy for a single candidate."""
        args = self.get_hash_args_batch(context, np.array([candidate], dtype=np.int64), word_pos)
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
        prev_pos: np.ndarray,
        targets: np.ndarray,
        target_pos: np.ndarray,
        prev2_words: Optional[np.ndarray] = None,
        prev2_pos: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ):
        """Apply positive NCE update (real pairs): table[hash] -= eta."""
        args = self.get_hash_args_nce(
            prev_words, prev_pos, targets, target_pos,
            prev2_words, prev2_pos, mask,
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
        prev_pos: np.ndarray,
        neg_words: np.ndarray,
        neg_pos: np.ndarray,
        prev2_words: Optional[np.ndarray] = None,
        prev2_pos: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ):
        """Apply negative NCE update (corrupt pairs): table[hash] += eta."""
        args = self.get_hash_args_nce(
            prev_words, prev_pos, neg_words, neg_pos,
            prev2_words, prev2_pos, mask,
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
# Bigram features (need context >= 1)
# ---------------------------------------------------------------------------

class LexBigramFeature(FeatureSpec):
    """
    hash(prev_word, cand_word) — lexical bigram transitions.

    The main workhorse: token-specific pairs like ("the", "cat"), ("a", "dog").
    With V=2000 and table_size=65537, about 6% collision rate — enough for
    smooth generalization without losing too much specificity.
    """

    def __init__(self, n_hashes=3, table_size=65537, eta=1, clip=100, weight=1.0):
        super().__init__("lex_bi", n_hashes, table_size, eta, clip, weight)

    def get_hash_args_batch(self, context, candidates, word_pos):
        if not context:
            return None
        K = len(candidates)
        prev = np.full(K, context[-1], dtype=np.int64)
        return (prev, candidates.astype(np.int64))

    def get_hash_args_nce(self, prev_words, prev_pos, right_words, right_pos,
                          prev2_words=None, prev2_pos=None, mask=None):
        return (prev_words, right_words)


class WordPosBigramFeature(FeatureSpec):
    """
    hash(prev_word, cand_pos) — word→POS transitions.

    REPLACES the old static POS bigram table. Instead of hash(pos, pos)
    with 169 keys, this uses hash(word, pos) with V*13 = 26000+ keys.

    This is the key fix for the "POS disaster": "the"->NOUN and "a"->NOUN
    hash to DIFFERENT slots, but collisions in a 65537-slot table create
    natural generalization. The model learns that after determiners, nouns
    are likely — but it also learns which specific determiners prefer which
    specific noun types.
    """

    def __init__(self, n_hashes=2, table_size=65537, eta=1, clip=50, weight=0.5):
        super().__init__("word_pos_bi", n_hashes, table_size, eta, clip, weight)

    def get_hash_args_batch(self, context, candidates, word_pos):
        if not context:
            return None
        K = len(candidates)
        prev = np.full(K, context[-1], dtype=np.int64)
        cand_pos = word_pos[candidates].astype(np.int64)
        return (prev, cand_pos)

    def get_hash_args_nce(self, prev_words, prev_pos, right_words, right_pos,
                          prev2_words=None, prev2_pos=None, mask=None):
        return (prev_words, right_pos)


class PosWordBigramFeature(FeatureSpec):
    """
    hash(prev_pos, cand_word) — POS→word transitions.

    Complementary to WordPosBigramFeature: after POS type X, which specific
    word is likely? Learns patterns like "after DET, 'the' is the most
    common word" while also capturing that "after AUX, 'was' is likely."
    """

    def __init__(self, n_hashes=2, table_size=65537, eta=1, clip=50, weight=0.5):
        super().__init__("pos_word_bi", n_hashes, table_size, eta, clip, weight)

    def get_hash_args_batch(self, context, candidates, word_pos):
        if not context:
            return None
        K = len(candidates)
        prev_pos = np.full(K, int(word_pos[context[-1]]), dtype=np.int64)
        return (prev_pos, candidates.astype(np.int64))

    def get_hash_args_nce(self, prev_words, prev_pos, right_words, right_pos,
                          prev2_words=None, prev2_pos=None, mask=None):
        return (prev_pos, right_words)


class PosBigramFeature(FeatureSpec):
    """
    hash(prev_pos, cand_pos) — pure POS pair transitions.

    WARNING: This is the OLD static POS feature. With 13x13 = 169 unique
    keys, it saturates quickly and provides poor gradient signal. Included
    for backward compatibility and ablation studies, but NOT in the default
    feature set.

    If used, set a SMALL table_size (e.g. 1009) and SMALL clip (e.g. 30)
    to prevent saturation.
    """

    def __init__(self, n_hashes=2, table_size=1009, eta=1, clip=30, weight=0.3):
        super().__init__("pos_bi", n_hashes, table_size, eta, clip, weight)

    def get_hash_args_batch(self, context, candidates, word_pos):
        if not context:
            return None
        K = len(candidates)
        prev_pos = np.full(K, int(word_pos[context[-1]]), dtype=np.int64)
        cand_pos = word_pos[candidates].astype(np.int64)
        return (prev_pos, cand_pos)

    def get_hash_args_nce(self, prev_words, prev_pos, right_words, right_pos,
                          prev2_words=None, prev2_pos=None, mask=None):
        return (prev_pos, right_pos)


# ---------------------------------------------------------------------------
# Skip-gram features (need context >= 2)
# ---------------------------------------------------------------------------

class LexSkipFeature(FeatureSpec):
    """
    hash(prev2_word, cand_word) — skip-gram lexical.

    Captures patterns where a word at distance 2 predicts the current word.
    Example: "the cat sat" → hash("the", "sat") captures DET→VERB at distance 2.
    """

    def __init__(self, n_hashes=2, table_size=65537, eta=1, clip=80, weight=0.3):
        super().__init__("lex_skip", n_hashes, table_size, eta, clip, weight)

    def get_hash_args_batch(self, context, candidates, word_pos):
        if len(context) < 2:
            return None
        K = len(candidates)
        prev2 = np.full(K, context[-2], dtype=np.int64)
        return (prev2, candidates.astype(np.int64))

    def get_hash_args_nce(self, prev_words, prev_pos, right_words, right_pos,
                          prev2_words=None, prev2_pos=None, mask=None):
        if prev2_words is None:
            return None
        if mask is None:
            return (prev2_words, right_words)
        if not np.any(mask):
            return None
        return (prev2_words[mask], right_words[mask])


class WordPosSkipFeature(FeatureSpec):
    """
    hash(prev2_word, cand_pos) — word→POS at distance 2.

    Like WordPosBigramFeature but at skip distance. Learns that after
    "the ... ", a VERB is unlikely (you'd expect NOUN after "the X ...").
    """

    def __init__(self, n_hashes=2, table_size=65537, eta=1, clip=50, weight=0.3):
        super().__init__("word_pos_skip", n_hashes, table_size, eta, clip, weight)

    def get_hash_args_batch(self, context, candidates, word_pos):
        if len(context) < 2:
            return None
        K = len(candidates)
        prev2 = np.full(K, context[-2], dtype=np.int64)
        cand_pos = word_pos[candidates].astype(np.int64)
        return (prev2, cand_pos)

    def get_hash_args_nce(self, prev_words, prev_pos, right_words, right_pos,
                          prev2_words=None, prev2_pos=None, mask=None):
        if prev2_words is None:
            return None
        if mask is None:
            return (prev2_words, right_pos)
        if not np.any(mask):
            return None
        return (prev2_words[mask], right_pos[mask])


class PosWordSkipFeature(FeatureSpec):
    """
    hash(prev2_pos, cand_word) — POS→word at distance 2.

    Like PosWordBigramFeature but at skip distance. Learns that after
    "DET ... ", specific nouns are likely to follow.
    """

    def __init__(self, n_hashes=2, table_size=65537, eta=1, clip=50, weight=0.3):
        super().__init__("pos_word_skip", n_hashes, table_size, eta, clip, weight)

    def get_hash_args_batch(self, context, candidates, word_pos):
        if len(context) < 2:
            return None
        K = len(candidates)
        prev2_pos = np.full(K, int(word_pos[context[-2]]), dtype=np.int64)
        return (prev2_pos, candidates.astype(np.int64))

    def get_hash_args_nce(self, prev_words, prev_pos, right_words, right_pos,
                          prev2_words=None, prev2_pos=None, mask=None):
        if prev2_pos is None:
            return None
        if mask is None:
            return (prev2_pos, right_words)
        if not np.any(mask):
            return None
        return (prev2_pos[mask], right_words[mask])


class PosSkipFeature(FeatureSpec):
    """
    hash(prev2_pos, cand_pos) — POS skip-gram.

    WARNING: Like PosBigramFeature, this has only 13x13 = 169 unique keys.
    Use with small clip to prevent saturation.
    """

    def __init__(self, n_hashes=2, table_size=1009, eta=1, clip=30, weight=0.2):
        super().__init__("pos_skip", n_hashes, table_size, eta, clip, weight)

    def get_hash_args_batch(self, context, candidates, word_pos):
        if len(context) < 2:
            return None
        K = len(candidates)
        prev2_pos = np.full(K, int(word_pos[context[-2]]), dtype=np.int64)
        cand_pos = word_pos[candidates].astype(np.int64)
        return (prev2_pos, cand_pos)

    def get_hash_args_nce(self, prev_words, prev_pos, right_words, right_pos,
                          prev2_words=None, prev2_pos=None, mask=None):
        if prev2_pos is None:
            return None
        if mask is None:
            return (prev2_pos, right_pos)
        if not np.any(mask):
            return None
        return (prev2_pos[mask], right_pos[mask])


# ---------------------------------------------------------------------------
# Trigram features (need context >= 2)
# ---------------------------------------------------------------------------

class LexTrigramFeature(FeatureSpec):
    """
    hash(prev2_word, prev_word, cand_word) — lexical trigram.

    Three-word token patterns: "once upon a", "there was a", etc.
    Much more specific than bigrams — captures local collocations.
    """

    def __init__(self, n_hashes=2, table_size=65537, eta=1, clip=80, weight=0.3):
        super().__init__("lex_tri", n_hashes, table_size, eta, clip, weight)

    def get_hash_args_batch(self, context, candidates, word_pos):
        if len(context) < 2:
            return None
        K = len(candidates)
        prev2 = np.full(K, context[-2], dtype=np.int64)
        prev1 = np.full(K, context[-1], dtype=np.int64)
        return (prev2, prev1, candidates.astype(np.int64))

    def get_hash_args_nce(self, prev_words, prev_pos, right_words, right_pos,
                          prev2_words=None, prev2_pos=None, mask=None):
        if prev2_words is None:
            return None
        if mask is None:
            return (prev2_words, prev_words, right_words)
        if not np.any(mask):
            return None
        return (prev2_words[mask], prev_words[mask], right_words[mask])


class PosTrigramFeature(FeatureSpec):
    """
    hash(prev2_pos, prev_pos, cand_pos) — POS trigram.

    With 13^3 = 2197 unique POS triple patterns, this is much richer than
    the 169-key POS bigram. Learns patterns like DET NOUN VERB, PRON AUX VERB,
    etc. With table_size=1301 and 2 hashes, there are meaningful collisions
    that create smooth generalization.
    """

    def __init__(self, n_hashes=2, table_size=1301, eta=1, clip=50, weight=0.5):
        super().__init__("pos_tri", n_hashes, table_size, eta, clip, weight)

    def get_hash_args_batch(self, context, candidates, word_pos):
        if len(context) < 2:
            return None
        K = len(candidates)
        prev2_pos = np.full(K, int(word_pos[context[-2]]), dtype=np.int64)
        prev1_pos = np.full(K, int(word_pos[context[-1]]), dtype=np.int64)
        cand_pos = word_pos[candidates].astype(np.int64)
        return (prev2_pos, prev1_pos, cand_pos)

    def get_hash_args_nce(self, prev_words, prev_pos, right_words, right_pos,
                          prev2_words=None, prev2_pos=None, mask=None):
        if prev2_pos is None:
            return None
        if mask is None:
            return (prev2_pos, prev_pos, right_pos)
        if not np.any(mask):
            return None
        return (prev2_pos[mask], prev_pos[mask], right_pos[mask])


# ---------------------------------------------------------------------------
# Default feature set factory
# ---------------------------------------------------------------------------

def default_features(
    vocab_size: int = 2000,
    n_pos_types: int = 13,
    lex_table_size: int = 65537,
    pos_table_size: int = 65537,
    tri_table_size: int = 65537,
    pos_tri_table_size: int = 1301,
) -> List[FeatureSpec]:
    """
    Create the recommended default feature set.

    These 6 features replace the old hardcoded 4-table system:
      - LexBigramFeature: main workhorse (token-specific pairs)
      - WordPosBigramFeature: word→POS (replaces static POS bigram!)
      - PosWordBigramFeature: POS→word (complementary direction)
      - LexSkipFeature: skip-gram lexical (long-range dependencies)
      - PosTrigramFeature: POS trigram patterns (2197 unique keys)
      - LexTrigramFeature: lexical trigram patterns (collocations)

    With default clip values (50-100) and weights (0.3-1.0), the maximum
    combined energy is ~546 — manageable with z-score normalization and
    alpha in [0.01, 0.5].

    Total memory: ~3.5 MB for V=2000.
    """
    return [
        LexBigramFeature(
            n_hashes=3, table_size=lex_table_size,
            eta=1, clip=100, weight=1.0,
        ),
        WordPosBigramFeature(
            n_hashes=2, table_size=pos_table_size,
            eta=1, clip=50, weight=0.5,
        ),
        PosWordBigramFeature(
            n_hashes=2, table_size=pos_table_size,
            eta=1, clip=50, weight=0.5,
        ),
        LexSkipFeature(
            n_hashes=2, table_size=lex_table_size,
            eta=1, clip=80, weight=0.3,
        ),
        PosTrigramFeature(
            n_hashes=2, table_size=pos_tri_table_size,
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
    Feature-Hashed Integer Energy Table with dynamic feature registration.

    Instead of hardcoded table types, features are registered via add_feature().
    Each feature is a self-contained FeatureSpec with its own tables, eta,
    clip, and weight. The energy table simply sums their weighted contributions.

    Usage:
        energy = FeatureHashEnergyTable(vocab_size=2000, word_pos=word_pos)
        for feat in default_features():
            energy.add_feature(feat)
        energy.train_nce(sequences)
        E = energy.compute_local_energy_batch(context, candidates)
    """

    def __init__(
        self,
        vocab_size: int,
        word_pos: np.ndarray,
        n_pos_types: int = 13,
        seed: int = 42,
    ):
        self.V = vocab_size
        self.word_pos = word_pos.astype(np.int32)
        self.n_pos_types = n_pos_types
        self.seed = seed
        self.features: OrderedDict[str, FeatureSpec] = OrderedDict()

        # Build POS-indexed word lists for BALANCED negative sampling
        self._pos_word_indices: Dict[int, np.ndarray] = {}
        for pt in range(n_pos_types):
            indices = np.where(self.word_pos == pt)[0]
            if len(indices) > 0:
                self._pos_word_indices[pt] = indices.astype(np.int64)

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
            e = feat.energy_batch(context_word_ids, candidates, self.word_pos)
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
                context_word_ids, candidate, self.word_pos
            )
        return int(total)

    # -------------------------------------------------------------------
    # NCE Training — vectorized batch integer updates
    # -------------------------------------------------------------------

    def _sample_balanced_negatives(
        self, rng: np.random.RandomState, size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample negative words with BALANCED POS distribution.

        Instead of random words (88% NOUN), we pick a random POS type
        uniformly from all types, then a random word with that POS.

        Returns (neg_word_ids, neg_pos_types) as int64 arrays.
        """
        neg_pos_types = rng.randint(0, self.n_pos_types, size=size)
        neg_words = np.empty(size, dtype=np.int64)

        for pt, indices in self._pos_word_indices.items():
            mask = neg_pos_types == pt
            n = int(mask.sum())
            if n > 0:
                neg_words[mask] = rng.choice(indices, size=n)

        return neg_words, neg_pos_types.astype(np.int64)

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

        KEY: POS negatives are BALANCED across POS types.
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

        all_prev_pos = self.word_pos[all_prev].astype(np.int64)
        all_target_pos = self.word_pos[all_target].astype(np.int64)
        all_prev2_pos = self.word_pos[all_prev2].astype(np.int64)
        has_prev2 = all_prev2 > 0

        feat_names = [f.name for f in self.features.values()]
        print(f"    {N:,} training pairs", flush=True)
        print(f"    Features: {', '.join(feat_names)}", flush=True)
        print(f"    Balanced POS negatives: {len(self._pos_word_indices)} types", flush=True)

        all_stats = []

        for epoch in range(n_epochs):
            t0 = _time.time()

            order = rng.permutation(N)
            sp = all_prev[order]
            st = all_target[order]
            sp2 = all_prev2[order]
            sp_pos = all_prev_pos[order]
            st_pos = all_target_pos[order]
            sp2_pos = all_prev2_pos[order]
            hp2 = has_prev2[order]

            chunk = 100000
            for c0 in range(0, N, chunk):
                c1 = min(c0 + chunk, N)
                cp = sp[c0:c1]; ct = st[c0:c1]; cp2 = sp2[c0:c1]
                cpp = sp_pos[c0:c1]; ctp = st_pos[c0:c1]; cp2p = sp2_pos[c0:c1]
                chp2 = hp2[c0:c1]
                C = len(cp)

                # Positive updates — all features
                for feat in self.features.values():
                    feat.nce_positive(
                        cp, cpp, ct, ctp,
                        prev2_words=cp2, prev2_pos=cp2p, mask=chp2,
                    )

                # Negative updates — balanced POS negatives
                for _ in range(n_negatives):
                    neg, neg_pos = self._sample_balanced_negatives(rng, C)

                    for feat in self.features.values():
                        feat.nce_negative(
                            cp, cpp, neg, neg_pos,
                            prev2_words=cp2, prev2_pos=cp2p, mask=chp2,
                        )

            t_elapsed = _time.time() - t0

            # Clip all features
            for feat in self.features.values():
                feat.clip_tables()

            # Discriminative accuracy per feature + combined
            n_check = min(2000, N)
            ci = rng.choice(N, n_check, replace=False)
            cp_chk = all_prev[ci]; ct_chk = all_target[ci]
            cpp_chk = all_prev_pos[ci]; ctp_chk = all_target_pos[ci]
            cp2_chk = all_prev2[ci]; cp2p_chk = all_prev2_pos[ci]
            hp2_chk = has_prev2[ci]

            neg_chk, neg_pos_chk = self._sample_balanced_negatives(rng, n_check)

            # Per-feature discrimination
            # NOTE: must pass (prev_words, prev_pos, right_words, right_pos, ...)
            # correctly — NOT mixing up word and POS arrays!
            feat_discs = {}
            for feat in self.features.values():
                # Positive: (prev_word, prev_pos, target_word, target_pos, prev2_word, prev2_pos, mask)
                pos_args = feat.get_hash_args_nce(
                    cp_chk, cpp_chk, ct_chk, ctp_chk,
                    cp2_chk, cp2p_chk, hp2_chk,
                )
                # Negative: same context, but target replaced by negative
                neg_args = feat.get_hash_args_nce(
                    cp_chk, cpp_chk, neg_chk, neg_pos_chk,
                    cp2_chk, cp2p_chk, hp2_chk,
                )
                if pos_args is not None and neg_args is not None:
                    # Skip/trigram features return masked arrays — may be shorter than n_check
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

            # Combined discrimination (only on bigram features for consistent shape)
            comb_real = np.zeros(n_check, dtype=np.float64)
            comb_neg = np.zeros(n_check, dtype=np.float64)
            for feat in self.features.values():
                # For combined metric, we need features that work on ALL n_check entries
                # (bigram features). Skip/trigram features would need separate handling.
                pos_args = feat.get_hash_args_nce(
                    cp_chk, cpp_chk, ct_chk, ctp_chk,
                    cp2_chk, cp2p_chk, None,  # pass mask=None to get unmasked arrays
                )
                neg_args = feat.get_hash_args_nce(
                    cp_chk, cpp_chk, neg_chk, neg_pos_chk,
                    cp2_chk, cp2p_chk, None,
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

    def get_pos_matrix(self) -> np.ndarray:
        """
        Return 13x13 POS transition energy matrix for visualization.

        NOTE: This is computed from the WordPosBigramFeature (if present),
        by marginalizing over prev_word. If not present, falls back to
        PosBigramFeature. If neither exists, returns zeros.
        """
        matrix = np.zeros((self.n_pos_types, self.n_pos_types), dtype=np.float64)

        # Try to use PosBigramFeature directly
        pos_bi = self.features.get("pos_bi")
        if pos_bi is not None:
            for t1 in range(self.n_pos_types):
                for t2 in range(self.n_pos_types):
                    total = sum(
                        int(pos_bi.tables[h][_hash2(t1, t2, h, pos_bi.table_size)])
                        for h in range(pos_bi.n_hashes)
                    )
                    matrix[t1, t2] = total / max(1, pos_bi.n_hashes)
            return matrix

        # Fall back: estimate from WordPosBigramFeature by averaging over words
        word_pos_feat = self.features.get("word_pos_bi")
        if word_pos_feat is not None:
            # For each (pos_prev, pos_cand) pair, average the energy over
            # all words with pos_prev
            from .vocabulary import IDX2POS
            for p1 in range(self.n_pos_types):
                words_with_p1 = np.where(self.word_pos == p1)[0]
                if len(words_with_p1) == 0:
                    continue
                for p2 in range(self.n_pos_types):
                    energies = []
                    # Sample up to 20 words with this POS
                    sample_words = words_with_p1[:20]
                    for w in sample_words:
                        e = 0
                        for h in range(word_pos_feat.n_hashes):
                            slot = _hash2(int(w), p2, h, word_pos_feat.table_size)
                            e += int(word_pos_feat.tables[h][slot])
                        energies.append(e)
                    matrix[p1, p2] = np.mean(energies)
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
        }
