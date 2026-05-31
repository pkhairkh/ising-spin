"""
Dynamic Feature-Hashed Integer Energy Table — v82 MULTI-CLASS ARCHITECTURE.

ARCHITECTURE CHANGE (v82):
  v81 replaced static POS with frequency buckets. PPL went from 12M to 13.89
  (base 27.74) — a huge win. BUT the class transition matrix was NEARLY
  UNIFORM because frequency buckets group words by HOW OFTEN they appear,
  not HOW THEY BEHAVE syntactically.

  v82 introduces a MULTI-CLASS system:
  - MULTIPLE word class arrays run SIMULTANEOUSLY
  - Frequency buckets ("freq"): K=20, captures importance/gradient
  - Distributional clusters ("dist"): K=30, captures syntactic role
  - Features declare WHICH class system they use via `class_key`
  - New features are added for distributional clusters

  WHY THIS MATTERS:
  The v81 class transition matrix showed all rows identical — frequency
  buckets can't distinguish "the" from "was" (both high-freq). But
  distributional clusters CAN: "the" clusters with "a" (similar followers),
  "was" clusters with "is" (similar followers). This gives NON-UNIFORM
  transition matrices with real syntactic patterns.

FEATURE REGISTRY:
  Features are self-contained objects. Add/remove at will:
    energy.add_feature(MyFeature(class_key="dist"))
    energy.remove_feature("old_feature")

  Each feature specifies its class_key to select which class array to use.

AVAILABLE FEATURES:
  Lexical (pure word ID, no class dependency):
    LexBigramFeature    — hash(prev_word, cand_word)
    LexSkipFeature      — hash(prev2_word, cand_word)
    LexTrigramFeature   — hash(prev2_word, prev_word, cand_word)

  Class-word mixed (DYNAMIC — works with ANY class system):
    ClassWordBigramFeature — hash(prev_class, cand_word)  [class_key selectable]
    WordClassBigramFeature — hash(prev_word, cand_class)  [class_key selectable]
    ClassTrigramFeature    — hash(prev2_class, prev_class, cand_class) [class_key selectable]
    ClassWordSkipFeature   — hash(prev2_class, cand_word) [class_key selectable]

DEFAULT FEATURE SET (v82):
  LexBigramFeature(class_key=None),           # pure lexical
  WordClassBigramFeature(class_key="freq"),    # freq word→class
  ClassWordBigramFeature(class_key="freq"),    # freq class→word
  LexSkipFeature(class_key=None),             # pure lexical skip
  WordClassBigramFeature(class_key="dist"),    # dist word→class [NEW!]
  ClassWordBigramFeature(class_key="dist"),    # dist class→word [NEW!]
  ClassTrigramFeature(class_key="dist"),       # dist class 3-gram [NEW!]
  LexTrigramFeature(class_key=None),          # pure lexical 3-gram

  8 features total: 3 lexical + 2 freq-class + 3 dist-class

ADD YOUR OWN FEATURE:
  1. Subclass FeatureSpec, set class_key
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
    its own hash tables, learning rate (eta), clipping range, weight,
    and class_key (selecting which word class system to use).

    The feature's contribution to the total energy is:
        E_feature = weight * sum_h(table[h][hash(inputs, h)])

    v82 CHANGE: Features have a `class_key` attribute that determines
    which word class array they use. class_key=None means pure lexical
    (no class dependency). class_key="freq" uses frequency buckets.
    class_key="dist" uses distributional clusters.
    """

    def __init__(
        self,
        name: str,
        n_hashes: int = 2,
        table_size: int = 65537,
        eta: int = 1,
        clip: int = 100,
        weight: float = 1.0,
        class_key: Optional[str] = None,
    ):
        self.name = name
        self.n_hashes = n_hashes
        self.table_size = table_size
        self.eta = eta
        self.clip = clip
        self.weight = weight
        self.class_key = class_key  # Which class system to use (None = lexical)
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
                        Which class system depends on self.class_key.

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

    def adaptive_clip(self, percentile: int = 99):
        """
        Adaptive clipping: clip to the given percentile of absolute values.

        This prevents energy saturation (tables hitting fixed clip limits)
        while preserving the learned distribution shape. Better than fixed
        clipping because it adapts to the actual learned distribution.
        """
        for h in range(self.n_hashes):
            abs_vals = np.abs(self.tables[h])
            nonzero = abs_vals[abs_vals > 0]
            if len(nonzero) > 0:
                adaptive_limit = max(int(np.percentile(nonzero, percentile)), self.clip // 2)
                np.clip(self.tables[h], -adaptive_limit, adaptive_limit, out=self.tables[h])
            else:
                # All zeros — just use default clip
                np.clip(self.tables[h], -self.clip, self.clip, out=self.tables[h])

    def statistics(self) -> Dict:
        """Return feature statistics."""
        all_vals = np.concatenate([t.ravel() for t in self.tables])
        return {
            'name': self.name,
            'weight': self.weight,
            'class_key': self.class_key,
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
                f"class_key={self.class_key!r}, "
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
    Pure lexical — no class dependency.
    """

    def __init__(self, n_hashes=3, table_size=65537, eta=1, clip=100, weight=1.0):
        super().__init__("lex_bi", n_hashes, table_size, eta, clip, weight, class_key=None)

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
    Pure lexical — no class dependency.
    """

    def __init__(self, n_hashes=2, table_size=65537, eta=1, clip=80, weight=0.3):
        super().__init__("lex_skip", n_hashes, table_size, eta, clip, weight, class_key=None)

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
    Pure lexical — no class dependency.
    """

    def __init__(self, n_hashes=2, table_size=65537, eta=1, clip=80, weight=0.3):
        super().__init__("lex_tri", n_hashes, table_size, eta, clip, weight, class_key=None)

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
# Class-word mixed features (DYNAMIC — works with ANY class system)
#
# v82 KEY CHANGE: class_key parameter selects which class array to use.
# "freq" = frequency buckets (captures importance gradient)
# "dist" = distributional clusters (captures syntactic role)
#
# When class_key="dist", the feature name is suffixed with the class_key
# to avoid name collisions. E.g., "cls_word_bi" with class_key="freq"
# vs "cls_word_bi_dist" with class_key="dist".
# ---------------------------------------------------------------------------

class ClassWordBigramFeature(FeatureSpec):
    """
    hash(prev_class, cand_word) — class→word transitions.

    Works with ANY class system. class_key determines which:
    - "freq": frequency buckets → "after high-freq words, specific words follow"
    - "dist": distributional clusters → "after DET-like words, nouns follow"

    The name is auto-suffixed with class_key for uniqueness.
    """

    def __init__(self, n_hashes=2, table_size=65537, eta=1, clip=50,
                 weight=0.5, class_key="freq"):
        name = f"cls_word_bi_{class_key}" if class_key else "cls_word_bi"
        super().__init__(name, n_hashes, table_size, eta, clip, weight, class_key)

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

    Works with ANY class system. class_key determines which:
    - "freq": "the" → predicts high-freq followers
    - "dist": "the" → predicts DET-like followers (nouns)

    The name is auto-suffixed with class_key for uniqueness.
    """

    def __init__(self, n_hashes=2, table_size=65537, eta=1, clip=50,
                 weight=0.5, class_key="freq"):
        name = f"word_cls_bi_{class_key}" if class_key else "word_cls_bi"
        super().__init__(name, n_hashes, table_size, eta, clip, weight, class_key)

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


class ClassTrigramFeature(FeatureSpec):
    """
    hash(prev2_class, prev_class, cand_class) — class trigram.

    With K=30 dist clusters: 30^3 = 27000 unique patterns.
    Learns patterns like: "DET → NOUN → VERB" (dist clusters)
    or "high-freq → mid-freq → low-freq" (freq buckets).
    """

    def __init__(self, n_hashes=2, table_size=65537, eta=1, clip=50,
                 weight=0.5, class_key="freq"):
        name = f"cls_tri_{class_key}" if class_key else "cls_tri"
        super().__init__(name, n_hashes, table_size, eta, clip, weight, class_key)

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


class ClassWordSkipFeature(FeatureSpec):
    """
    hash(prev2_class, cand_word) — class→word at distance 2.
    """

    def __init__(self, n_hashes=2, table_size=65537, eta=1, clip=50,
                 weight=0.3, class_key="freq"):
        name = f"cls_word_skip_{class_key}" if class_key else "cls_word_skip"
        super().__init__(name, n_hashes, table_size, eta, clip, weight, class_key)

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
    """

    def __init__(self, n_hashes=2, table_size=65537, eta=1, clip=50,
                 weight=0.3, class_key="freq"):
        name = f"word_cls_skip_{class_key}" if class_key else "word_cls_skip"
        super().__init__(name, n_hashes, table_size, eta, clip, weight, class_key)

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


# ---------------------------------------------------------------------------
# Default feature set factory — v82 MULTI-CLASS
# ---------------------------------------------------------------------------

def default_features(
    vocab_size: int = 2000,
    n_freq_classes: int = 20,
    n_dist_classes: int = 30,
    lex_table_size: int = 65537,
    class_table_size: int = 65537,
    tri_table_size: int = 65537,
    class_tri_table_size: int = 65537,
    include_dist: bool = True,
) -> List[FeatureSpec]:
    """
    Create the recommended default feature set for v82.

    MULTI-CLASS: Features use BOTH frequency buckets AND distributional
    clusters simultaneously. This gives the model access to:
    - Frequency-based patterns (importance, gradient)
    - Syntax-based patterns (part-of-speech-like, from data)

    Default 8 features:
      Lexical (3):
        - LexBigramFeature: main workhorse
        - LexSkipFeature: skip-gram
        - LexTrigramFeature: 3-gram collocations

      Frequency bucket class (2):
        - WordClassBigramFeature(class_key="freq"): word→freq-class
        - ClassWordBigramFeature(class_key="freq"): freq-class→word

      Distributional cluster class (3):
        - WordClassBigramFeature(class_key="dist"): word→dist-cluster [NEW!]
        - ClassWordBigramFeature(class_key="dist"): dist-cluster→word [NEW!]
        - ClassTrigramFeature(class_key="dist"): dist-cluster 3-gram [NEW!]

    Total memory: ~5 MB for V=2000, K_freq=20, K_dist=30.
    """
    features = [
        # Lexical features (no class dependency)
        LexBigramFeature(
            n_hashes=3, table_size=lex_table_size,
            eta=1, clip=100, weight=1.0,
        ),
        # Frequency bucket class features
        WordClassBigramFeature(
            n_hashes=2, table_size=class_table_size,
            eta=1, clip=50, weight=0.5, class_key="freq",
        ),
        ClassWordBigramFeature(
            n_hashes=2, table_size=class_table_size,
            eta=1, clip=50, weight=0.5, class_key="freq",
        ),
        # Lexical skip
        LexSkipFeature(
            n_hashes=2, table_size=lex_table_size,
            eta=1, clip=80, weight=0.3,
        ),
    ]

    if include_dist:
        # Distributional cluster class features — THE KEY v82 ADDITION
        features.extend([
            WordClassBigramFeature(
                n_hashes=2, table_size=class_table_size,
                eta=1, clip=50, weight=0.5, class_key="dist",
            ),
            ClassWordBigramFeature(
                n_hashes=2, table_size=class_table_size,
                eta=1, clip=50, weight=0.5, class_key="dist",
            ),
            ClassTrigramFeature(
                n_hashes=2, table_size=class_tri_table_size,
                eta=1, clip=50, weight=0.5, class_key="dist",
            ),
        ])

    # Always include lexical trigram
    features.append(
        LexTrigramFeature(
            n_hashes=2, table_size=tri_table_size,
            eta=1, clip=80, weight=0.3,
        )
    )

    return features


# ===========================================================================
# FeatureHashEnergyTable — MULTI-CLASS dynamic feature registry
# ===========================================================================

class FeatureHashEnergyTable:
    """
    Feature-Hashed Integer Energy Table with MULTI-CLASS word system.

    v82 BREAKING CHANGE: Supports MULTIPLE word class arrays simultaneously.
    Instead of a single word_class array, accepts a dict:
        word_classes = {"freq": word_bucket_array, "dist": word_cluster_array}

    Each feature declares its class_key to select which array to use.
    Features with class_key=None use no class array (pure lexical).
    Features with class_key="freq" use frequency buckets.
    Features with class_key="dist" use distributional clusters.

    This architecture is DYNAMIC:
    - Add new class systems at any time: word_classes["topic"] = topic_array
    - Add features that use new class systems: FeatureSpec(class_key="topic")
    - Remove features that don't help: remove_feature("cls_word_bi_freq")
    - The number and type of features is NOT fixed at compile time

    Usage:
        word_classes = {"freq": vocab.word_bucket, "dist": vocab.word_cluster}
        energy = FeatureHashEnergyTable(vocab_size=2000, word_classes=word_classes)
        for feat in default_features(include_dist=True):
            energy.add_feature(feat)
        energy.train_nce(sequences)
        E = energy.compute_local_energy_batch(context, candidates)
    """

    def __init__(
        self,
        vocab_size: int,
        word_classes: Dict[str, np.ndarray],
        seed: int = 42,
    ):
        self.V = vocab_size
        self.word_classes = {k: v.astype(np.int32) for k, v in word_classes.items()}
        self.seed = seed
        self.features: OrderedDict[str, FeatureSpec] = OrderedDict()

        # Primary class system for balanced negative sampling
        # Use the first available class system (usually "freq")
        self.primary_class_key = next(iter(word_classes.keys())) if word_classes else None
        self.word_class = self.word_classes[self.primary_class_key] if self.primary_class_key else np.zeros(vocab_size, dtype=np.int32)

        # Compute n_classes for each class system
        self.n_classes_map = {}
        for key, arr in self.word_classes.items():
            self.n_classes_map[key] = int(arr.max()) + 1

        # Build class-indexed word lists for BALANCED negative sampling
        # Build for ALL class systems so each feature's class gets balanced negatives
        self._class_word_indices: Dict[str, Dict[int, np.ndarray]] = {}
        for key, arr in self.word_classes.items():
            n_cls = self.n_classes_map[key]
            indices_map = {}
            for cls in range(n_cls):
                indices = np.where(arr == cls)[0]
                if len(indices) > 0:
                    indices_map[cls] = indices.astype(np.int64)
            self._class_word_indices[key] = indices_map

    def add_feature(self, feature: FeatureSpec):
        """Register a feature. Can be called any time before training."""
        # Verify the feature's class_key is available
        if feature.class_key is not None and feature.class_key not in self.word_classes:
            print(f"    WARNING: Feature '{feature.name}' needs class_key="
                  f"'{feature.class_key}' but only {list(self.word_classes.keys())} "
                  f"are available. Skipping.", flush=True)
            return
        self.features[feature.name] = feature

    def remove_feature(self, name: str):
        """Remove a feature by name."""
        if name in self.features:
            del self.features[name]

    def get_feature(self, name: str) -> Optional[FeatureSpec]:
        """Get a feature by name."""
        return self.features.get(name)

    def _get_class_array(self, feature: FeatureSpec) -> np.ndarray:
        """Get the word class array for a feature based on its class_key."""
        if feature.class_key is None:
            return self.word_class  # Fallback (lexical features don't use it)
        return self.word_classes[feature.class_key]

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

        Each feature uses its own class array (determined by class_key).
        Returns integer energy array. Lower = more likely = better.
        """
        K = len(candidates)
        if not context_word_ids:
            return np.zeros(K, dtype=np.int64)

        total = np.zeros(K, dtype=np.float64)
        for feat in self.features.values():
            wc = self._get_class_array(feat)
            e = feat.energy_batch(context_word_ids, candidates, wc)
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
            wc = self._get_class_array(feat)
            total += feat.weight * feat.energy_scalar(
                context_word_ids, candidate, wc
            )
        return int(total)

    # -------------------------------------------------------------------
    # NCE Training — vectorized batch integer updates
    # -------------------------------------------------------------------

    def _sample_balanced_negatives(
        self, rng: np.random.RandomState, size: int,
        class_key: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Sample negative words with BALANCED class distribution.

        Instead of random words (which would be dominated by high-freq classes),
        we pick a random class uniformly from EACH class system, then a random
        word from that class.

        Returns (neg_word_ids, neg_class_dict) where neg_class_dict maps
        class_key -> neg_class_array for each class system.
        """
        # Use primary class system for word selection
        key = class_key or self.primary_class_key
        if key and key in self._class_word_indices:
            valid_classes = list(self._class_word_indices[key].keys())
        else:
            valid_classes = []

        if not valid_classes:
            neg_words = rng.randint(4, self.V, size=size)
        else:
            neg_class_ids = np.array(
                [rng.choice(valid_classes) for _ in range(size)],
                dtype=np.int64,
            )
            neg_words = np.empty(size, dtype=np.int64)
            for cls, indices in self._class_word_indices[key].items():
                mask = neg_class_ids == cls
                n = int(mask.sum())
                if n > 0:
                    neg_words[mask] = rng.choice(indices, size=n)

        # Compute class arrays for ALL class systems
        neg_class_dict = {}
        for ckey, arr in self.word_classes.items():
            neg_class_dict[ckey] = arr[neg_words].astype(np.int64)

        return neg_words, neg_class_dict

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

        KEY v82: Each feature uses its OWN class array (determined by class_key).
        Negative sampling is balanced across ALL class systems.
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

        # Class arrays for EACH class system
        all_class_arrays = {}  # class_key -> (prev_class, target_class, prev2_class)
        for key, arr in self.word_classes.items():
            all_class_arrays[key] = (
                arr[all_prev].astype(np.int64),
                arr[all_target].astype(np.int64),
                arr[all_prev2].astype(np.int64),
            )

        has_prev2 = all_prev2 > 0

        feat_names = [f.name for f in self.features.values()]
        print(f"    {N:,} training pairs", flush=True)
        print(f"    Features: {', '.join(feat_names)}", flush=True)

        # Print class system info
        for key, n_cls in self.n_classes_map.items():
            n_indices = len(self._class_word_indices.get(key, {}))
            print(f"    Class system '{key}': {n_cls} classes, "
                  f"{n_indices} non-empty", flush=True)

        all_stats = []

        for epoch in range(n_epochs):
            t0 = _time.time()

            order = rng.permutation(N)
            sp = all_prev[order]
            st = all_target[order]
            sp2 = all_prev2[order]
            hp2 = has_prev2[order]

            # Permute class arrays for each system
            perm_class = {}
            for key, (pc, tc, p2c) in all_class_arrays.items():
                perm_class[key] = (pc[order], tc[order], p2c[order])

            chunk = 100000
            for c0 in range(0, N, chunk):
                c1 = min(c0 + chunk, N)
                cp = sp[c0:c1]; ct = st[c0:c1]; cp2 = sp2[c0:c1]
                chp2 = hp2[c0:c1]
                C = len(cp)

                # Get class arrays for this chunk (per class system)
                chunk_class = {}
                for key, (pc, tc, p2c) in perm_class.items():
                    chunk_class[key] = (pc[c0:c1], tc[c0:c1], p2c[c0:c1])

                # Positive updates — all features
                for feat in self.features.values():
                    ckey = feat.class_key or self.primary_class_key
                    cpc, ctc, cp2c = chunk_class.get(ckey, (None, None, None))
                    if cpc is None:
                        continue
                    feat.nce_positive(
                        cp, cpc, ct, ctc,
                        prev2_words=cp2, prev2_class=cp2c, mask=chp2,
                    )

                # Negative updates — balanced across primary class system
                for _ in range(n_negatives):
                    neg, neg_class_dict = self._sample_balanced_negatives(rng, C)

                    for feat in self.features.values():
                        ckey = feat.class_key or self.primary_class_key
                        _, ctc, cp2c = chunk_class.get(ckey, (None, None, None))
                        neg_cls = neg_class_dict.get(ckey)
                        if ctc is None or neg_cls is None:
                            continue
                        cpc_pos, _, cp2c_pos = chunk_class.get(ckey, (None, None, None))
                        feat.nce_negative(
                            cp, cpc_pos, neg, neg_cls,
                            prev2_words=cp2, prev2_class=cp2c_pos, mask=chp2,
                        )

            t_elapsed = _time.time() - t0

            # Clip all features (adaptive)
            for feat in self.features.values():
                feat.adaptive_clip(percentile=99)

            # Discriminative accuracy per feature + combined
            n_check = min(2000, N)
            ci = rng.choice(N, n_check, replace=False)
            cp_chk = all_prev[ci]; ct_chk = all_target[ci]
            cp2_chk = all_prev2[ci]
            hp2_chk = has_prev2[ci]

            neg_chk, neg_cls_dict = self._sample_balanced_negatives(rng, n_check)

            # Per-feature discrimination
            feat_discs = {}
            for feat in self.features.values():
                ckey = feat.class_key or self.primary_class_key
                chk_class = all_class_arrays.get(ckey)
                if chk_class is None:
                    feat_discs[feat.name] = 0.5
                    continue

                cpc_chk, ctc_chk, cp2c_chk = (
                    chk_class[0][ci], chk_class[1][ci], chk_class[2][ci]
                )
                neg_cls_chk = neg_cls_dict.get(ckey)
                if neg_cls_chk is None:
                    feat_discs[feat.name] = 0.5
                    continue

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
                ckey = feat.class_key or self.primary_class_key
                chk_class = all_class_arrays.get(ckey)
                if chk_class is None:
                    continue

                cpc_chk, ctc_chk, cp2c_chk = (
                    chk_class[0][ci], chk_class[1][ci], chk_class[2][ci]
                )
                neg_cls_chk = neg_cls_dict.get(ckey)
                if neg_cls_chk is None:
                    continue

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

    def get_class_matrix(self, class_key: Optional[str] = None) -> np.ndarray:
        """
        Return K×K class transition energy matrix for visualization.

        Args:
            class_key: Which class system to visualize. Default = primary.
        """
        key = class_key or self.primary_class_key
        if key not in self.n_classes_map:
            return np.zeros((1, 1))

        K = self.n_classes_map[key]
        matrix = np.zeros((K, K), dtype=np.float64)

        # Find a ClassWordBigramFeature with this class_key
        target_name = f"cls_word_bi_{key}"
        cls_word_feat = self.features.get(target_name)

        if cls_word_feat is not None:
            for c1 in range(min(K, 30)):  # Cap at 30 for performance
                for c2 in range(min(K, 30)):
                    total = sum(
                        int(cls_word_feat.tables[h][_hash2(c1, c2, h, cls_word_feat.table_size)])
                        for h in range(cls_word_feat.n_hashes)
                    )
                    matrix[c1, c2] = total / max(1, cls_word_feat.n_hashes)
            return matrix

        # Try ClassTrigramFeature — marginalize to get bigram
        target_name = f"cls_tri_{key}"
        cls_tri_feat = self.features.get(target_name)

        if cls_tri_feat is not None:
            for c1 in range(min(K, 20)):
                for c2 in range(min(K, 20)):
                    total = 0.0
                    for c0 in range(min(K, 20)):
                        t = sum(
                            int(cls_tri_feat.tables[h][_hash3(c0, c1, c2, h, cls_tri_feat.table_size)])
                            for h in range(cls_tri_feat.n_hashes)
                        )
                        total += t
                    matrix[c1, c2] = total / max(1, min(K, 20) * cls_tri_feat.n_hashes)
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
            'class_systems': list(self.word_classes.keys()),
            'n_classes_map': self.n_classes_map,
            'memory_mb': self.memory_mb(),
            'features': feat_stats,
        }
