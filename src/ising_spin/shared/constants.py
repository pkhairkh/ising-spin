"""
Deduplicated constants used across the Ising Spin Glass Language Model.

Previously, TAG_PRIORITY was copy-pasted into helpers.py, generator.py,
pos_index.py, and pos.py.  SPECIAL_TOKEN_COUNT / SENT_TOKEN_IDX / NS
were magic numbers scattered across word_index.py, pos_index.py,
topic_index.py, and generator.py.  CLOSED_CLASS_POS_TAGS was duplicated
between pos.py and generator.py.

This module is the single source of truth for all such constants.
"""

from __future__ import annotations

from ..vocabulary.pos import POS2IDX


# ===========================================================================
# POS TAG PRIORITY
# ===========================================================================

# POS tag disambiguation priority (lower = higher priority).
#
# When a word has multiple allowed POS tags (e.g. "run" can be NOUN or
# VERB), the tag with the LOWEST priority value is chosen as the
# *primary* tag.  The ordering favours closed-class tags (PUNCT, DET,
# PRON, ...) because they are more specific — a word that can be a
# determiner is almost always functioning as a determiner.
#
# Duplicated out of:
#   - helpers.py   (TAG_PRIORITY)
#   - generator.py (_get_word_type local dict)
#   - pos_index.py (_TAG_PRIORITY module-level dict)
#   - pos.py       (compute_type_couplings local dict)
TAG_PRIORITY: dict[int, int] = {
    POS2IDX["PUNCT"]: 0,
    POS2IDX["DET"]: 1,
    POS2IDX["PRON"]: 2,
    POS2IDX["AUX"]: 3,
    POS2IDX["CONJ"]: 4,
    POS2IDX["PART"]: 5,
    POS2IDX["PREP"]: 6,
    POS2IDX["NUM"]: 7,
    POS2IDX["ADV"]: 8,
    POS2IDX["ADJ"]: 9,
    POS2IDX["NOUN"]: 10,
    POS2IDX["VERB"]: 11,
    POS2IDX["X"]: 12,
}


# ===========================================================================
# SPECIAL TOKEN CONSTANTS
# ===========================================================================

SPECIAL_TOKEN_COUNT: int = 5
"""Number of reserved special token indices at the start of the vocabulary.

Layout:
    0 = UNK  (unknown)
    1 = BOS  (beginning of sequence)
    2 = EOS  (end of sequence)
    3 = PAD  (padding)
    4 = SENT (sentence boundary)

Tokens with index < SPECIAL_TOKEN_COUNT are excluded from n-gram contexts
and continuations.

Previously hard-coded as ``NS = 5`` in word_index.py, pos_index.py,
topic_index.py, and as ``idx >= 5`` in generator.py.
"""

SENT_TOKEN_IDX: int = 4
"""Index of the sentence-boundary token (<S>) in the vocabulary.

The SENT token acts as a hard barrier: n-gram contexts are truncated at
the last SENT occurrence so that lookups never cross sentence boundaries.

Previously hard-coded as ``SENT_IDX = 4`` in word_index.py, pos_index.py,
topic_index.py, and generator.py.
"""


# ===========================================================================
# CLOSED-CLASS POS TAGS
# ===========================================================================

CLOSED_CLASS_POS_TAGS: frozenset[int] = frozenset({
    POS2IDX["DET"],
    POS2IDX["PREP"],
    POS2IDX["PRON"],
    POS2IDX["AUX"],
    POS2IDX["CONJ"],
    POS2IDX["PART"],
})
"""POS tags that belong to the *closed class*.

Closed-class words (determiners, prepositions, pronouns, auxiliaries,
conjunctions, particles) have a fixed, small membership.  They are
treated specially in several places:

  - generator.py uses CLOSED_CLASS_IDS for anti-loop penalties.
  - pos.py defines CLOSED_CLASS for grammar penalty construction.
  - energy/computer.py may apply different scaling to closed-class runs.

Using a frozenset makes membership tests O(1) and signals immutability.
"""
