"""
Shared utilities for the Ising Spin Glass Language Model.

This package deduplicates common constants and helper functions that were
previously copy-pasted across 5+ files.  Import from the top-level
``ising_spin.shared`` namespace::

    from ising_spin.shared import TAG_PRIORITY, get_rss_mb
    from ising_spin.shared import get_primary_pos, word_to_pos_tag, seq_to_pos_tags
    from ising_spin.shared import truncate_at_sentence_boundary, find_effective_start
    from ising_spin.shared import is_special_token, SPECIAL_TOKEN_COUNT, SENT_TOKEN_IDX
    from ising_spin.shared import CLOSED_CLASS_POS_TAGS

Sub-modules
-----------
- **constants**  : TAG_PRIORITY, SPECIAL_TOKEN_COUNT, SENT_TOKEN_IDX,
                   CLOSED_CLASS_POS_TAGS
- **pos_utils**  : get_primary_pos, word_to_pos_tag, seq_to_pos_tags
- **memory**     : get_rss_mb
- **boundaries** : truncate_at_sentence_boundary, find_effective_start,
                   is_special_token
"""

# --- Constants ---
from .constants import (
    CLOSED_CLASS_POS_TAGS,
    SENT_TOKEN_IDX,
    SPECIAL_TOKEN_COUNT,
    TAG_PRIORITY,
)

# --- POS utilities ---
from .pos_utils import (
    get_primary_pos,
    seq_to_pos_tags,
    word_to_pos_tag,
)

# --- Memory monitoring ---
from .memory import get_rss_mb

# --- Sentence boundary utilities ---
from .boundaries import (
    find_effective_start,
    is_special_token,
    truncate_at_sentence_boundary,
)

__all__ = [
    # Constants
    "TAG_PRIORITY",
    "SPECIAL_TOKEN_COUNT",
    "SENT_TOKEN_IDX",
    "CLOSED_CLASS_POS_TAGS",
    # POS utilities
    "get_primary_pos",
    "word_to_pos_tag",
    "seq_to_pos_tags",
    # Memory monitoring
    "get_rss_mb",
    # Sentence boundary utilities
    "truncate_at_sentence_boundary",
    "find_effective_start",
    "is_special_token",
]
