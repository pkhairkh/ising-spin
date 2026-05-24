"""
Backward-compatibility shim — re-exports from utils.py.

All canonical implementations live in ising_spin.utils.
This module exists solely so that any code still importing from
ising_spin.helpers continues to work without modification.
"""

from .utils import TAG_PRIORITY
from .utils import get_rss_mb
from .utils import load_fineweb_edu
from .utils import tokenize_texts
from .utils import truncate_sequences

# Backward-compat alias: helpers.py called it get_primary_pos,
# utils.py calls it primary_pos_tag.
from .utils import primary_pos_tag as get_primary_pos  # noqa: F811

__all__ = [
    "TAG_PRIORITY",
    "get_primary_pos",
    "get_rss_mb",
    "load_fineweb_edu",
    "tokenize_texts",
    "truncate_sequences",
]
