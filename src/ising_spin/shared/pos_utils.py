"""
POS tag utility functions — deduplicated from across the codebase.

Previously, ``get_primary_pos`` (or equivalent lambda/logic) was
copy-pasted into helpers.py, generator.py, pos_index.py, and pos.py.
The ``word_to_pos_tag`` / ``seq_to_pos_tags`` helpers were inlined in
pos_index.py as ``_word_to_pos`` / ``_seq_to_pos``.

All functions are integer-only on hot paths and use input validation
via ``ising_spin.errors.ValidationError``.
"""

from __future__ import annotations

from ..errors import ValidationError
from ..vocabulary.pos import POS2IDX
from .constants import TAG_PRIORITY


def get_primary_pos(allowed_types: set[int]) -> int:
    """Return the primary (most-specific) POS tag from an allowed set.

    Uses :data:`TAG_PRIORITY` to disambiguate: the tag with the lowest
    priority value wins.  Closed-class tags (PUNCT, DET, PRON, …) have
    higher priority (lower number) because they are more specific.

    Args:
        allowed_types: Set of allowed POS tag indices for a word.

    Returns:
        The primary POS tag index.  If *allowed_types* is empty,
        returns ``POS2IDX["X"]`` (the catch-all tag).

    Raises:
        ValidationError: If *allowed_types* is not a set or contains
            non-integer values.

    Examples::

        >>> get_primary_pos({0, 1})            # NOUN(0), VERB(1) → NOUN
        0
        >>> get_primary_pos(set())             # empty → X
        12
    """
    if not isinstance(allowed_types, (set, frozenset)):
        raise ValidationError(
            f"allowed_types must be a set, got {type(allowed_types).__name__}"
        )
    if not allowed_types:
        return POS2IDX["X"]
    # Validate that all elements are integers
    for t in allowed_types:
        if not isinstance(t, int):
            raise ValidationError(
                f"allowed_types must contain integers, got {type(t).__name__}"
            )
    return min(allowed_types, key=lambda t: TAG_PRIORITY.get(t, 99))


def word_to_pos_tag(
    word_id: int,
    word_pos_tags: dict[int, int],
    default: int = 12,
) -> int:
    """Convert a word ID to its primary POS tag.

    This is a simple dict lookup with a default fallback.  It replaces
    the inline ``self.word_pos_tags.get(word_id, POS2IDX["X"])``
    pattern that was scattered across pos_index.py.

    Args:
        word_id:     Word index to look up.
        word_pos_tags: Mapping from word index → primary POS tag index.
        default:     POS tag to return when *word_id* is not in the
                     mapping.  Defaults to ``12`` (``POS2IDX["X"]``).

    Returns:
        The POS tag index for *word_id*, or *default* if not found.

    Raises:
        ValidationError: If *word_id* is negative, *word_pos_tags* is
            not a dict, or *default* is not a non-negative integer.

    Examples::

        >>> word_to_pos_tag(42, {42: 0, 43: 1})   # word 42 → NOUN(0)
        0
        >>> word_to_pos_tag(99, {42: 0})           # not found → X(12)
        12
        >>> word_to_pos_tag(99, {42: 0}, default=1) # not found → VERB(1)
        1
    """
    if not isinstance(word_id, int) or word_id < 0:
        raise ValidationError(
            f"word_id must be a non-negative integer, got {word_id!r}"
        )
    if not isinstance(word_pos_tags, dict):
        raise ValidationError(
            f"word_pos_tags must be a dict, got {type(word_pos_tags).__name__}"
        )
    if not isinstance(default, int) or default < 0:
        raise ValidationError(
            f"default must be a non-negative integer, got {default!r}"
        )
    return word_pos_tags.get(word_id, default)


def seq_to_pos_tags(
    seq: list[int],
    word_pos_tags: dict[int, int],
) -> list[int]:
    """Convert a word ID sequence to a POS tag sequence.

    Applies :func:`word_to_pos_tag` to every element of *seq*.

    Args:
        seq:           List of word indices.
        word_pos_tags: Mapping from word index → primary POS tag index.

    Returns:
        List of POS tag indices with the same length as *seq*.

    Raises:
        ValidationError: If *seq* is not a list or contains negative
            integers, or if *word_pos_tags* is not a dict.

    Examples::

        >>> seq_to_pos_tags([42, 43, 44], {42: 0, 43: 1, 44: 0})
        [0, 1, 0]
    """
    if not isinstance(seq, list):
        raise ValidationError(
            f"seq must be a list, got {type(seq).__name__}"
        )
    if not isinstance(word_pos_tags, dict):
        raise ValidationError(
            f"word_pos_tags must be a dict, got {type(word_pos_tags).__name__}"
        )
    # Validate seq elements are non-negative integers
    for i, w in enumerate(seq):
        if not isinstance(w, int) or w < 0:
            raise ValidationError(
                f"seq[{i}] must be a non-negative integer, got {w!r}"
            )
    return [word_pos_tags.get(w, POS2IDX["X"]) for w in seq]
