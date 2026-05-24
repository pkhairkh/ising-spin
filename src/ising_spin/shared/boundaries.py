"""
Sentence boundary utilities — deduplicated from across the codebase.

Sentence-boundary handling was duplicated in:
  - recall/word_index.py  (lookup: truncate at SENT; build: find effective start)
  - recall/pos_index.py   (lookup: truncate at SENT; build: find effective start)
  - recall/topic_index.py (lookup: truncate at SENT; build: find effective start)
  - generator.py          (prompt filtering: skip tokens with idx >= 5)

All functions are integer-only and use input validation via
``ising_spin.errors.ValidationError``.
"""

from __future__ import annotations

from ..errors import ValidationError
from .constants import SENT_TOKEN_IDX, SPECIAL_TOKEN_COUNT


def is_special_token(
    token_id: int,
    special_count: int = SPECIAL_TOKEN_COUNT,
) -> bool:
    """Check whether *token_id* refers to a special (reserved) token.

    Special tokens occupy indices ``0 .. special_count - 1`` in the
    vocabulary (UNK, BOS, EOS, PAD, SENT).

    Args:
        token_id:      Token index to test.
        special_count: Number of special tokens at the start of the
                       vocabulary.  Defaults to
                       :data:`SPECIAL_TOKEN_COUNT` (5).

    Returns:
        ``True`` if *token_id* is a special token, ``False`` otherwise.

    Raises:
        ValidationError: If *token_id* is not an integer or is negative,
            or if *special_count* is not a positive integer.

    Examples::

        >>> is_special_token(0)    # UNK
        True
        >>> is_special_token(4)    # SENT
        True
        >>> is_special_token(5)    # first content word
        False
        >>> is_special_token(3, special_count=4)
        True
        >>> is_special_token(4, special_count=4)
        False
    """
    if not isinstance(token_id, int) or token_id < 0:
        raise ValidationError(
            f"token_id must be a non-negative integer, got {token_id!r}"
        )
    if not isinstance(special_count, int) or special_count <= 0:
        raise ValidationError(
            f"special_count must be a positive integer, got {special_count!r}"
        )
    return token_id < special_count


def truncate_at_sentence_boundary(
    context_words: list[int],
    sent_idx: int = SENT_TOKEN_IDX,
) -> list[int]:
    """Truncate a context word list at the last sentence boundary.

    N-gram lookups must not cross sentence boundaries.  This function
    finds the rightmost sentence-boundary token (default index 4, <S>)
    and returns only the words that follow it.

    If the context contains no sentence boundary, it is returned
    unchanged.

    Args:
        context_words: List of word indices (typically the generation
                       context).
        sent_idx:      Index of the sentence-boundary token in the
                       vocabulary.  Defaults to
                       :data:`SENT_TOKEN_IDX` (4).

    Returns:
        A (possibly truncated) list of word indices containing only
        tokens after the last sentence boundary.

    Raises:
        ValidationError: If *context_words* is not a list, contains
            negative integers, or if *sent_idx* is not a non-negative
            integer.

    Examples::

        >>> truncate_at_sentence_boundary([5, 6, 4, 7, 8])
        [7, 8]
        >>> truncate_at_sentence_boundary([5, 6, 7])
        [5, 6, 7]
        >>> truncate_at_sentence_boundary([4, 5, 4, 6])
        [6]
    """
    if not isinstance(context_words, list):
        raise ValidationError(
            f"context_words must be a list, got {type(context_words).__name__}"
        )
    if not isinstance(sent_idx, int) or sent_idx < 0:
        raise ValidationError(
            f"sent_idx must be a non-negative integer, got {sent_idx!r}"
        )
    for i, w in enumerate(context_words):
        if not isinstance(w, int) or w < 0:
            raise ValidationError(
                f"context_words[{i}] must be a non-negative integer, got {w!r}"
            )

    last_sent = -1
    for i, w in enumerate(context_words):
        if w == sent_idx:
            last_sent = i

    if last_sent >= 0:
        return context_words[last_sent + 1:]
    return context_words


def find_effective_start(
    seq: list[int],
    sent_idx: int = SENT_TOKEN_IDX,
    special_count: int = SPECIAL_TOKEN_COUNT,
) -> int:
    """Find the effective start position in a token sequence.

    During n-gram index construction, the first few tokens in a sequence
    may be special tokens (UNK, BOS, EOS, PAD) or sentence boundaries
    (<S>).  This function scans from the beginning to find the first
    position that contains a *content* token (index >= *special_count*).

    The scan rules are:
      1. If a content token (index >= *special_count*) is encountered,
         that position is the effective start.
      2. If a sentence boundary token (index == *sent_idx*) is
         encountered *before* any content token, the effective start
         is set to the position immediately after it (because n-grams
         should not span across sentence boundaries).
      3. Other special tokens (UNK, BOS, EOS, PAD) before the first
         content token are skipped.

    Args:
        seq:           List of token indices.
        sent_idx:      Index of the sentence-boundary token.
                       Defaults to :data:`SENT_TOKEN_IDX` (4).
        special_count: Number of special tokens.  Defaults to
                       :data:`SPECIAL_TOKEN_COUNT` (5).

    Returns:
        The effective start index into *seq*.  If no content token is
        found, returns ``len(seq)`` (empty effective range).

    Raises:
        ValidationError: If *seq* is not a list, contains negative
            integers, or if *sent_idx* / *special_count* are invalid.

    Examples::

        >>> find_effective_start([1, 4, 5, 6])   # BOS, <S>, word, word
        2
        >>> find_effective_start([5, 6, 7])       # starts with content
        0
        >>> find_effective_start([0, 1, 2, 5])    # UNK, BOS, EOS, word
        3
        >>> find_effective_start([4, 4, 5])        # <S>, <S>, word
        2
        >>> find_effective_start([0, 1, 2, 3])     # all special, no content
        4
    """
    if not isinstance(seq, list):
        raise ValidationError(
            f"seq must be a list, got {type(seq).__name__}"
        )
    if not isinstance(sent_idx, int) or sent_idx < 0:
        raise ValidationError(
            f"sent_idx must be a non-negative integer, got {sent_idx!r}"
        )
    if not isinstance(special_count, int) or special_count <= 0:
        raise ValidationError(
            f"special_count must be a positive integer, got {special_count!r}"
        )
    for i, w in enumerate(seq):
        if not isinstance(w, int) or w < 0:
            raise ValidationError(
                f"seq[{i}] must be a non-negative integer, got {w!r}"
            )

    effective_start = 0
    for i, w in enumerate(seq):
        if w >= special_count:
            # First content token — start here
            effective_start = i
            break
        elif w == sent_idx:
            # Sentence boundary — start after it
            effective_start = i + 1
            break
    else:
        # No content token or <S> found in the entire sequence.
        # effective_start remains 0, but since the loop exhausted seq
        # without finding a content word, return len(seq) to indicate
        # an empty effective range.
        return len(seq)

    return effective_start
