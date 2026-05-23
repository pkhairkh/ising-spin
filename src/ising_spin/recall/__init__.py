"""
Multi-scale recall module for the Ising Spin Glass Language Model v17.

Three recall indexes at different abstraction levels:
  - WordNgramIndex:  Exact word n-gram matching (max_n=5)
  - PosNgramIndex:   POS-tag n-gram matching (max_n=15, much less sparse)
  - TopicNgramIndex: Topic n-gram matching (max_n=10, discourse-level)

Combined via MultiScaleRecall using product-of-experts energy fusion.
"""

from .base import AbstractRecallIndex
from .word_index import WordNgramIndex
from .pos_index import PosNgramIndex
from .topic_index import TopicNgramIndex
from .multiscale import MultiScaleRecall

__all__ = [
    "AbstractRecallIndex",
    "WordNgramIndex",
    "PosNgramIndex",
    "TopicNgramIndex",
    "MultiScaleRecall",
]
