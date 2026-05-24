"""
Multi-scale recall module for the Ising Spin Glass Language Model.

Three recall indexes at different abstraction levels:
  - WordNgramIndex:  Exact word n-gram matching (max_n=5)
  - PosNgramIndex:   POS-tag n-gram matching (max_n=15, much less sparse)
  - TopicNgramIndex: Topic n-gram matching (max_n=10, discourse-level)

Combined via MultiScaleRecall using product-of-experts energy fusion.

Shared build/prune/energy logic lives in NgramIndexBase (recall/base.py).
"""

from .base import NgramIndexBase
from .word_index import WordNgramIndex
from .pos_index import PosNgramIndex
from .topic_index import TopicNgramIndex
from .multiscale import MultiScaleRecall

__all__ = [
    "NgramIndexBase",
    "WordNgramIndex",
    "PosNgramIndex",
    "TopicNgramIndex",
    "MultiScaleRecall",
]
