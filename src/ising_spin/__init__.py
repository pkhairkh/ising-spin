"""
Ising Spin Glass Language Model — v17.0

Multi-Scale Abstract Recall + Evolving Document State.

Architecture:
    - Word-level n-gram recall (5-gram) — exact word context
    - POS-level n-gram recall (15-gram) — abstract syntactic generalization
    - Topic-level n-gram recall (10-gram) — discourse coherence
    - Document state (7 evolving integer variables) — full-document context
    - Product of Experts fusion — each scale vetoes the others' mistakes
    - Integer-only Boltzmann sampler (ZERO float ops in hot loop)

v17 Key Insight:
    v1-v16 had recall + weak perturbation layers → recall dominated everything.
    v17 has recall at MULTIPLE SCALES → each scale independently constrains
    predictions. No single scale dominates. They VETO each other's mistakes.
    
    When the 5-word n-gram is unseen, the POS 10-gram IS seen (thousands of
    times). When the POS n-gram is ambiguous, the topic n-gram disambiguates.
    When all n-grams miss, the document state carries discourse coherence
    across the entire document.
"""

from .vocabulary import Vocabulary, POSTypeSystem, TopicAssigner
from .vocabulary.pos import (
    COARSE_POS_TAGS, POS2IDX, IDX2POS, N_POS,
    NOUN_LIKE, VERB_LIKE, OPEN_CLASS, CLOSED_CLASS,
)
from .sampling import IntegerBoltzmannSampler, LN2_NUM, LN2_DEN, LOG2_SCALE
from .recall import (
    WordNgramIndex, PosNgramIndex, TopicNgramIndex, MultiScaleRecall,
)
from .state import DocumentState
from .energy import EnergyComputer
from .model_v17 import IsingLMModel

__version__ = "17.0.0"
