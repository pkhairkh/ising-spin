"""
Ising Spin Glass Language Model — Multi-Scale Abstract Recall + Evolving Document State.

Architecture:
    - Word-level n-gram recall (5-gram) — exact word context
    - POS-level n-gram recall (15-gram) — abstract syntactic generalization
    - Topic-level n-gram recall (10-gram) — discourse coherence
    - Document state (7 evolving integer variables) — full-document context
    - Product of Experts fusion — each scale vetoes the others' mistakes
    - Integer-only Boltzmann sampler (ZERO float ops in hot loop)
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
from .orchestrator import IsingLMModel
from .exceptions import (
    IsingSpinError, BuildError, VocabularyBuildError, IndexBuildError,
    StateBuildError, TopicBuildError, InferenceError, SamplingError,
    EnergyError, ValidationError, VocabularyError, POSValidationError,
    StateValidationError, ConfigurationError,
)
from .utils import get_rss_mb, TAG_PRIORITY, primary_pos_tag

__version__ = "17.1.0"
