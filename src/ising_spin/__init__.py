"""
Ising Spin Glass Language Model — Multi-Scale Abstract Recall + Evolving Document State.

Architecture:
    - Word-level n-gram recall (5-gram) — exact word context
    - POS-level n-gram recall (15-gram) — abstract syntactic generalization
    - Topic-level n-gram recall (10-gram) — discourse coherence
    - Document state (7 evolving integer variables) — full-document context
    - Additive energy fusion — each scale contributes independently
    - Integer-only Boltzmann sampler (ZERO float ops in hot loop)

v18 extensions:
    - Factorial State Coupling: 5 pairwise compatibility tables, mean-field inference
    - Integer ESN Reservoir: long-range temporal dynamics (~50 token lookback)
    - VSA/qFHRR: compositional vector symbolic architecture for cross-scale binding
    - Dense Associative Memory: random feature energy with polynomial nonlinearity
    - Cross-Scale RFF: joint word+POS+topic random feature projection
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
    IsingSpinError, IsingError,
    BuildError, VocabularyBuildError, CorpusError, IndexBuildError,
    PreAggregationError, StateBuildError, TopicBuildError,
    InferenceError, SamplingError, EnergyError, StateError,
    ValidationError, VocabularyError, POSValidationError,
    StateValidationError, ConfigError, ConfigurationError,
)
from .utils import (
    get_rss_mb, TAG_PRIORITY, primary_pos_tag,
    load_fineweb_edu, load_tinystories, load_tiny_textbooks,
    load_writingprompts, DATASET_LOADERS, DEFAULT_DATASET,
)

# v18 modules
from .reservoir import IntegerESN
from .vsa import QFHRRVectors, VSAEncoder
from .dense_am import RandomFeatureProjector, DenseAMEnergy
from .rff import CrossScaleRFF

__version__ = "18.0.0"
