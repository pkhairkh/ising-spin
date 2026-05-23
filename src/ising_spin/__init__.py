"""
Ising Spin Glass Language Model — unified architecture.

Energy terms (additive; disabled terms contribute 0):
  1. E_recall      — word n-gram (5-gram)
  2. E_pos         — POS n-gram (10-gram)
  3. E_topic       — topic n-gram (10-gram)
  4. E_dense_am    — Dense Associative Memory (random features)
  5. E_vsa         — VSA qFHRR binding (compositional encoding)
  6. E_reservoir   — Integer ESN Reservoir (~50 token lookback)
  7. E_rff         — Cross-Scale RFF (joint word+POS+topic)
  8. E_coupling    — Factorial state coupling (mean-field)
  9. E_state       — Document state (7 evolving integer variables)
 10. E_hard        — Hard constraints (POS type, same-word, etc.)

ALL computation is integer-only. Float operations appear ONLY during
module initialization (LUT construction, random matrix generation).
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
from .vsa import QFHRRVectors, VSAEncoder
from .dense_am import RandomFeatureProjector, DenseAMEnergy
from .reservoir import IntegerESN
from .rff import CrossScaleRFF
from .model import IsingLMModel, ModelConfig
from .errors import (
    IsingError, BuildError, VocabularyError, CorpusError,
    IndexBuildError, PreAggregationError, InferenceError,
    EnergyError, SamplingError, StateError, ValidationError, ConfigError,
)
from .helpers import (
    TAG_PRIORITY, get_primary_pos, get_rss_mb,
    load_fineweb_edu, tokenize_texts, truncate_sequences,
)

__version__ = "19.0.0"
