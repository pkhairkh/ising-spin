"""
Attractor Language Machine v28 — DEEP FIXES.

The attractor dynamics of a properly trained Dense Associative Memory
ARE a language model. Not an approximation. Not a component. They ARE one.

Architecture:
  - SDR encoding: Sparse Distributed Representations (~2% active bits)
  - DAM layers: Dense Associative Memory with F-lookup energy (exponential capacity)
  - F function: exp_approx (piecewise integer exponential) — TRUE exponential capacity
  - Hierarchy: L0-Lexical -> L1-Syntactic -> L2-Semantic -> L3-Discourse
  - RG flow: J_eff DERIVED from L0 via coupling-space decimation, REPLACES J at higher levels
  - UV completeness: Ward identities + cutoff independence + coupling flow stability
  - Episodic memory: Content-addressable sparse pattern storage
  - Integer-only Boltzmann sampler (ZERO float ops in hot loop)

DEEP FIXES (v28):
  1. F_EXP_APPROX: piecewise integer exponential — TRUE exponential capacity
  2. RG-derived J_eff REPLACES J at higher levels (not just diagnostic)
  3. Ward identity UV checks (not just spectral gap / cutoff sensitivity)
  4. Pure Hebbian ONLY (PCD removed — unnecessary at right sparsity)
  5. Anomalous dimensions from operator spectrum of J (not running correlations)
  6. DAM energy alone drives word selection (no n-gram crutch)
  7. D decreasing: 512->256->128->64 (RG reduces DOF at coarser scales)

All integer arithmetic. Zero floats in the hot path. Runs on Pi 5.
"""

from .vocabulary import Vocabulary, POSTypeSystem, TopicAssigner
from .vocabulary.pos import (
    COARSE_POS_TAGS, POS2IDX, IDX2POS, N_POS,
    NOUN_LIKE, VERB_LIKE, OPEN_CLASS, CLOSED_CLASS,
)
from .sampling import IntegerBoltzmannSampler, LN2_NUM, LN2_DEN, LOG2_SCALE
from .attractor import (
    SDREncoder,
    DAMLayer,
    HierarchicalDAM,
    EpisodicMemory,
    AttractorLanguageModel,
)
from .exceptions import (
    AttractorError,
    BuildError, VocabularyBuildError, CorpusError,
    TopicBuildError,
    InferenceError, SamplingError, EnergyError,
    ValidationError, VocabularyError, POSValidationError, ConfigError,
)
from .utils import (
    get_rss_mb, TAG_PRIORITY, primary_pos_tag,
    load_fineweb_edu, load_tinystories, load_tiny_textbooks,
    load_writingprompts, DATASET_LOADERS, DEFAULT_DATASET,
)

__version__ = "28.0.0"
