"""
Attractor Language Machine v27 — Dense Associative Memory Engine.

The attractor dynamics of a properly trained Dense Associative Memory
ARE a language model. Not an approximation. Not a component. They ARE one.

Architecture:
  - SDR encoding: Sparse Distributed Representations (~2% active bits)
  - DAM layers: Dense Associative Memory with F-lookup energy (exponential capacity)
  - Hierarchy: L0-Lexical → L1-Syntactic → L2-Semantic → L3-Discourse
  - RG flow: Coupling-space Wilsonian RG (decimation of J matrices)
  - UV completeness: Cutoff independence + coupling flow stability
  - Episodic memory: Content-addressable sparse pattern storage
  - Integer-only Boltzmann sampler (ZERO float ops in hot loop)

Key physics (v27 fixes based on knowledge base analysis):
  - Energy: E = -Σ F(J_ij * s_i * s_j) with NONLINEAR F (exponential capacity)
  - Hebbian learning: RG fixed point at right sparsity (Agliari 2025, Eugenio 2025)
  - Coupling-space RG: J_eff[l+1] = Decimate(J[l]), NOT state-space block-spin
  - UV completeness: cutoff independence + operator spectrum stability
  - Anomalous dimensions: from operator spectrum of J, NOT running correlations

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

__version__ = "27.0.0"
