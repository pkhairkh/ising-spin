"""
Attractor Language Machine — Dense Associative Memory Engine.

The attractor dynamics of a properly trained Dense Associative Memory
ARE a language model. Not an approximation. Not a component. They ARE one.

Architecture:
  - SDR encoding: Sparse Distributed Representations (~2% active bits)
  - DAM layers: Dense Associative Memory with F-lookup energy (exponential capacity)
  - Hierarchy: L0-Lexical → L1-Syntactic → L2-Semantic → L3-Discourse
  - RG flow: Wilsonian renormalization group between layers (UV-complete)
  - Episodic memory: Content-addressable sparse pattern storage
  - Integer-only Boltzmann sampler (ZERO float ops in hot loop)

Key physics:
  - Energy: E = -Σ F(J_ij * s_i * s_j) - Σ h_i * s_i
  - F-lookup: nonlinear energy function → exponential storage capacity
  - PCD learning: ΔJ = η(data_correlations - model_correlations)
  - UV completeness: couplings renormalizable at all hierarchical scales
  - RG beta functions: g_{l+1} = β(g_l) govern inter-layer flow

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

__version__ = "25.0.0"
