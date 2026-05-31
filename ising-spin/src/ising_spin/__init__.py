"""
Integer Language Model v82 — pure integer, no neural nets, runs on a Pi 5.

A 100% interpretable, integer-only language model that produces
grammatically coherent text for simple domains using:
  - Bigram counting for base probabilities
  - DYNAMIC feature registry: add/remove features at runtime
  - MULTI-CLASS word system: frequency buckets + distributional clusters
  - Features declare which class system they use via class_key
  - Class-balanced NCE training
  - Metropolis gate for grammaticality enforcement
  - N-gram blocking to prevent repetition loops
  - LEGD: P(c) proportional to P_base(c) * exp(-alpha * E_norm(c))

v82 BREAKING CHANGE — MULTI-CLASS ARCHITECTURE:
  - MULTIPLE word class systems running simultaneously
  - Frequency buckets ("freq"): K=20, captures importance gradient
  - Distributional clusters ("dist"): K=30, captures syntactic role
  - Features declare which class system via class_key parameter
  - New features for distributional clusters (word→dist, dist→word, dist-tri)
  - N-gram blocking for generation (prevent repeated bigrams)
  - Adaptive clipping for energy tables

No torch. No neural nets. No float32 in the hot path. ~25 MB memory.
"""

from .vocabulary import Vocabulary, COARSE_POS, POS2IDX, IDX2POS, N_POS
from .bigram_model import BigramModel
from .feature_hash_energy import (
    FeatureHashEnergyTable,
    FeatureSpec,
    default_features,
    # Lexical features (pure word ID, no class dependency)
    LexBigramFeature,
    LexSkipFeature,
    LexTrigramFeature,
    # Class-word mixed features (DYNAMIC — works with ANY class system)
    ClassWordBigramFeature,
    WordClassBigramFeature,
    ClassWordSkipFeature,
    WordClassSkipFeature,
    ClassTrigramFeature,
)
from .boltzmann import IntegerBoltzmannSampler
from .integer_lm import IntegerLM

__all__ = [
    "Vocabulary",
    "COARSE_POS", "POS2IDX", "IDX2POS", "N_POS",
    "BigramModel",
    "FeatureHashEnergyTable",
    "FeatureSpec",
    "default_features",
    "LexBigramFeature",
    "LexSkipFeature",
    "LexTrigramFeature",
    "ClassWordBigramFeature",
    "WordClassBigramFeature",
    "ClassWordSkipFeature",
    "WordClassSkipFeature",
    "ClassTrigramFeature",
    "IntegerBoltzmannSampler",
    "IntegerLM",
]
