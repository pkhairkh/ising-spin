"""
Integer Language Model v85 — pure integer, no neural nets, runs on a Pi 5.

A 100% interpretable, integer-only language model that produces
grammatically coherent text for simple domains using:
  - Bigram counting for base probabilities
  - DYNAMIC feature registry: add/remove features at runtime
  - MULTI-CLASS word system: frequency buckets + distributional clusters
  - Features declare which class system they use via class_key
  - Per-feature class-balanced NCE training
  - Per-feature NCE subsampling (nce_rate) to prevent class feature saturation
  - Metropolis gate for grammaticality enforcement
  - Soft exponential decay repetition penalty
  - LEGD: P(c) proportional to P_base(c) * exp(-alpha * E_norm(c))

v85 FIXES over v84 (PPL 15.07 — REGRESSION from v83's 13.77):
  - v84 raised class clip 50→200 + sqrt(n_classes) scaling → energy explosion
  - v85: Reverted class clip to 50, reverted adaptive_clip to v83 formula
  - v85: Added per-feature NCE subsampling (nce_rate=0.02 for class features)
    This reduces update density from ~2950 to ~59 updates/pattern/epoch,
    preventing saturation without needing higher clips

No torch. No neural nets. No float32 in the hot path. ~28 MB memory.
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
