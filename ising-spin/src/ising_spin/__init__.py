"""
Integer Language Model v84 — pure integer, no neural nets, runs on a Pi 5.

A 100% interpretable, integer-only language model that produces
grammatically coherent text for simple domains using:
  - Bigram counting for base probabilities
  - DYNAMIC feature registry: add/remove features at runtime
  - MULTI-CLASS word system: frequency buckets + distributional clusters
  - Features declare which class system they use via class_key
  - Per-feature class-balanced NCE training
  - Per-feature adaptive clip scaling (anti-saturation)
  - Metropolis gate for grammaticality enforcement
  - Soft exponential decay repetition penalty
  - LEGD: P(c) proportional to P_base(c) * exp(-alpha * E_norm(c))

v84 FIXES over v83 (PPL 13.77):
  - Per-feature adaptive clip scaling: class features get higher limits
    based on sqrt(n_classes) to prevent saturation at clip boundaries
  - Added cls_tri_freq to default features (9 total, was 8)
  - Per-feature class-balanced negatives: dist features get dist-balanced
    negatives instead of sharing freq-balanced negatives
  - Softer bigram repetition penalty: exponential decay instead of hard kill
  - Class feature clips raised from 50 → 200 to prevent saturation

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
