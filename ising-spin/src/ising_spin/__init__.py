"""
Integer Language Model v81 — pure integer, no neural nets, runs on a Pi 5.

A 100% interpretable, integer-only language model that produces
grammatically coherent text for simple domains using:
  - Bigram counting for base probabilities
  - DYNAMIC feature registry: add/remove features at runtime
  - DATA-DRIVEN word classes (frequency buckets), NOT static POS tags
  - Class-balanced NCE training
  - Metropolis gate for grammaticality enforcement
  - LEGD: P(c) proportional to P_base(c) * exp(-alpha * E_norm(c))

v81 BREAKING CHANGE:
  - ALL POS dependency removed from features
  - Word classes are frequency buckets (K=20, balanced, data-driven)
  - NOT static POS tags (K=13, 88% NOUN, degenerate)
  - hash(word, class) has V*K=40000 keys vs hash(word, pos) ≈ V*1=2000

No torch. No neural nets. No float32 in the hot path. ~20 MB memory.
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
    # Class-word mixed features (DATA-DRIVEN, replaces all POS features)
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
