"""
Integer Language Model v80 — pure integer, no neural nets, runs on a Pi 5.

A 100% interpretable, integer-only language model that produces
grammatically coherent text for simple domains using:
  - Bigram counting for base probabilities
  - DYNAMIC feature registry: add/remove features at runtime
  - Mixed word-POS features (hash(word, pos)) — no more static 13x13 matrix
  - Balanced NCE training across POS types
  - Metropolis gate for grammaticality enforcement
  - LEGD: P(c) proportional to P_base(c) * exp(-alpha * E_norm(c))

No torch. No neural nets. No float32 in the hot path. ~20 MB memory.
"""

from .vocabulary import Vocabulary, COARSE_POS, POS2IDX, IDX2POS, N_POS
from .bigram_model import BigramModel
from .feature_hash_energy import (
    FeatureHashEnergyTable,
    FeatureSpec,
    default_features,
    # Concrete feature classes — for custom feature sets
    LexBigramFeature,
    WordPosBigramFeature,
    PosWordBigramFeature,
    PosBigramFeature,
    LexSkipFeature,
    WordPosSkipFeature,
    PosWordSkipFeature,
    PosSkipFeature,
    PosTrigramFeature,
    LexTrigramFeature,
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
    "WordPosBigramFeature",
    "PosWordBigramFeature",
    "PosBigramFeature",
    "LexSkipFeature",
    "WordPosSkipFeature",
    "PosWordSkipFeature",
    "PosSkipFeature",
    "PosTrigramFeature",
    "LexTrigramFeature",
    "IntegerBoltzmannSampler",
    "IntegerLM",
]
