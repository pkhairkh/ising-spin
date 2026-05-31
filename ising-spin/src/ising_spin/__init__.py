"""
Integer Language Model v90 — pure integer, no neural nets, runs on a Pi 5.

A 100% interpretable, integer-only language model that produces
grammatically coherent text for simple domains using:
  - Bigram counting with Jelinek-Mercer interpolation (v90 fix — was Laplace alpha=1.0)
  - DYNAMIC feature registry: add/remove features at runtime
  - MULTI-CLASS word system: POS + frequency buckets + distributional clusters
  - Features declare which class system they use via class_key
  - Per-feature class-balanced NCE training (v90 fix — aligned negatives)
  - INDEPENDENT HASH FUNCTIONS (v90 fix — double-hashing, 65%→3% collision correlation)
  - LARGER TABLE SIZES (v90 fix — 262147 for lexical, was 65537)
  - PER-FEATURE Z-SCORE NORMALIZATION
  - PPL-BASED CALIBRATION with expanded alpha range [0, 5.0]
  - NO METROPOLIS GATE (v90 fix — was killing 25% of correct candidates)
  - top_k=200 for evaluation (v90 fix — was 50, 15-25% of tokens invisible)
  - LEGD: P(c) proportional to P_base(c) * exp(-alpha * E(c))

No torch. No neural nets. No float32 in the hot path. ~30 MB memory.
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
