"""
Integer Language Model v89 — pure integer, no neural nets, runs on a Pi 5.

A 100% interpretable, integer-only language model that produces
grammatically coherent text for simple domains using:
  - Bigram counting for base probabilities
  - DYNAMIC feature registry: add/remove features at runtime
  - MULTI-CLASS word system: frequency buckets + distributional clusters
  - Features declare which class system they use via class_key
  - Per-feature class-balanced NCE training
  - PER-FEATURE Z-SCORE NORMALIZATION (v89 fix — rescales each feature
    independently so lex_bi doesn't drown out cls_tri_freq)
  - PPL-BASED CALIBRATION (v89 fix — minimizes perplexity instead of
    maximizing argmax accuracy, preventing energy from dominating P_base)
  - Metropolis gate for grammaticality enforcement
  - Soft exponential decay repetition penalty
  - LEGD: P(c) proportional to P_base(c) * exp(-alpha * E(c))

v89 FUNDAMENTAL FIXES over v88 (PPL 27.35 — barely better than base 27.74):
  - Per-feature z-score normalization: each feature's energy is standardized
    to (E_f - mu_f) / sigma_f before weighting, so features contribute
    equally by default (lex_bi no longer drowns out everything else)
  - PPL-based calibration: search alpha minimizing PPL instead of maximizing
    argmax accuracy (argmax selects alpha=2.0 which dominates P_base)

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
