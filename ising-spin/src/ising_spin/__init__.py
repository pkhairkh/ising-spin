"""
Integer Language Model v87 — pure integer, no neural nets, runs on a Pi 5.

A 100% interpretable, integer-only language model that produces
grammatically coherent text for simple domains using:
  - Bigram counting for base probabilities
  - DYNAMIC feature registry: add/remove features at runtime
  - MULTI-CLASS word system: frequency buckets + distributional clusters
  - Features declare which class system they use via class_key
  - Per-feature class-balanced NCE training
  - Per-feature NCE subsampling (nce_rate) to prevent class feature saturation
  - RAW energy combination (per-feature normalization removed in v87)
  - Metropolis gate for grammaticality enforcement
  - Soft exponential decay repetition penalty
  - LEGD: P(c) proportional to P_base(c) * exp(-alpha * E_norm(c))

v87 FIXES over v86 (PPL 14.16 — per-feature normalization was harmful):
  - Removed per-feature z-score normalization: it destroyed the mean
    discriminative signal and over-compressed energy (std=3.2)
  - Raw energies + global z-score only (like v83, which had PPL 13.77)
  - Increased class nce_rate from 0.10 → 0.50 (5x more updates)
  - Class features now get enough signal to be useful

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
