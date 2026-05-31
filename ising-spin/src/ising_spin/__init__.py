"""
Integer Language Model — pure integer, no neural nets, runs on a Pi 5.

A 100% interpretable, integer-only language model that produces
grammatically coherent text for simple domains using:
  - Bigram counting for base probabilities
  - POS energy rules for syntactic generalization (balanced NCE training)
  - Lexical hash tables for token-specific knowledge
  - Skip-gram patterns for structural dependencies
  - Metropolis gate for grammaticality enforcement
  - LEGD: P(c) ∝ P_base(c) × exp(-α × E_norm(c))

No torch. No neural nets. No float32 in the hot path. ~20 MB memory.
"""

from .vocabulary import Vocabulary, COARSE_POS, POS2IDX, IDX2POS, N_POS
from .bigram_model import BigramModel
from .feature_hash_energy import FeatureHashEnergyTable
from .boltzmann import IntegerBoltzmannSampler
from .integer_lm import IntegerLM

__all__ = [
    "Vocabulary",
    "COARSE_POS", "POS2IDX", "IDX2POS", "N_POS",
    "BigramModel",
    "FeatureHashEnergyTable",
    "IntegerBoltzmannSampler",
    "IntegerLM",
]
