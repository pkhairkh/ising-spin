"""
Vector Symbolic Architecture (VSA) module — qFHRR binding.

Implements Quantized Fourier Holographic Reduced Representations (qFHRR),
a compositional VSA via modular addition in the phase domain.

Key insight: Instead of treating word, POS, and topic as INDEPENDENT
energy signals (v17 additive combination), qFHRR BINDS them into a
single compositional code. This captures interactions like:
  "bank" + VERB + FINANCE  !=  "bank" + NOUN + RIVER
which the additive model cannot distinguish.

All arithmetic is integer-only (uint8 phases, int32 similarity, Q30 energy).
"""

from .qfhrr import QFHRRVectors, VSAEncoder

__all__ = ["QFHRRVectors", "VSAEncoder"]
