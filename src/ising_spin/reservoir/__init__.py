"""
Reservoir module — Integer Echo State Network for long-range temporal dynamics.

Implements an integer-only Echo State Network (ESN) that maintains a fading
memory of the entire document history. Unlike n-gram recall which only looks
at the last 5-10 tokens, the ESN's exponential decay provides ~50 token
effective lookback (with spectral radius alpha ≈ 0.95 in Q15).

Key insight: The reservoir state h(t) is a compressed summary of ALL previous
tokens, not just the recent n-gram window. This enables the model to:
  - Track discourse coherence across sentence boundaries
  - Distinguish between "the X the Y" and "the Y the X" (position-sensitive)
  - Maintain topic continuity over 50+ token spans
  - Capture long-range syntactic dependencies (subject-verb agreement)

Architecture:
  - Fixed random input matrix W_in: (reservoir_dim, vocab_size) int8, sparse ternary
  - Reservoir state: h(t) = clip(alpha * h(t-1) + W_in[:, w_t], -2^15, 2^15)
  - Precomputed readout R: (vocab_size, reservoir_dim) int16 via training pre-aggregation
  - Energy: E_reservoir(w) = -(h(t) · R[w]) * reservoir_scale / norm_factor

All arithmetic is integer-only (int8/int16/int32, Q15 alpha, Q30 energy).
Float operations are used ONLY during module initialization.
"""

from .integer_esn import IntegerESN

__all__ = ["IntegerESN"]
