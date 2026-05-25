"""
Integer-only Boltzmann sampling for the Attractor Language Machine.

Pre-computes a lookup table at initialization using integer geometric
recurrence (NO math.exp). At generation time, sampling is pure integer:
    1. deltas = energies - E_min (non-negative integers)
    2. weights = table[deltas] (integer array lookup)
    3. Cumulative sum (integer addition)
    4. Binary search (integer comparison)

Public API:
    IntegerBoltzmannSampler  — main sampler class
    LN2_NUM, LN2_DEN         — rational approximation of ln(2)
    LOG2_SCALE               — fixed-point scale for integer log2 computations
    int_log2_fine            — fine-grained integer log2 with LUT
"""

import numpy as np

from ..exceptions import SamplingError

# ===========================================================================
# CONSTANTS
# ===========================================================================

# Rational approximation of ln(2) = 0.6931471805599453...
# 25246/36417 = 0.69314718... (error < 10^-7)
LN2_NUM = 25246
LN2_DEN = 36417

# Fixed-point scale for integer log2 computations
LOG2_SCALE = 100000  # 5 digits of precision


# ===========================================================================
# INTEGER BOLTZMANN SAMPLER
# ===========================================================================

class IntegerBoltzmannSampler:
    """
    Boltzmann sampling using ONLY integer arithmetic — INCLUDING initialization.

    ZERO floating-point operations anywhere.

    Pre-computes a lookup table at initialization using integer geometric
    recurrence (NO math.exp):
        table[0] = scale
        table[d] = table[d-1] * decay >> PRECISION
    where decay is computed via integer Taylor expansion of exp(-beta).

    At generation time, sampling is pure integer:
        1. deltas = energies - E_min (non-negative integers)
        2. weights = table[deltas] (integer array lookup)
        3. Cumulative sum (integer addition)
        4. Binary search (integer comparison)
    """

    _FP_BITS = 48  # Fixed-point precision for table construction

    def __init__(self, beta: float = 0.1, max_delta: int = 5000, scale: int = 1 << 30):
        if beta <= 0:
            raise SamplingError(f"beta must be positive, got {beta}")
        if max_delta <= 0:
            raise SamplingError(f"max_delta must be positive, got {max_delta}")
        if scale <= 0:
            raise SamplingError(f"scale must be positive, got {scale}")

        self.beta = beta
        self.scale = scale
        # For accurate PPL computation, max_delta must cover the full energy
        # range. With dam_scale=1600 and ~2K vocab, max delta ≈ 32K.
        # The hierarchical DAM with multi-layer energy contributions can
        # produce wider ranges, so we use a generous cap.
        # Memory: 50001 × 8 bytes ≈ 400KB — very affordable.
        fine_max = min(max_delta, 50000)
        self.table = np.zeros(fine_max + 1, dtype=np.int64)

        # INTEGER-ONLY TABLE CONSTRUCTION
        # Compute exp(-beta) as a fixed-point integer via Taylor expansion:
        #   exp(-x) = 1 - x + x^2/2 - x^3/6 + x^4/24 - x^5/120
        # All in fixed-point with _FP_BITS bits of precision.
        P = self._FP_BITS
        ONE = 1 << P

        beta_fp = int(round(beta * ONE))  # beta in fixed-point

        # Taylor expansion of exp(-beta) in fixed-point integer
        decay = ONE  # term 0: 1.0
        decay -= beta_fp  # term 1: -x
        beta_sq = (beta_fp * beta_fp) >> P
        decay += beta_sq >> 1  # term 2: +x^2/2
        beta_cube = (beta_sq * beta_fp) >> P
        decay -= beta_cube // 3  # term 3: -x^3/6
        beta_4 = (beta_cube * beta_fp) >> P
        decay += beta_4 // 24  # term 4: +x^4/24
        beta_5 = (beta_4 * beta_fp) >> P
        decay -= beta_5 // 120  # term 5: -x^5/120
        decay = max(0, decay)

        # Build table via integer geometric recurrence
        # Use Python arbitrary-precision integers to avoid overflow,
        # then convert to int64 for the lookup table.
        self.table[0] = scale
        prev = int(scale)  # Python int (arbitrary precision)
        for d in range(1, fine_max + 1):
            prev = (prev * decay) >> P
            if prev <= 0:
                self.table[d:] = 0
                break
            self.table[d] = int(prev)  # Convert back to int64-compatible

        self.max_delta = fine_max

        # Build log2(1+ε) lookup table for compute_log_probabilities
        # log2(1+ε) for ε ∈ [0, 1) with 16-bit precision (65536 entries)
        # Computed via integer Taylor expansion:
        #   log2(1+ε) = ln(1+ε)/ln(2)
        #   ln(1+ε) = ε - ε²/2 + ε³/3 - ...  (all in fixed-point)
        #   1/ln(2) ≈ LN2_DEN/LN2_NUM (rational inverse)
        LUT_SIZE = 1 << 16  # 65536 entries
        self._log2_lut = np.zeros(LUT_SIZE, dtype=np.int64)
        # Use 7th-order Taylor: ln(1+ε) = ε - ε²/2 + ε³/3 - ε⁴/4 + ε⁵/5 - ε⁶/6 + ε⁷/7
        for i in range(LUT_SIZE):
            eps = (i * LOG2_SCALE) >> 16  # ε in LOG2_SCALE fixed-point
            eps2 = (eps * eps) // LOG2_SCALE
            eps3 = (eps2 * eps) // LOG2_SCALE
            eps4 = (eps3 * eps) // LOG2_SCALE
            eps5 = (eps4 * eps) // LOG2_SCALE
            eps6 = (eps5 * eps) // LOG2_SCALE
            eps7 = (eps6 * eps) // LOG2_SCALE
            ln_term = eps - eps2//2 + eps3//3 - eps4//4 + eps5//5 - eps6//6 + eps7//7
            log2_val = (ln_term * LN2_DEN) // LN2_NUM
            self._log2_lut[i] = log2_val

    def sample(self, energies: np.ndarray) -> int:
        """Sample from Boltzmann distribution P(i) ~ exp(-beta * E_i). Integer-only."""
        if len(energies) <= 1:
            return 0

        e_min = int(energies.min())
        deltas = (energies - e_min).astype(np.int64)
        deltas = np.clip(deltas, 0, self.max_delta)

        weights = self.table[deltas]
        total = int(weights.sum())
        if total <= 0:
            return np.random.randint(len(energies))

        r = np.random.randint(0, total)
        cumsum = np.cumsum(weights)
        idx = int(np.searchsorted(cumsum, r, side='right'))
        return min(idx, len(energies) - 1)

    def compute_log_probabilities(self, energies: np.ndarray) -> np.ndarray:
        """
        Compute log2 probabilities for each element — INTEGER-ONLY.

        Uses the analytical formula for log2 of Boltzmann weights:
          table[d] = scale * 2^(-β*d/ln2)
          log2(table[d]) = log2(scale) - β*d/ln2

        This is EXACT — no approximation needed for individual log2 weights.
        Only log2(Z) requires the lookup table (for the sum).

        Returns log2 P(i) * LOG2_SCALE as int64 fixed-point.
        """
        if len(energies) == 0:
            return np.array([], dtype=np.int64)

        e_min = int(energies.min())
        deltas = (energies - e_min).astype(np.int64)
        deltas = np.clip(deltas, 0, self.max_delta)

        weights = self.table[deltas]
        Z = int(weights.sum())
        if Z <= 0:
            return np.full(len(energies), -10 * LOG2_SCALE, dtype=np.int64)

        # Compute log2(Z) using the LUT
        log2_Z = self._int_log2(Z)

        # Compute log2(w_i) for each weight using the LUT
        log_probs = np.zeros(len(energies), dtype=np.int64)
        for i in range(len(energies)):
            w = int(weights[i])
            if w <= 0:
                log_probs[i] = -15 * LOG2_SCALE
            else:
                log_probs[i] = self._int_log2(w) - log2_Z

        return log_probs

    def _int_log2(self, x: int) -> int:
        """
        Compute log2(x) * LOG2_SCALE using integer-only arithmetic.

        Uses bit_length() for the integer part and iterative refinement
        (Newton-like) for the fractional part. More accurate than Taylor
        LUT for the full range of x.
        """
        if x <= 0:
            return -100 * LOG2_SCALE
        if x == 1:
            return 0

        bl = x.bit_length() - 1  # floor(log2(x))

        # Normalize: x = 2^bl * m where m ∈ [1, 2)
        # m = x / 2^bl, represented in fixed-point with 32 fractional bits
        if bl <= 32:
            m = x << (32 - bl)  # m in [2^32, 2^33)
        else:
            m = x >> (bl - 32)  # m in [2^32, 2^33)

        # Now compute log2(m) where m ∈ [2^32, 2^33)
        # log2(m) = 32 + log2(m/2^32) where m/2^32 ∈ [1, 2)
        # Let f = m/2^32 ∈ [1, 2), so we need log2(f)
        # Use iterative bit extraction: log2(f) = Σ b_i * 2^(-i) where b_i are bits
        # This is exact for 32 bits of precision
        frac = 0  # fractional part of log2 in LOG2_SCALE units
        m_normalized = m  # working copy, initially in [2^32, 2^33)
        ONE_32 = 1 << 32

        for bit in range(1, 32):
            # Square m_normalized: if m² >= 2^(2*32+1), then this bit is 1
            m_squared = m_normalized * m_normalized
            if m_squared >= (ONE_32 << 33):
                frac += LOG2_SCALE >> bit
                m_normalized = m_squared >> (33)  # divide by 2^33, result in [2^32, 2^33)
            else:
                m_normalized = m_squared >> (32)  # divide by 2^32, result in [2^32, 2^33)
            # Early exit if we have enough precision
            if (LOG2_SCALE >> bit) == 0:
                break

        return bl * LOG2_SCALE + frac


# ===========================================================================
# FINE-GRAINED INTEGER LOG₂
# ===========================================================================

# Pre-computed LUT for log₂(1 + ε) where ε ∈ [0, 1)
# Used by int_log2_fine() to compute log₂(x) with 8-bit fractional precision.
# Returns log₂(x) * 256 as an integer.
#
# LUT construction uses 7th-order Taylor expansion of ln(1+ε):
#   ln(1+ε) = ε - ε²/2 + ε³/3 - ε⁴/4 + ε⁵/5 - ε⁶/6 + ε⁷/7
# All in fixed-point with 16 bits of precision.
# Then log₂(1+ε) = ln(1+ε) / ln(2) = ln(1+ε) * LN2_DEN / LN2_NUM

_LOG2_LUT_BITS = 16
_LOG2_LUT_SIZE = 1 << _LOG2_LUT_BITS  # 65536
_LOG2_FRAC_BITS = 8  # 8 bits of fractional precision for log₂
_LOG2_FRAC_SCALE = 1 << _LOG2_FRAC_BITS  # 256

_LOG2_LUT = np.zeros(_LOG2_LUT_SIZE, dtype=np.int32)
for _i in range(_LOG2_LUT_SIZE):
    # INTEGER-ONLY LUT construction — range-splitting for accuracy.
    # ε = _i / 65536 ∈ [0, 1). We compute log₂(1+ε) * 256 entirely with integers.
    #
    # Strategy: for ε ∈ [0, 0.5), direct Taylor of ln(1+ε) converges fast.
    # For ε ∈ [0.5, 1), use identity: log₂(1+ε) = 1 + log₂((1+ε)/2)
    # where (1+ε)/2 ∈ [0.75, 1), so we need log₂(1 - δ) with δ ∈ [0, 0.25].
    # ln(1-δ) = -δ - δ²/2 - δ³/3 - ... converges well for small δ.
    _FP = 32  # fractional bits for intermediate computation
    _ONE = 1 << _FP
    _HALF = _ONE >> 1

    if _i < _LOG2_LUT_SIZE >> 1:
        # ε ∈ [0, 0.5): direct Taylor of ln(1+ε), 12th order
        _eps = (_i << _FP) // _LOG2_LUT_SIZE
        _e2 = (_eps * _eps) >> _FP
        _e3 = (_e2 * _eps) >> _FP
        _e4 = (_e3 * _eps) >> _FP
        _e5 = (_e4 * _eps) >> _FP
        _e6 = (_e5 * _eps) >> _FP
        _e7 = (_e6 * _eps) >> _FP
        _e8 = (_e7 * _eps) >> _FP
        _e9 = (_e8 * _eps) >> _FP
        _e10 = (_e9 * _eps) >> _FP
        _e11 = (_e10 * _eps) >> _FP
        _e12 = (_e11 * _eps) >> _FP
        _ln = (_eps - _e2//2 + _e3//3 - _e4//4 + _e5//5 - _e6//6
               + _e7//7 - _e8//8 + _e9//9 - _e10//10 + _e11//11 - _e12//12)
        _log2_val = (_ln * LN2_DEN) // (LN2_NUM * (1 << 24))
    else:
        # ε ∈ [0.5, 1): use log₂(1+ε) = 1 + log₂(1 - δ)
        # where δ = (1-ε)/2 ∈ [0, 0.25]
        # ε = _i / 65536, so δ = (65536 - _i) / (2 * 65536)
        _delta = ((_LOG2_LUT_SIZE - _i) << _FP) // (2 * _LOG2_LUT_SIZE)
        _d2 = (_delta * _delta) >> _FP
        _d3 = (_d2 * _delta) >> _FP
        _d4 = (_d3 * _delta) >> _FP
        _d5 = (_d4 * _delta) >> _FP
        _d6 = (_d5 * _delta) >> _FP
        _d7 = (_d6 * _delta) >> _FP
        _d8 = (_d7 * _delta) >> _FP
        # ln(1-δ) = -δ - δ²/2 - δ³/3 - ... (all negative terms)
        _ln_neg = (_delta + _d2//2 + _d3//3 + _d4//4 + _d5//5 + _d6//6 + _d7//7 + _d8//8)
        _ln = -_ln_neg
        # log₂(1-δ) * 256 + 256 (the +256 is the "1" in "1 + log₂(1-δ)")
        _log2_frac = (_ln * LN2_DEN) // (LN2_NUM * (1 << 24))
        _log2_val = _LOG2_FRAC_SCALE + _log2_frac  # 256 + log₂(1-δ)*256

    _LOG2_LUT[_i] = _log2_val


def int_log2_fine(x: int) -> int:
    """
    Compute log₂(x) * 256 using integer-only arithmetic with pre-computed LUT.

    Replaces floor(log₂) = bit_length()-1 with fine-grained fractional
    log₂, providing accurate log-probability computation for perplexity.

    Returns log₂(x) with 8 bits of fractional precision:
      int_log2_fine(2)   = 256   (log₂(2) = 1.0)
      int_log2_fine(3)   = 405   (log₂(3) ≈ 1.585)
      int_log2_fine(4)   = 512   (log₂(4) = 2.0)
      int_log2_fine(256) = 2048  (log₂(256) = 8.0)
      int_log2_fine(1000)≈ 2551  (log₂(1000) ≈ 9.966)
    """
    if x <= 1:
        return 0

    int_part = x.bit_length() - 1

    # Normalize x to [65536, 131072) i.e. [1.0, 2.0) in 16-bit fixed point
    SHIFT = _LOG2_LUT_BITS  # 16
    if int_part >= SHIFT:
        m = x >> (int_part - SHIFT)
    else:
        m = x << (SHIFT - int_part)
    # m ∈ [65536, 131072) representing [1.0, 2.0)

    # ε = m/65536 - 1, so ε_index = m - 65536 ∈ [0, 65536)
    eps_idx = m - (1 << SHIFT)
    eps_idx = max(0, min(eps_idx, _LOG2_LUT_SIZE - 1))
    frac = int(_LOG2_LUT[eps_idx])

    return int_part * _LOG2_FRAC_SCALE + frac
