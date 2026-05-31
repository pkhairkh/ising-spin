"""
Integer-only Boltzmann sampling.

Pre-computes a lookup table at initialization using integer geometric
recurrence (NO math.exp). At generation time, sampling is pure integer:
    1. deltas = energies - E_min (non-negative integers)
    2. weights = table[deltas] (integer array lookup)
    3. Cumulative sum (integer addition)
    4. Binary search (integer comparison)

No floating-point operations in the sampling hot path.
"""

import numpy as np


class IntegerBoltzmannSampler:
    """
    Boltzmann sampling using ONLY integer arithmetic.

    Pre-computes a lookup table via integer geometric recurrence:
        table[0] = scale
        table[d] = table[d-1] * decay >> PRECISION
    where decay is computed via integer Taylor expansion of exp(-beta).
    """

    _FP_BITS = 48  # Fixed-point precision for table construction

    def __init__(self, beta: float = 0.1, max_delta: int = 5000, scale: int = 1 << 30):
        if beta <= 0:
            beta = 0.001
        if max_delta <= 0:
            max_delta = 1000
        if scale <= 0:
            scale = 1 << 30

        self.beta = beta
        self.scale = scale
        fine_max = min(max_delta, 50000)
        self.table = np.zeros(fine_max + 1, dtype=np.int64)

        # INTEGER-ONLY TABLE CONSTRUCTION
        P = self._FP_BITS
        ONE = 1 << P
        beta_fp = int(round(beta * ONE))

        # Taylor expansion of exp(-beta) in fixed-point integer
        decay = ONE
        decay -= beta_fp
        beta_sq = (beta_fp * beta_fp) >> P
        decay += beta_sq >> 1
        beta_cube = (beta_sq * beta_fp) >> P
        decay -= beta_cube // 3
        beta_4 = (beta_cube * beta_fp) >> P
        decay += beta_4 // 24
        beta_5 = (beta_4 * beta_fp) >> P
        decay -= beta_5 // 120
        decay = max(0, decay)

        self.table[0] = scale
        prev = int(scale)
        for d in range(1, fine_max + 1):
            prev = (prev * decay) >> P
            if prev <= 0:
                self.table[d:] = 0
                break
            self.table[d] = int(prev)

        self.max_delta = fine_max

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
