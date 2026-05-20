"""
Ising Spin Language Model — V14 Clean Rebuild

Integer-only text generation using Ising PMI couplings with exact n-gram recall.

Architecture (V14 — Honest):
    - N-gram recall: PRIMARY next-word signal
    - Ising PMI coupling: SECONDARY signal when recall misses
    - POS grammar: HARD CONSTRAINTS on word types
    - Integer Boltzmann sampling: lookup-table, NO np.exp in hot loop
    - Ablation framework: measure Ising contribution

Key differences from V8-V13:
    - Genuinely integer-only hot path (lookup-table Boltzmann)
    - ~650 lines vs 7000+ (clean single-file architecture)
    - 6 generation parameters vs 30+ (no ad-hoc patching)
    - No dead MCMC code (V8's 126KB sampler is removed)
    - Honest naming (no pretending MCMC does generation)
    - Built-in ablation to measure Ising vs n-gram contribution

References:
    - Levy & Goldberg (2014): Word2Vec as log-PMI matrix factorization
    - Marcolli et al. (arXiv:1508.00504): Spin Glass Models of Syntax
    - Haydarov et al. (arXiv:2502.12014): Coupled Ising-Potts Model
    - Creutz (1983): Demon algorithm for integer MCMC acceptance
"""

__version__ = "14.0.0"
