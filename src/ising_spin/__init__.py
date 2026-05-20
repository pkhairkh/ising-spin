"""
Ising Spin Language Model

Integer-only text generation using Ising PMI couplings with exact n-gram recall.

Architecture:
    - N-gram recall: PRIMARY next-word signal
    - Ising PMI coupling: SECONDARY signal when recall misses
    - POS grammar: HARD CONSTRAINTS on word types
    - Integer Boltzmann sampling: lookup-table, NO np.exp in hot loop
    - Ablation framework: measure Ising contribution

References:
    - Levy & Goldberg (2014): Word2Vec as log-PMI matrix factorization
    - Marcolli et al. (arXiv:1508.00504): Spin Glass Models of Syntax
    - Haydarov et al. (arXiv:2502.12014): Coupled Ising-Potts Model
    - Creutz (1983): Demon algorithm for integer MCMC acceptance
"""

from .model import (
    Vocabulary,
    POSTypeSystem,
    IntegerBoltzmannSampler,
    NGramIndex,
    IsingLM,
    IsingLMModel,
    compute_log_floor_pmi,
    compute_pmi_couplings,
    COARSE_POS_TAGS,
    POS2IDX,
    IDX2POS,
    N_POS,
    load_fineweb_edu,
    tokenize_texts,
    truncate_sequences,
)

__version__ = "1.0.0"
