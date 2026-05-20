"""
Ising Spin Glass Language Model — v8.0 Recall-Primary Architecture

ALL word selection goes through the Hamiltonian. No overrides, no bypasses.
Every word is chosen by Boltzmann sampling from the energy landscape.

v8.0 Key Insight: Recall energy E = log₂(1/P) * scale IS the correct
Boltzmann energy. With β ≈ 0.5*ln(2)/recall_scale, the Boltzmann
distribution recovers the n-gram probabilities EXACTLY:
    P(w) ~ exp(-β * E_recall(w)) = P(w)^0.5

This gives PPL ≈ 125 on recall-only — the best result. All other layers
must be SMALL perturbations (≤10% of recall_scale) to avoid disrupting
the recall signal. Graded couplings are DISABLED by default because they
are REDUNDANT with recall (both encode n-gram continuation info).

Architecture (6 layers, RECALL is PRIMARY):
    Layer 1: PMI couplings J[w,w'] + local field h[w] (legacy fallback)
    Layer 1b: Graded Couplings (DISABLED by default in v8.0 — redundant with recall)
    Layer 2: Knowledge external field h_knowledge[w] (≤10% of recall_scale)
    Layer 3: 3-Spin couplings J3[(s,p)] -> o (≤10% of recall_scale)
    Layer 4: Category couplings J_category (≤5% of recall_scale)
    Layer 5: Markov logic penalty (≤5% of recall_scale)

Scale hierarchy (recall-primary mode, default ON):
    recall_scale     = 800       [PRIMARY]
    knowledge_scale  = 80        [10% of recall]
    spin3_scale      = 80        [10% of recall]
    category_scale   = 40        [5% of recall]
    logic_rule_scale = 40        [5% of recall]
    graded_couplings = DISABLED  [redundant with recall]

Generation:
    - All layers compete through integer energy function E(w|ctx)
    - Boltzmann sampling: P(w) ~ exp(-beta * E(w))
    - MCMC spin-flip refinement (Metropolis criterion)
    - No overrides. Knowledge creates small perturbative energy wells.

β auto-calibration from RECALL-ONLY energies (v8.0):
    - Theoretical optimal: β = 0.5 * ln(2) / recall_scale
    - Validated against observed recall energy distribution
    - No longer calibrated from graded couplings

INTEGER-ONLY CONSTRAINT (enforced):
    - ALL generation-path computation uses integer arithmetic
    - Boltzmann sampling via pre-computed lookup table (NO np.exp in hot loop)
    - MCMC acceptance via the same lookup table (integer-only)

References:
    - Levy & Goldberg (2014): Word2Vec as log-PMI matrix factorization
    - Marcoli et al. (arXiv:1508.00504): Spin Glass Models of Syntax
    - Haydarov et al. (arXiv:2502.12014): Coupled Ising-Potts Model
    - Creutz (1983): Demon algorithm for integer MCMC acceptance
    - Nishimori (2001): Statistical Physics of Spin Glasses
"""

from .model import (
    Vocabulary,
    POSTypeSystem,
    IntegerBoltzmannSampler,
    NGramIndex,
    IsingLM,
    IsingLMModel,
    KnowledgeLayer,
    CategoryLayer,
    MarkovLogicLayer,
    WalshSpectralLayer,
    GradedCouplings,
    compute_log_floor_pmi,
    compute_pmi_couplings,
    compute_skip_pmi_couplings,
    fetch_conceptnet_triples,
    COARSE_POS_TAGS,
    POS2IDX,
    IDX2POS,
    N_POS,
    load_fineweb_edu,
    tokenize_texts,
    truncate_sequences,
)

__version__ = "8.0.0"
