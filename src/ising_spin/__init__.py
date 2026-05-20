"""
Ising Spin Glass Language Model — v6.0 Walsh-Hadamard Spectral Couplings

ALL word selection goes through the Hamiltonian. No overrides, no bypasses.
Every word is chosen by Boltzmann sampling from the energy landscape.

Architecture (6 layers, ALL compete through E(w|ctx)):
    Layer 1: PMI couplings J[w,w'] + local field h[w]
    Layer 1b: Walsh-Hadamard Spectral Couplings (replaces PMI when enabled)
              — Householder subspace rotation V→d for efficiency
              — Order-1 (ĥ₁): graded context-target (replaces PMI)
              — Order-2 (ĥ₂): pairwise context interaction
              — Order-3 (ĥ₃): triple context interaction
    Layer 2: Knowledge external field h_knowledge[w] (SPO triples)
    Layer 3: 3-Spin couplings J3[(s,p)] -> o (many-body Ising interaction)
    Layer 4: Category couplings J_category (hypernym-based semantic smoothing)
    Layer 5: Markov logic penalty (factual consistency, soft + hard)

Generation:
    - All layers compete through integer energy function E(w|ctx)
    - Boltzmann sampling: P(w) ~ exp(-beta * E(w))
    - MCMC spin-flip refinement (Metropolis criterion)
    - No overrides. Knowledge creates competing energy wells.

Key principle: When (dog, barks)->bark and (dog, chases)->chase both fire,
they create COMPETING energy wells. Boltzmann at temperature beta picks
between them stochastically. Near the phase transition, knowledge has
maximum influence with some thermal noise.

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

__version__ = "6.0.0"
