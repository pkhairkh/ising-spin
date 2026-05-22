"""
Ising Spin Glass Language Model — v11.7

Integer-only Boltzmann sampling with recall-primary energy model.

Architecture:
    - Recall-primary: E = log2(1/P) * recall_scale
    - PMI backoff for unseen n-grams
    - Integer-only Boltzmann sampler (ZERO float ops in hot loop)
    - FineWeb-Edu training corpus

PPL progression:
    v8.1 (50K, 5K vocab, recall-only): PPL = 124
    v9.0 (50K, fine-grained log2):     PPL = 98
    v10.0 (KN backoff experiments):     PPL = 79
    v11.0 (PMI backoff, 2K vocab):     PPL = 52
    v11.7 (200K, 2K vocab, PMI=5):     PPL = 51.54

Entry points:
    run.py       — Full train + eval + generate (best config)
    eval.py      — Standalone PPL evaluation with β sweep
    generate.py  — Text generation
    cache_200k.py — Download and cache FineWeb-Edu data
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
    LN2_NUM,
    LN2_DEN,
    LOG2_SCALE,
)

__version__ = "11.7.0"
