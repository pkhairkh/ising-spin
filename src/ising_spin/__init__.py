"""
Ising Spin Language Model — V12 Coherent Generation

Integer-only text generation using Ising spin models with exact token recall.

Architecture (V12 — Coherent Autoregressive + Exact Recall):
    - Autoregressive generation: P(w_t | w_1,...,w_{t-1}) at each position
    - Exact n-gram recall: GFST-HMB-inspired exact token storage and retrieval
    - Kneser-Ney backoff: continuation counts for graceful fallback
    - Interpolation: adaptive recall/PMI/unigram weighting
    - Type-compatible recall: grammar constraints override recall suggestions
    - Function-word anti-loop: max 2 consecutive closed-class words
    - Copy-fade: smooth transitions between copied and generated segments
    - Copy loop detection: prevent infinite phrase repetition
    - ALL generation-path computation is INTEGER ARITHMETIC ONLY

Pipeline:
    V8 training infrastructure → V12 coherent generation
    (PMI, types, emissions, deps, NMF → autoregressive + exact recall)

References:
    - Reinhart & De las Coves (arXiv:2208.08301): Grammar of the Ising Model
    - Marcolli et al. (arXiv:1508.00504): Spin Glass Models of Syntax
    - Haydarov et al. (arXiv:2502.12014): Coupled Ising-Potts Model
    - GFST-HMB (github.com/pkhairkh/gfst-hmb-public): Exact token recall
"""

__version__ = "12.0.0"
