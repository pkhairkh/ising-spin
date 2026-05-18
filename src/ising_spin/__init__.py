"""
Ising Spin Language Model — Zero Floating-Point Text Generation

A proof-of-concept demonstrating that grammatically and semantically
structured text can be generated using only integer arithmetic, with
no floating-point operations in the generation loop.

Architecture (v3 — Enhanced Typed Ising-Potts):
    - Each position has a TYPE (POS tag, ~13 states) and VALUE (word, ~8K states)
    - Type layer: Potts model over POS tag sequences (spaCy-accurate)
    - Value layer: Ising-like model with PMI couplings (NMF-factorized)
    - Coupling structure: Ising-Potts gating (Haydarov et al., arXiv:2502.12014)
    - PMI couplings: log-floor integer approximation via bit_length()
    - Grammar: integer quadratic penalties (Marcolli-style implicational couplings)
    - Dependency couplings: J_tree from parse trees for long-range agreement
    - Integer NMF: J ≈ W×H, scaling memory from O(V²) to O(V×K)
    - Semantics: type compatibility matrix gating Hebbian coupling
    - Generation: staged annealing (types → types+words → words)
    - ALL generation-path computation is integer arithmetic only

New in v3:
    - SpaCy POS tagger replaces rule-based assignment (accurate type couplings)
    - Dependency tree couplings (J_tree) for subject-verb agreement
    - Integer matrix factorization (J ≈ W×H) for vocabulary scaling beyond 3K
    - Larger corpus training (100K samples default) to densify PMI matrix

References:
    - Reinhart & De las Coves (arXiv:2208.08301): Grammar of the Ising Model
    - Marcolli et al. (arXiv:1508.00504): Spin Glass Models of Syntax
    - Haydarov et al. (arXiv:2502.12014): Coupled Ising-Potts Model
    - Levy & Goldberg (2014): Word2Vec ≈ SVD of PMI matrix
    - Lee & Seung (2001): Algorithms for Non-negative Matrix Factorization
"""

__version__ = "5.0.0"
