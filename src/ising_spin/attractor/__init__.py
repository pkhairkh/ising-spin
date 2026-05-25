"""
Attractor Language Machine v27 — Dense Associative Memory as the ENGINE.

The attractor dynamics of a properly trained Dense Associative Memory
ARE a language model. Not an approximation. Not a component. They ARE one.

Architecture:
  - SDR encoding: Sparse Distributed Representations (~2% active bits)
  - DAM layers: Dense Associative Memory with F-lookup energy (exponential capacity)
  - Hierarchy: L0-Lexical → L1-Syntactic → L2-Semantic → L3-Discourse
  - RG flow: Coupling-space Wilsonian RG (decimation of J matrices)
  - UV completeness: Cutoff independence + coupling flow stability
  - Episodic memory: Content-addressable sparse pattern storage

Key physics (v27 fixes based on knowledge base):
  - Energy: E = -Σ F(J_ij * s_i * s_j) with NONLINEAR F (exponential capacity)
  - Hebbian learning: RG fixed point at right sparsity (Agliari 2025)
  - Coupling-space RG: J_eff[l+1] = Decimate(J[l]), NOT state-space block-spin
  - UV completeness: cutoff independence + operator spectrum stability
  - Anomalous dimensions: from operator spectrum of J, NOT running correlations

All integer arithmetic. Zero floats in the hot path. Runs on Pi 5.
"""

from .sdr import SDREncoder
from .dam import DAMLayer
from .hierarchy import HierarchicalDAM
from .episodic import EpisodicMemory
from .engine import AttractorLanguageModel

__all__ = [
    "SDREncoder",
    "DAMLayer",
    "HierarchicalDAM",
    "EpisodicMemory",
    "AttractorLanguageModel",
]
