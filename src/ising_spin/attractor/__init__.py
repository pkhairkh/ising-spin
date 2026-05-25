"""
Attractor Language Machine — Dense Associative Memory as the ENGINE.

The attractor dynamics of a properly trained Dense Associative Memory
ARE a language model. Not an approximation. Not a component. They ARE one.

Architecture:
  - SDR encoding: Sparse Distributed Representations (~2% active bits)
  - DAM layers: Dense Associative Memory with F-lookup energy (exponential capacity)
  - Hierarchy: L0-Lexical → L1-Syntactic → L2-Semantic → L3-Discourse
  - RG flow: Wilsonian renormalization group between layers (UV-complete)
  - Episodic memory: Content-addressable sparse pattern storage

Key physics:
  - Energy: E = -Σ F(J_ij * s_i * s_j) - Σ h_i * s_i
  - F-lookup: nonlinear energy function → exponential storage capacity
  - PCD learning: ΔJ = η(data_correlations - model_correlations)
  - UV completeness: couplings renormalizable at all hierarchical scales
  - RG beta functions: g_{l+1} = β(g_l) govern inter-layer flow

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
