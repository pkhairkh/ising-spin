"""
Attractor Language Machine v39 — ENERGY RESOLUTION + BINDING FIX.

The attractor dynamics of a properly trained Dense Associative Memory
ARE a language model. Not an approximation. Not a component. They ARE one.

Architecture:
  - SDR encoding: Sparse Distributed Representations (~2% active bits)
  - DAM layers: Dense Associative Memory with F-lookup energy (exponential capacity)
  - F function: exp_approx (piecewise integer exponential) — TRUE exponential capacity
  - Hierarchy: L0-Lexical -> L1-Syntactic -> L2-Semantic -> L3-Discourse
  - RG flow: J_eff DERIVED from L0 via coupling-space decimation, REPLACES J at higher levels
  - UV completeness: Ward identities + cutoff independence + coupling flow stability
  - Episodic memory: Content-addressable sparse pattern storage
  - VSA binding: Permutation-based compositional context (v38-v39)

DEEP FIXES (v28):
  1. F_EXP_APPROX: piecewise integer exponential — TRUE exponential capacity
  2. RG-derived J_eff REPLACES J at higher levels (not just diagnostic)
  3. Ward identity UV checks (not just spectral gap / cutoff sensitivity)
  4. Pure Hebbian ONLY (PCD removed — unnecessary at right sparsity)
  5. Anomalous dimensions from operator spectrum of J (not running correlations)
  6. DAM energy alone drives word selection (no n-gram crutch)
  7. D decreasing: 512->256->128->64 (RG reduces DOF at coarser scales)

v39 ENERGY RESOLUTION + BINDING FIX:
  - LOG2_NORM reduced from 4096 to 512 — 8x more energy levels
    v37/v38 used LOG2_NORM=4096, giving only ~5 distinct energy levels
    after integer division. The Boltzmann sampler couldn't discriminate.
    This was the root cause of v35 PPL=461 → v37 PPL=5587 regression.
  - M_bind NOT OR'd into context_sdr for DAM energy — DAM was trained
    on standard context SDRs; injecting binding bits added noise.
    M_bind still used for attractor dynamics (step_all) context field.
  - Multi-step unbinding: unbind with last 3 words, not just last 1.
    Gives richer context beyond bigrams — trigram patterns emerge.
  - Binding formula: direct overlap*weight (no //10 integer truncation).
  - bind_weight=30 (adjusted for new LOG2_NORM=512 energy scale).

All integer arithmetic. Zero floats in the hot path. Runs on Pi 5.
"""

from .sdr import SDREncoder
from .dam import DAMLayer
from .hierarchy import HierarchicalDAM
from .episodic import EpisodicMemory
from .binding import BindingContext
from .engine import AttractorLanguageModel
from .expressivity import ManifoldCapacity

__all__ = [
    "SDREncoder",
    "DAMLayer",
    "HierarchicalDAM",
    "EpisodicMemory",
    "BindingContext",
    "AttractorLanguageModel",
    "ManifoldCapacity",
]
