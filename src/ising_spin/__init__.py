"""
Ising Spin Glass Language Model — v18.3

Multi-Scale Abstract Recall + Dense AM + VSA Binding + Integer ESN Reservoir
+ Cross-Scale RFF + Factorial State Coupling + Evolving Document State.

Architecture:
    - Word-level n-gram recall (5-gram) — exact word context
    - POS-level n-gram recall (10-gram) — abstract syntactic generalization
    - Topic-level n-gram recall (10-gram) — discourse coherence
    - Dense AM (v18.1) — nonlinear pattern matching with random features
    - VSA qFHRR binding (v18.0) — compositional word+POS+topic encoding
    - Integer ESN Reservoir (v18.2) — long-range temporal dynamics (~50 token lookback)
    - Cross-Scale RFF (v18.3 NEW) — joint word+POS+topic random Fourier features
    - Factorial State Coupling (v18.2) — mean-field inference + coupling energy
    - Document state (7 evolving integer variables) — full-document context
    - ADDITIVE energy fusion — all scales reinforce each other
    - Integer-only Boltzmann sampler (ZERO float ops in hot loop)

v18.3 Changes:
    - Added Cross-Scale RFF (E_rff energy term)
      Combines word+POS+topic into joint random Fourier features
      Captures cross-scale interactions that independent per-scale terms miss
      Pre-aggregated Theta matrix: (V, D) int8

v18.2 Changes:
    - Added Integer ESN Reservoir (E_reservoir energy term)
      Fixed random recurrent network with exponential decay (~50 token lookback)
      Pre-aggregated readout R: (V, 512) int16
    - Added Factorial State Coupling with mean-field inference
      5 pairwise compatibility tables for state variable correlations
      E_coupling energy term penalizes unlikely state combinations
    - Generator now tracks reservoir state per-token

v18.1 Changes:
    - Added Dense Associative Memory module with polynomial nonlinearity
    - E_dense_am energy term creates sharper energy basins (capacity ~N)
    - Random feature pre-aggregation: O(D) per candidate instead of O(N*D)
    - Degree parameter: degree=1 (linear) or degree=2 (Dense AM, sharper)
    - Pre-aggregated Phi matrix: (V, 256) int16, ~25 MB for V=49K

v18.0 Changes:
    - Added VSA qFHRR binding module for compositional token encoding
    - E_vsa_bind energy term captures word+POS+topic interactions
    - State scale rebalanced from 50 to 400 for meaningful contribution
    - VSA energy scale default = 800 (comparable to POS recall)
"""

from .vocabulary import Vocabulary, POSTypeSystem, TopicAssigner
from .vocabulary.pos import (
    COARSE_POS_TAGS, POS2IDX, IDX2POS, N_POS,
    NOUN_LIKE, VERB_LIKE, OPEN_CLASS, CLOSED_CLASS,
)
from .sampling import IntegerBoltzmannSampler, LN2_NUM, LN2_DEN, LOG2_SCALE
from .recall import (
    WordNgramIndex, PosNgramIndex, TopicNgramIndex, MultiScaleRecall,
)
from .state import DocumentState
from .energy import EnergyComputer
from .vsa import QFHRRVectors, VSAEncoder
from .dense_am import RandomFeatureProjector, DenseAMEnergy
from .reservoir import IntegerESN
from .rff import CrossScaleRFF
from .model_v17 import IsingLMModel

__version__ = "18.3.0"
