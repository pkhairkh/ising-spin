"""
Ising Spin Glass Language Model — v18.1

Multi-Scale Abstract Recall + Dense AM + VSA Binding + Evolving Document State.

Architecture:
    - Word-level n-gram recall (5-gram) — exact word context
    - POS-level n-gram recall (10-gram) — abstract syntactic generalization
    - Topic-level n-gram recall (10-gram) — discourse coherence
    - Dense AM (v18.1 NEW) — nonlinear pattern matching with random features
    - VSA qFHRR binding (v18.0) — compositional word+POS+topic encoding
    - Document state (7 evolving integer variables) — full-document context
    - ADDITIVE energy fusion — all scales reinforce each other
    - Integer-only Boltzmann sampler (ZERO float ops in hot loop)

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

v17.1 Bug Fixes (carried forward):
    - DocumentState.build() now receives idx2word → state update rules fire
      correctly. Previously, word_str was always None because POSTypeSystem
      didn't have idx2word, causing all state vars (mode, tense, etc.) to
      stay at defaults → compatibility tables were useless.
    - Generator diagnostics now track POS/topic recall hits and state energy.
    - POS n-gram max_n reduced from 15 to 10 (13GB → ~6GB on 1M corpus).
    - Energy combination switched from PoE (min) to additive — all scales
      now contribute to the final energy, reinforcing each other.
    - POS/topic recall scales halved (400/200) for additive combination.
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
from .model_v17 import IsingLMModel

__version__ = "18.1.0"
