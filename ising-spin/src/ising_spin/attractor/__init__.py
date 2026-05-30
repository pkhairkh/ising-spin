"""
Attractor Language Machine v76 — INTEGER EBM RE-RANKER.

v76 ARCHITECTURAL PIVOT:
  The attractor dynamics of a Dense Associative Memory are no longer used
  as a standalone language model. Instead, they serve as a DISCRIMINATOR
  that re-ranks candidates from a frozen neural LM (GPT-2).

  "Base model proposes, discriminator disposes."

  This is the DEXA / NCE approach:
    - GPT-2 provides top-k candidates with log-probabilities
    - DAM + spin + episodic assigns energy to each candidate
    - Combined energy determines the final ranking
    - Boltzmann sampling for stochastic generation

Architecture:
  - Base model: Frozen GPT-2 (or DummyBaseLM for testing without torch)
  - Discriminator: Single DAMLayer with F_EXP_APPROX energy
  - Spin state: ThreeBandState (Z/X/Y magnetizations)
  - Episodic memory: Content-addressable sparse pattern storage
  - Binding context: VSA permutation for word-order information
  - Training: NCE with 4 corruption types (not PCD)
  - Generation: Combined energy re-ranking + Boltzmann sampling

Components:
  - SDREncoder: Sparse Distributed Representations (kWTA, ~2% active bits)
  - DAMLayer: Dense Associative Memory with F-lookup energy
  - HierarchicalDAM: Multi-level DAM (kept for reference, not used in v76)
  - ThreeBandState: Three-band spin hidden state (Z/X/Y)
  - EpisodicMemory: Content-addressable pattern storage
  - BindingContext: VSA permutation binding for word order
  - ReRankerEngine: v76 main engine (replaces AttractorLanguageModel)
  - BaseLMInterface: GPT-2 wrapper (optional torch dependency)
  - Corruptor: NCE corruption generator
  - NCETrainer: NCE Hebbian trainer for DAM discriminator

All integer arithmetic in the DAM hot path. Runs on Pi 5.
"""

from .sdr import SDREncoder
from .dam import DAMLayer
from .hierarchy import HierarchicalDAM
from .episodic import EpisodicMemory
from .engine import AttractorLanguageModel
from .expressivity import ManifoldCapacity
from .binding import BindingContext
from .three_band import ThreeBandState
from .reranker_engine import ReRankerEngine
from .base_model import BaseLMInterface, DummyBaseLM
from .corruptions import Corruptor
from .nce import NCETrainer

__all__ = [
    "SDREncoder",
    "DAMLayer",
    "HierarchicalDAM",
    "EpisodicMemory",
    "AttractorLanguageModel",
    "ManifoldCapacity",
    "BindingContext",
    "ThreeBandState",
    "ReRankerEngine",
    "BaseLMInterface",
    "DummyBaseLM",
    "Corruptor",
    "NCETrainer",
]
