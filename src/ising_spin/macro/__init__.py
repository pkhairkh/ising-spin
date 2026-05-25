"""
Macro-Spin Layer — Coupled timescale spin glass for long-range coherence.

In real spin glasses, fast spins (local word choices) equilibrate quickly
while slow spins (document-level properties) persist in metastable states
across arbitrarily long timescales.  The macro-spin layer makes this
hierarchical structure explicit.

Three macro-spin modules:
  1. EntityTracker: Maintains active entity slots with activation/decay.
     Entities persist across the entire document (not exponential decay).
     Pronouns and references refresh entity activation.
  2. NarrativePhaseTracker: Tracks where we are in the narrative arc
     (setting → introduction → rising → climax → resolution).
     Phase transitions are driven by token count + trigger words.
  3. SceneTracker: Maintains active scene/location slots.
     Scene vocabulary persists until explicitly replaced.

Cross-Scale Coupling:
  E_macro(w) = E_entity(w) + E_phase(w) + E_scene(w)

  This energy is ADDITIVE with the existing Hamiltonian:
  E_total(w) = E_recall + E_state + E_coupling + E_reservoir + E_vsa
             + E_macro + E_hard

Key physics: Macro-spins have ENERGY BARRIERS preventing random flipping.
Unlike the reservoir (exponential decay, no barrier), macro-spins persist
indefinitely until explicitly "flipped" by a strong enough input.
This gives correlation length ξ >> 400 tokens.
"""

from .entity_tracker import EntityTracker
from .narrative_phase import NarrativePhaseTracker
from .scene_tracker import SceneTracker
from .macro_coupling import MacroSpinCoupling

__all__ = [
    "EntityTracker",
    "NarrativePhaseTracker",
    "SceneTracker",
    "MacroSpinCoupling",
]
