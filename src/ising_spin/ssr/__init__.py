"""
Semantic Spin Resonance (SSR) — Emergent understanding via frustrated spin dynamics.

This module implements genuine long-range dependency through a hierarchical
Ising spin glass with ADAPTIVE couplings. Unlike all previous modules which
are either hand-coded (entity/phase/scene trackers), linear (ESN, MTR, RFF),
or static (all pre-aggregated readout matrices), SSR creates understanding
as an EMERGENT PROPERTY of the physics.

Key innovations:
  1. Binary spin dynamics with FRUSTRATION (competing constraints)
  2. Hebbian EPISODIC MEMORY that adapts during generation
  3. Genuinely NONLINEAR dynamics (sign function, not linear recurrence)
  4. FEEDBACK LOOP: word selection depends on sigma, sigma depends on history

The 256-dimensional binary spin vector sigma encodes the document's current
semantic state as a DISTRIBUTED representation. The spin configuration
evolves through mean-field dynamics influenced by:
  - J_struct: Structural couplings (generic frustrated landscape, fixed)
  - J_episodic: Episodic memory (document-specific, Hebbian during generation)
  - W_word: External field from the current word selection

The Hebbian episodic memory is the KEY innovation. When "dragon" appears,
it activates a specific spin pattern. The Hebbian update strengthens couplings
within this pattern, creating an ATTRACTOR. Later, when the text should
reference the dragon again, the J_episodic matrix helps sigma fall back
into the dragon-attractor configuration, creating GENUINE long-range recall.

This is NOT a transformer (no attention), NOT mamba (no state-space model),
NOT an ESN (nonlinear discrete dynamics + adaptive couplings). This is
PURE SPIN GLASS PHYSICS applied to language.

All arithmetic is integer-only.
"""

from .semantic_spin import SemanticSpinResonance

__all__ = ["SemanticSpinResonance"]
