"""
Learned Latent Spin Glass — GENUINE understanding from physics, not hand-coded rules.

The CORE INSIGHT: In a real spin glass, NOBODY declares what the spins "mean."
The coupling matrix J_ij determines what structure EMERGES. Meaning comes from
the LEARNED interaction structure, not from human-declared features.

This module replaces ALL hand-coded state tracking:
  - DocumentState (7 declared variables: tense, mode, entity, etc.)
  - Macro spins (entity/phase/scene trackers with hand-coded rules)
  - SSR (random W_word and random J_struct)

With a SINGLE unified system where:
  1. Each word has a LEARNED binary spin vector sigma_w (from data, not declared)
  2. The coupling matrix J is LEARNED from training data (not random)
  3. Document state evolves via Ising dynamics with the LEARNED J
  4. Long-range dependencies EMERGE from the learned coupling structure

The spin vectors are learned via context-sign hashing:
  sigma_w[d] = sign(sum of random projections of w's context words)

This means words in similar contexts get similar spin vectors — NATURALLY.
Nobody tells the model about "gender" or "tense" — it DISCOVERS what
dimensions matter from the data.

The coupling matrix is learned via Hopfield storage rule:
  J = sum_mu sigma_mu * sigma_mu^T / N

This captures the ACTUAL dependency structure between spin dimensions.
During generation, J mediates long-range dependencies: if dimension i and j
co-occur in training, J_ij > 0, and the model maintains that correlation
across the entire document.

This is PURE SPIN GLASS PHYSICS. No attention, no state-space model,
no hand-coded rules. Just Ising spins with LEARNED couplings.

All arithmetic is integer-only.
"""

from .latent_spin import LatentSpinGlass

__all__ = ["LatentSpinGlass"]
