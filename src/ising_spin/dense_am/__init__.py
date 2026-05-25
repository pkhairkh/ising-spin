"""
Dense Associative Memory module — random feature energy.

Implements Dense AM energy with polynomial nonlinearity and random feature
pre-aggregation. The key insight is that replacing the LINEAR energy
function E = x with a NONLINEAR one E = F(x) where F(x) = x^2
(Dense AM, degree=2) creates much SHARPER energy basins:

  - Correct completions get much lower energy
  - Incorrect completions get much higher energy
  - Capacity increases from ~0.14N (standard Hopfield) to ~N (Dense AM)

Random feature approximation: Instead of storing all N patterns explicitly,
we use random projections to map contexts to a D-dimensional feature space,
then pre-aggregate feature vectors per word. Inference is a single D-dim
dot product per candidate word — O(D) instead of O(N*D).

All arithmetic is integer-only (int8/int16/int32, Q30 energy).
Float operations are used ONLY during module initialization (cos LUT build).
"""

from .energy import RandomFeatureProjector, DenseAMEnergy

__all__ = ["RandomFeatureProjector", "DenseAMEnergy"]
