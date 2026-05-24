"""
Backward-compatible re-exports from exceptions.py.

This module previously defined its own exception hierarchy. All exceptions
have been consolidated into exceptions.py. This file re-exports them so
that existing ``from ising_spin.errors import ...`` statements continue
to work.
"""

from .exceptions import (  # noqa: F401
    IsingError,
    BuildError,
    VocabularyBuildError,
    CorpusError,
    IndexBuildError,
    PreAggregationError,
    StateBuildError,
    TopicBuildError,
    InferenceError,
    SamplingError,
    EnergyError,
    StateError,
    ValidationError,
    VocabularyError,
    POSValidationError,
    StateValidationError,
    ConfigError,
    ConfigurationError,
)
