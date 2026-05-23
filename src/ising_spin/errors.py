"""
Custom exceptions for the Ising Spin Glass Language Model.

All domain-specific errors inherit from IsingError so callers can
catch the entire hierarchy or specific subtypes as needed.
"""


class IsingError(Exception):
    """Base exception for all ISG-LM errors."""


# --- Build / Training Errors ---

class BuildError(IsingError):
    """Raised when a model component fails to build."""


class VocabularyError(BuildError):
    """Raised for vocabulary-related failures (empty, too small, etc.)."""


class CorpusError(BuildError):
    """Raised when corpus loading or tokenization fails."""


class IndexBuildError(BuildError):
    """Raised when an n-gram index cannot be built."""


class PreAggregationError(BuildError):
    """Raised when pre-aggregation (Dense AM / RFF / ESN readout) fails."""


# --- Inference Errors ---

class InferenceError(IsingError):
    """Raised during generation or perplexity computation."""


class EnergyError(InferenceError):
    """Raised when energy computation fails."""


class SamplingError(InferenceError):
    """Raised when Boltzmann sampling fails."""


class StateError(InferenceError):
    """Raised when document state update fails."""


# --- Validation Errors ---

class ValidationError(IsingError):
    """Raised when input validation fails."""


class ConfigError(ValidationError):
    """Raised when configuration parameters are invalid."""
