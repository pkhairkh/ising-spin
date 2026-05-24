"""
Custom exception hierarchy for the Ising Spin Glass Language Model.

All domain-specific exceptions inherit from IsingSpinError, enabling
catch-all handling at the package level while still allowing fine-grained
per-domain catch blocks.

IsingError is kept as a backward-compatible alias for IsingSpinError.

Usage:
    try:
        model.train(texts=texts)
    except BuildError as e:
        logger.error("Build failed: %s", e)
    except IsingSpinError:
        logger.error("Something went wrong in ising_spin")
"""


class IsingSpinError(Exception):
    """Base exception for all ising_spin errors."""


# Backward-compatible alias
IsingError = IsingSpinError


# ── Build / Training Errors ──────────────────────────────────────────────

class BuildError(IsingSpinError):
    """Raised when a model component fails to build."""


class VocabularyBuildError(BuildError):
    """Raised when vocabulary construction fails."""


class CorpusError(BuildError):
    """Raised when corpus loading or tokenization fails."""


class IndexBuildError(BuildError):
    """Raised when an n-gram index fails to build."""


class PreAggregationError(BuildError):
    """Raised when pre-aggregation (Dense AM / RFF / ESN readout) fails."""


class StateBuildError(BuildError):
    """Raised when DocumentState compatibility tables fail to build."""


class TopicBuildError(BuildError):
    """Raised when topic assignment fails to build."""


# ── Runtime / Inference Errors ────────────────────────────────────────────

class InferenceError(IsingSpinError):
    """Raised during text generation or perplexity computation."""


class SamplingError(InferenceError):
    """Raised when Boltzmann sampling fails (e.g., all weights zero)."""


class EnergyError(InferenceError):
    """Raised when energy computation encounters invalid state."""


class StateError(InferenceError):
    """Raised when document state update fails."""


# ── Validation Errors ────────────────────────────────────────────────────

class ValidationError(IsingSpinError):
    """Raised when input validation fails."""


class VocabularyError(ValidationError):
    """Raised for vocabulary-related validation failures."""


class POSValidationError(ValidationError):
    """Raised for POS tag validation failures."""


class StateValidationError(ValidationError):
    """Raised when document state is accessed before build."""


class ConfigError(ValidationError):
    """Raised when configuration parameters are invalid."""


class ConfigurationError(IsingSpinError):
    """Raised when an invalid configuration is detected."""
