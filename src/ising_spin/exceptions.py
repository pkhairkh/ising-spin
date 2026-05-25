"""
Custom exception hierarchy for the Attractor Language Machine.

All domain-specific exceptions inherit from AttractorError, enabling
catch-all handling at the package level while still allowing fine-grained
per-domain catch blocks.
"""


class AttractorError(Exception):
    """Base exception for all ising_spin errors."""


# ── Build / Training Errors ──────────────────────────────────────────────

class BuildError(AttractorError):
    """Raised when a model component fails to build."""


class VocabularyBuildError(BuildError):
    """Raised when vocabulary construction fails."""


class CorpusError(BuildError):
    """Raised when corpus loading or tokenization fails."""


class TopicBuildError(BuildError):
    """Raised when topic assignment fails to build."""


# ── Runtime / Inference Errors ────────────────────────────────────────────

class InferenceError(AttractorError):
    """Raised during text generation or perplexity computation."""


class SamplingError(InferenceError):
    """Raised when Boltzmann sampling fails (e.g., all weights zero)."""


class EnergyError(InferenceError):
    """Raised when energy computation encounters invalid state."""


# ── Validation Errors ────────────────────────────────────────────────────

class ValidationError(AttractorError):
    """Raised when input validation fails."""


class VocabularyError(ValidationError):
    """Raised for vocabulary-related validation failures."""


class POSValidationError(ValidationError):
    """Raised for POS tag validation failures."""


class ConfigError(ValidationError):
    """Raised when configuration parameters are invalid."""
