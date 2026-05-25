"""Pytest configuration — add src/ to Python path and define shared fixtures."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Shared synthetic texts ──────────────────────────────────────────────────

_SYNTHETIC_TEXTS = [
    "the cat sat on the mat and the dog ran in the park",
    "she went to the store to buy some food for dinner",
    "the children played in the garden while the sun was shining",
    "he read a book about the history of science and technology",
    "they built a small house near the lake in the forest",
    "the students studied hard for their final exams at school",
    "she cooked a delicious meal for her family on sunday",
    "the weather was warm and sunny during the summer months",
    "he walked along the beach and watched the waves roll in",
    "the city was busy with people going to work each day",
    "the old man told stories about his adventures at sea",
    "a young girl found a beautiful shell on the sandy shore",
    "the team worked together to finish the project on time",
    "music played softly in the background as they danced slowly",
    "the scientist discovered a new species in the deep ocean",
] * 3


# ── Basic fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def sample_texts():
    """Small corpus for testing."""
    return _SYNTHETIC_TEXTS


@pytest.fixture
def small_vocab():
    """Pre-built vocabulary from synthetic texts."""
    from ising_spin import Vocabulary
    v = Vocabulary(min_freq=1, max_size=200)
    v.build(_SYNTHETIC_TEXTS)
    return v


@pytest.fixture
def small_attractor_model():
    """Build a small AttractorLanguageModel for integration tests."""
    from ising_spin.attractor import AttractorLanguageModel

    model = AttractorLanguageModel(
        vocab_min_freq=1,
        vocab_max_size=200,
        sdr_dim=64,
        sdr_sparsity=0.08,
        dam_scale=400,
        grammar_penalty_scale=30,
        same_word_penalty=200,
        max_episodes=100,
        episodic_scale=200,
        max_seq_len=15,
        seed=42,
    )
    model.train(n_samples=len(_SYNTHETIC_TEXTS), texts=_SYNTHETIC_TEXTS)
    return model
