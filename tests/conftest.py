"""Pytest configuration — add src/ to Python path and define shared fixtures."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "src"))


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
def small_model():
    """Build a small IsingLMModel with v18 modules for integration/property tests."""
    from ising_spin.orchestrator import IsingLMModel

    model = IsingLMModel(
        vocab_min_freq=1,
        vocab_max_size=200,
        ngram_max_n=3,
        ngram_min_count=1,
        pos_ngram_max_n=5,
        pos_ngram_min_count=1,
        topic_ngram_max_n=5,
        topic_ngram_min_count=1,
        n_topics=4,
        reservoir_dim=32,
        reservoir_alpha_q15=31130,
        reservoir_scale=800,
        vsa_dim=64,
        vsa_scale=800,
        coupling_scale=200,
        recall_scale=1600,
        pos_recall_scale=800,
        topic_recall_scale=400,
        state_scale=400,
        same_word_penalty=200,
        max_closed_class_run=2,
        auto_calibrate_beta=False,
        beta_word=0.1,
        beta_type=0.01,
        interpolated=True,
        kn_backoff=True,
        max_seq_len=30,
        enable_reservoir=True,
        enable_coupling=True,
        enable_vsa=True,
    )
    model.train(texts=_SYNTHETIC_TEXTS)
    return model
