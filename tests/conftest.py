"""Pytest configuration for Integer Language Model tests."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "ising-spin" / "src"))


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
] * 3


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
def small_model(small_vocab):
    """Build a small IntegerLM for integration tests."""
    from ising_spin import IntegerLM
    model = IntegerLM(
        vocab=small_vocab,
        n_pos_hashes=1,
        pos_table_size=101,
        n_lex_hashes=1,
        lex_table_size=1009,
        use_skip=False,
        use_trigram=False,
        top_k=20,
        seed=42,
    )
    sequences = small_vocab.tokenize(_SYNTHETIC_TEXTS)
    model.train(sequences, n_epochs=1, n_negatives=2)
    model.calibrate(sequences[:10])
    return model
