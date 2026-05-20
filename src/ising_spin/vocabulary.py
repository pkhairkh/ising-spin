"""
Vocabulary management for the Ising Spin Language Model.

Builds a word-level vocabulary from a corpus using pure integer
frequency counting. No floating-point operations.
"""

from collections import Counter
from typing import List, Optional
import json


class Vocabulary:
    """
    Integer-only vocabulary mapping between words and indices.

    Special tokens:
        <UNK> — unknown / out-of-vocabulary words
        <BOS> — beginning of sequence
        <EOS> — end of sequence
        <PAD> — padding (for fixed-length sequences)
    """

    UNK = "<UNK>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    PAD = "<PAD>"

    SPECIALS = [UNK, BOS, EOS, PAD]

    def __init__(self, min_freq: int = 5, max_size: Optional[int] = None):
        """
        Args:
            min_freq: Minimum corpus frequency for a word to be included.
            max_size: Maximum vocabulary size (excluding specials). None = unlimited.
        """
        self.min_freq = min_freq
        self.max_size = max_size
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()  # integer counts only
        self._built = False

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple whitespace + punctuation tokenizer.
        Splits on whitespace, separates trailing punctuation.
        No FP operations — pure string manipulation.
        """
        tokens = []
        for word in text.split():
            # Separate trailing punctuation
            stripped = word.strip()
            if not stripped:
                continue

            # Split off leading punctuation
            while stripped and not stripped[0].isalnum():
                tokens.append(stripped[0])
                stripped = stripped[1:]

            # Split off trailing punctuation
            tail = []
            while stripped and not stripped[-1].isalnum():
                tail.append(stripped[-1])
                stripped = stripped[:-1]

            if stripped:
                tokens.append(stripped.lower())  # lowercase for compression
            tokens.extend(reversed(tail))

        return tokens

    def build(self, texts: List[str]) -> "Vocabulary":
        """
        Build vocabulary from a list of text strings.
        Pure integer counting — no floating-point.

        Args:
            texts: List of raw text strings.

        Returns:
            self (for chaining)
        """
        # Count all words (integer counting only)
        for text in texts:
            tokens = self._tokenize(text)
            self.word_counts.update(tokens)

        # Build mapping: specials first, then by frequency
        idx = 0
        for special in self.SPECIALS:
            self.word2idx[special] = idx
            self.idx2word[idx] = special
            idx += 1

        # Sort by count (descending), filter by min_freq
        filtered = [
            (word, count)
            for word, count in self.word_counts.most_common()
            if count >= self.min_freq and word not in self.SPECIALS
        ]

        if self.max_size is not None:
            filtered = filtered[: self.max_size]

        for word, count in filtered:
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            idx += 1

        self._built = True
        return self

    def encode(self, text: str) -> List[int]:
        """Encode a text string to a list of integer token indices."""
        tokens = self._tokenize(text)
        unk_idx = self.word2idx[self.UNK]
        return [self.word2idx.get(t, unk_idx) for t in tokens]

    def decode(self, indices: List[int]) -> str:
        """Decode a list of integer token indices to a text string."""
        unk_idx = self.word2idx[self.UNK]
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, self.UNK)
            # Don't print special tokens (except in debug)
            if word in (self.BOS, self.EOS, self.PAD):
                continue
            words.append(word)
        return " ".join(words)

    def __len__(self) -> int:
        return len(self.word2idx)

    def save(self, path: str):
        """Save vocabulary to JSON (integer data only)."""
        data = {
            "min_freq": self.min_freq,
            "max_size": self.max_size,
            "word2idx": self.word2idx,
            "word_counts": {w: int(c) for w, c in self.word_counts.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """Load vocabulary from JSON."""
        with open(path) as f:
            data = json.load(f)
        vocab = cls(min_freq=data["min_freq"], max_size=data["max_size"])
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(v): k for k, v in vocab.word2idx.items()}
        vocab.word_counts = Counter(data["word_counts"])
        vocab._built = True
        return vocab
