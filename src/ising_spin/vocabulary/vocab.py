"""
Integer-only vocabulary mapping between words and indices.

Special tokens:
    <UNK>=0, <BOS>=1, <EOS>=2, <PAD>=3, <S>=4

v17.4: <S> (sentence boundary) inserted after '.', '!', '?' to prevent
       cross-sentence n-gram contamination.

Path 3a: Enhanced tokenizer handles contractions, hyphens, and numbers.
"""

import re
from collections import Counter
from typing import Dict, List, Optional


class Vocabulary:
    """
    Integer-only vocabulary mapping between words and indices.

    Special tokens:
        <UNK>=0, <BOS>=1, <EOS>=2, <PAD>=3

    Path 3a: Enhanced tokenizer handles contractions, hyphens, and numbers.
    """

    UNK = "<UNK>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    PAD = "<PAD>"
    SENT = "<S>"  # v17.4: Sentence boundary marker — prevents cross-sentence n-grams
    SPECIALS = [UNK, BOS, EOS, PAD, SENT]

    # Contraction suffixes to split off
    CONTRACTION_SUFFIXES = [
        "n't", "'t",  # negation: don't -> do + n't, can't -> ca + n't
        "'s",         # possessive/aux: it's, he's
        "'re",        # they're
        "'ve",        # they've
        "'ll",        # they'll
        "'d",         # they'd
        "'m",         # I'm
    ]

    def __init__(self, min_freq: int = 5, max_size: Optional[int] = None):
        self.min_freq = min_freq
        self.max_size = max_size
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_counts: Counter = Counter()
        self._built = False

    # Sentence-ending punctuation — v17.4: insert <S> after these
    SENTENCE_ENDS = {'.', '!', '?'}

    def _tokenize(self, text: str) -> List[str]:
        """
        Enhanced tokenizer with better handling of contractions, hyphens,
        and numbers. Pure string manipulation — no external dependencies.

        Path 3a improvements:
          - Contractions: "don't" -> "do" + "n't", "it's" -> "it" + "'s"
          - Hyphens: "well-known" -> "well-known" (kept as one token)
          - Numbers: "3.14" stays as one token, "1,000" stays as one token
        v17.4 improvement:
          - Sentence boundaries: insert <S> after '.', '!', '?' to prevent
            cross-sentence n-gram contamination
        """
        tokens = []
        for word in text.split():
            stripped = word.strip()
            if not stripped:
                continue

            # Split off leading punctuation
            leading_punct = []
            while stripped and not stripped[0].isalnum() and stripped[0] != '-':
                leading_punct.append(stripped[0])
                stripped = stripped[1:]

            # Split off trailing punctuation
            trailing_punct = []
            while stripped and not stripped[-1].isalnum() and stripped[-1] != '-':
                trailing_punct.append(stripped[-1])
                stripped = stripped[:-1]

            # Add leading punctuation tokens
            tokens.extend(leading_punct)

            # v17.4: Track if trailing punct includes sentence end
            has_sentence_end = any(p in self.SENTENCE_ENDS for p in trailing_punct)

            if not stripped:
                tokens.extend(reversed(trailing_punct))
                continue

            lower = stripped.lower()

            # === Handle contractions ===
            contraction_found = False
            for suffix in self.CONTRACTION_SUFFIXES:
                if lower.endswith(suffix) and len(lower) > len(suffix):
                    stem = lower[:-len(suffix)]
                    if stem and any(c.isalpha() for c in stem):
                        # Special case: "can't" -> "ca" + "n't" (not "can")
                        # But we keep it simple: "don't" -> "do" + "n't"
                        tokens.append(stem)
                        tokens.append(suffix)
                        contraction_found = True
                        break

            if contraction_found:
                tokens.extend(reversed(trailing_punct))
                if has_sentence_end:
                    tokens.append(self.SENT)
                continue

            # === Handle numbers (keep as single token) ===
            # "3.14", "1,000", "0.5" should stay as one token
            cleaned = lower.replace(".", "").replace(",", "")
            if cleaned.replace("-", "").isdigit() and len(lower) > 0:
                tokens.append(lower)
                tokens.extend(reversed(trailing_punct))
                if has_sentence_end:
                    tokens.append(self.SENT)
                continue

            # === Handle hyphenated words (keep as single token) ===
            # "well-known", "state-of-the-art" stay as one token
            if '-' in lower and not lower.startswith('-') and not lower.endswith('-'):
                parts = lower.split('-')
                if all(len(p) >= 1 and (p.isalpha() or p.isdigit()) for p in parts):
                    tokens.append(lower)
                    tokens.extend(reversed(trailing_punct))
                    if has_sentence_end:
                        tokens.append(self.SENT)
                    continue

            # === Default: use the word as-is (lowercased) ===
            tokens.append(lower)
            tokens.extend(reversed(trailing_punct))

            # v17.4: Insert sentence boundary marker after sentence-ending punct
            if has_sentence_end:
                tokens.append(self.SENT)

        return tokens

    def build(self, texts: List[str]) -> "Vocabulary":
        """Build vocabulary from a list of text strings. Pure integer counting."""
        for text in texts:
            tokens = self._tokenize(text)
            self.word_counts.update(tokens)

        idx = 0
        for special in self.SPECIALS:
            self.word2idx[special] = idx
            self.idx2word[idx] = special
            idx += 1

        filtered = [
            (word, count)
            for word, count in self.word_counts.most_common()
            if count >= self.min_freq and word not in self.SPECIALS
            # v12: Filter out pure number tokens — they hurt coherence
            # Numbers like "2012", "3", "27" add noise without helping fluency
            and not word.replace(".", "").replace(",", "").replace("-", "").isdigit()
        ]
        if self.max_size is not None:
            filtered = filtered[:self.max_size]

        for word, count in filtered:
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            idx += 1

        self._built = True
        return self

    def add_words(self, words: List[str]) -> int:
        """
        Add words to vocabulary even if they don't appear in corpus.

        Used for knowledge vocabulary augmentation: words like "bark", "meow",
        "gallop" that appear in knowledge triples but not in the training corpus.
        These words get count=1 and no PMI/n-gram stats, but they DO get
        knowledge layer bonuses (h_knowledge, J3) and POS type assignments.

        Returns the number of words actually added.
        """
        if not self._built:
            return 0

        n_added = 0
        for word in words:
            w = word.lower().strip()
            if not w or w in self.SPECIALS:
                continue
            if w in self.word2idx:
                continue  # Already in vocabulary
            if self.max_size is not None and len(self.word2idx) >= self.max_size + len(self.SPECIALS) + 200:
                # Leave room for more additions; don't exceed by too much
                pass  # Allow overflow for knowledge words

            idx = len(self.word2idx)
            self.word2idx[w] = idx
            self.idx2word[idx] = w
            self.word_counts[w] = 1  # Minimal count
            n_added += 1

        return n_added

    def encode(self, text: str) -> List[int]:
        """Encode a text string to a list of integer token indices."""
        tokens = self._tokenize(text)
        unk_idx = self.word2idx[self.UNK]
        return [self.word2idx.get(t, unk_idx) for t in tokens]

    def decode(self, indices: List[int]) -> str:
        """Decode a list of integer token indices to a text string."""
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, self.UNK)
            if word in (self.BOS, self.EOS, self.PAD, self.SENT):
                # v17.4: Also suppress SENT in decoded output
                continue
            words.append(word)
        return " ".join(words)

    def __len__(self) -> int:
        return len(self.word2idx)
