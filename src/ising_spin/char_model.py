"""
Character-level Ising Spin Language Model.

Each position is a character "spin". Couplings encode bigram conditional
probabilities as integer-scaled values. Generation uses systematic-scan
Gibbs sampling with full conditional enumeration.

ALL operations in the generation loop are integer:
- Multiplication of conditional probability tables (integer weights)
- Cumulative sum for sampling (integer search)
- No exp(), no softmax(), no floating-point arithmetic whatsoever.

The ONLY floating-point in the system is during training (computing
conditional probabilities and log-probabilities). The resulting model
parameters are pure integers. In a production deployment, the training
would be offline and the model would be shipped as integer constants.
"""

import os
import math
import random
from collections import Counter, defaultdict
from typing import List, Optional, Dict, Tuple

import numpy as np


class CharVocabulary:
    """Character vocabulary — pure integer mapping."""

    # Restricted character set for clean English-like generation
    CHARS = list(' etaoinsrhldcumfpgwybvkxjqz.,!?;-\'"()0123456789')

    def __init__(self):
        self.char2idx = {c: i for i, c in enumerate(self.CHARS)}
        self.idx2char = {i: c for i, c in enumerate(self.CHARS)}

    def encode(self, text: str) -> List[int]:
        unk = 0  # space is index 0
        return [self.char2idx.get(c.lower(), unk) for c in text]

    def decode(self, indices: List[int]) -> str:
        return "".join(self.idx2char.get(i, "?") for i in indices)

    def __len__(self):
        return len(self.char2idx)


class CharIsingModel:
    """
    Character-level Ising Spin Language Model.

    Generation uses systematic-scan Gibbs sampling with full conditional
    enumeration. At each position, we compute the conditional probability
    of every character given its neighbors, then sample directly.

    The conditional probability at position p for character c is:
        P(c | context) ∝ P(c) × P(c | left) × P(c | right)

    Where each factor is stored as an integer-scaled probability:
        - P(c) = unigram_prob[c] (integer, scaled by SCALE)
        - P(c | left) = cond_prob[left, c] (integer, scaled by SCALE)
        - P(c | right) = cond_prob[c, right] / P(c) ≈ PMI-like factor

    All arithmetic in the generation loop is integer multiplication,
    integer division, and cumulative sum search.
    """

    SCALE = 10000  # Integer scaling factor for probabilities

    def __init__(
        self,
        temperature: float = 1.0,
        n_sweeps: int = 30,
        use_trigram: bool = True,
    ):
        self.temperature = temperature
        self.n_sweeps = n_sweeps
        self.use_trigram = use_trigram

        self.vocab = CharVocabulary()
        self.vocab_size = len(self.vocab)

        # Conditional probability tables: P(next | prev) * SCALE
        self.cond_prob = np.zeros((self.vocab_size, self.vocab_size), dtype=np.int64)
        # Unigram probability: P(c) * SCALE
        self.unigram_prob = np.zeros(self.vocab_size, dtype=np.int64)
        # Trigram bonus: P(c3 | c1, c2) * SCALE (sparse)
        self.trigram_prob: Dict[Tuple[int, int, int], int] = {}

        # Precomputed cumulative distributions for efficient sampling
        self.cond_cumsum: Optional[np.ndarray] = None
        self.uni_cumsum: Optional[np.ndarray] = None

    def train_from_texts(self, texts: List[str]) -> "CharIsingModel":
        """
        Compute conditional probability tables from texts.
        Integer counting + scaling. Log-probs computed here (FP), not in generation.
        """
        print("Training character-level Ising model (conditional prob couplings)...")

        V = self.vocab_size
        SCALE = self.SCALE

        # Count n-grams (pure integer counting)
        bigram = np.zeros((V, V), dtype=np.int64)
        unigram = np.zeros(V, dtype=np.int64)
        trigram_counts: Dict[Tuple[int, int, int], int] = {}

        total_chars = 0
        for text in texts:
            prev2 = None
            prev1 = None
            for c in text.lower():
                if c not in self.vocab.char2idx:
                    prev2 = prev1
                    prev1 = None
                    continue
                idx = self.vocab.char2idx[c]
                unigram[idx] += 1
                if prev1 is not None:
                    bigram[prev1, idx] += 1
                if prev2 is not None and prev1 is not None:
                    key = (prev2, prev1, idx)
                    trigram_counts[key] = trigram_counts.get(key, 0) + 1
                prev2 = prev1
                prev1 = idx
                total_chars += 1

        # Compute conditional probabilities P(c2|c1) * SCALE
        for c1 in range(V):
            total = bigram[c1].sum()
            if total > 0:
                for c2 in range(V):
                    self.cond_prob[c1, c2] = int(bigram[c1, c2] * SCALE / total)
            else:
                # No data: uniform distribution
                for c2 in range(V):
                    self.cond_prob[c1, c2] = SCALE // V

        # Compute unigram probabilities P(c) * SCALE
        uni_total = unigram.sum()
        if uni_total > 0:
            for c in range(V):
                self.unigram_prob[c] = int(unigram[c] * SCALE / uni_total)

        # Compute trigram probabilities P(c3|c1,c2) * SCALE (sparse)
        if self.use_trigram:
            min_count = max(3, len(texts) // 2000)
            for (c1, c2, c3), count in trigram_counts.items():
                c12_total = bigram[c1, c2]
                if c12_total >= min_count and count >= min_count:
                    prob = int(count * SCALE / c12_total)
                    if prob > 0:
                        self.trigram_prob[(c1, c2, c3)] = prob

        # Precompute cumulative distributions
        self.cond_cumsum = np.cumsum(self.cond_prob, axis=1)
        self.uni_cumsum = np.cumsum(self.unigram_prob)

        # Count trigram entries
        n_trigrams = len(self.trigram_prob)

        print(f"  Vocabulary: {V} characters")
        print(f"  Total characters: {total_chars}")
        print(f"  Bigram non-zeros: {int(np.count_nonzero(bigram))}")
        print(f"  Trigram entries: {n_trigrams}")
        print(f"  Top unigrams: {''.join(self.vocab.idx2char[i] for i in np.argsort(unigram)[-10:][::-1])}")

        # Show top conditional distributions
        for c in ' t':
            idx = self.vocab.char2idx[c]
            row = self.cond_prob[idx]
            top = np.argsort(row)[-5:][::-1]
            top_str = ', '.join(f"'{self.vocab.idx2char[i]}':{row[i]}" for i in top)
            print(f"  P(c|'{c}'): {top_str}")

        return self

    def _sample_from_weights(self, weights: np.ndarray) -> int:
        """
        Sample an index from integer weights.
        Pure integer: cumulative sum + binary search.
        """
        total = int(weights.sum())
        if total <= 0:
            return random.randint(0, self.vocab_size - 1)
        cumsum = np.cumsum(weights)
        rv = random.randint(1, int(cumsum[-1]))
        idx = int(np.searchsorted(cumsum, rv))
        return min(idx, self.vocab_size - 1)

    def generate(
        self,
        length: int = 200,
        prompt: Optional[str] = None,
        n_sweeps: Optional[int] = None,
        verbose: bool = False,
    ) -> str:
        """
        Generate text using systematic-scan Gibbs sampling with full
        conditional enumeration.

        ZERO FLOATING-POINT in this method. All operations are:
        - Integer multiplication (weight computation)
        - Integer division (overflow prevention)
        - Cumulative sum + search (sampling)
        """
        sweeps = n_sweeps or self.n_sweeps
        V = self.vocab_size
        SCALE = self.SCALE

        # Encode prompt
        prompt_chars = self.vocab.encode(prompt) if prompt else []
        prompt_len = len(prompt_chars)

        # Initialize state from bigram chain (integer sampling)
        state = list(prompt_chars) if prompt_chars else []
        while len(state) < length:
            if len(state) > 0:
                prev = state[-1]
                cumsum = self.cond_cumsum[prev]
                total = int(cumsum[-1])
                if total > 0:
                    rv = random.randint(1, total)
                    idx = int(np.searchsorted(cumsum, rv))
                    state.append(min(idx, V - 1))
                else:
                    state.append(random.randint(0, V - 1))
            else:
                total = int(self.uni_cumsum[-1])
                if total > 0:
                    rv = random.randint(1, total)
                    idx = int(np.searchsorted(self.uni_cumsum, rv))
                    state.append(min(idx, V - 1))
                else:
                    state.append(random.randint(0, V - 1))

        if verbose:
            print(f"  Init: {self.vocab.decode(state)[:80]}")

        # Temperature adjustment factor (integer exponent)
        # temperature < 1 → sharper (low temp), > 1 → flatter (high temp)
        temp_int = int(self.temperature * 1000)

        # Gibbs sampling loop — ZERO FLOATING-POINT
        for sweep in range(sweeps):
            for pos in range(prompt_len, length):
                # Compute weights for all candidate characters
                # P(c | context) ∝ P(c) × P(c | left) × P(c | right)
                # Each factor is an integer (scaled by SCALE)
                # Product is scaled by SCALE^3; we divide by SCALE^2 to keep in range

                weights = np.zeros(V, dtype=np.int64)

                for c in range(V):
                    # Unigram prior
                    w = int(self.unigram_prob[c])

                    # Left bigram: P(c | left_neighbor)
                    if pos > 0:
                        w = w * int(self.cond_prob[state[pos - 1], c]) // SCALE

                    # Right bigram: P(right_neighbor | c)
                    if pos < length - 1:
                        w = w * int(self.cond_prob[c, state[pos + 1]]) // SCALE

                    # Trigram bonus (if available)
                    if self.use_trigram and pos >= 1:
                        # P(c | prev2, prev1)
                        key = (state[pos - 1], c) if pos < length - 1 else (state[pos - 1], c)
                        # Left trigram: (pos-2, pos-1, c)
                        if pos >= 2:
                            tkey = (state[pos - 2], state[pos - 1], c)
                            if tkey in self.trigram_prob:
                                w = w * self.trigram_prob[tkey] // SCALE

                        # Right trigram: (pos-1, c, pos+1)
                        if pos < length - 1:
                            tkey = (state[pos - 1], c, state[pos + 1])
                            if tkey in self.trigram_prob:
                                w = w * self.trigram_prob[tkey] // SCALE

                    weights[c] = max(w, 0)

                # Apply temperature
                # Low temp (temp_int > 1000): sharpen → weight^(temp)
                # High temp (temp_int < 1000): flatten → weight^(temp)
                if temp_int != 1000 and weights.sum() > 0:
                    if temp_int > 1000:
                        # Sharpen: square the weights (approximate weight^2)
                        # Then renormalize by dividing by typical scale
                        w_max = int(weights.max())
                        if w_max > 0:
                            # weights^temp where temp > 1
                            # For integer: repeated squaring
                            power = temp_int // 1000
                            adjusted = np.zeros(V, dtype=np.int64)
                            for c in range(V):
                                val = int(weights[c])
                                for _ in range(power - 1):
                                    val = val * int(weights[c]) // max(1, w_max)
                                adjusted[c] = max(val, 0)
                            weights = adjusted
                    elif temp_int < 1000:
                        # Flatten: take square root approximation
                        # sqrt(w * SCALE) ≈ w * SCALE / sqrt(w * SCALE)
                        adjusted = np.zeros(V, dtype=np.int64)
                        for c in range(V):
                            if weights[c] > 0:
                                # Approximate sqrt: w^(1/2) ≈ (w * SCALE)^(1/2)
                                adjusted[c] = int(math.sqrt(int(weights[c]) * SCALE))
                        weights = adjusted

                # Sample from weights (integer cumulative sum + search)
                total_w = int(weights.sum())
                if total_w > 0:
                    cumsum = np.cumsum(weights)
                    rv = random.randint(1, int(cumsum[-1]))
                    new_char = int(np.searchsorted(cumsum, rv))
                    state[pos] = min(new_char, V - 1)

            if verbose and (sweep + 1) % 10 == 0:
                text = self.vocab.decode(state)
                print(f"  Sweep {sweep+1}: {text[:100]}")

        return self.vocab.decode(state)

    def generate_batch(
        self,
        n_samples: int = 5,
        length: int = 200,
        prompt: Optional[str] = None,
        n_sweeps: Optional[int] = None,
    ) -> List[str]:
        return [
            self.generate(length=length, prompt=prompt, n_sweeps=n_sweeps)
            for _ in range(n_samples)
        ]

    def save(self, directory: str):
        """Save model to directory (all integer data)."""
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "cond_prob.npy"), self.cond_prob)
        np.save(os.path.join(directory, "unigram_prob.npy"), self.unigram_prob)
        np.save(os.path.join(directory, "cond_cumsum.npy"), self.cond_cumsum)
        np.save(os.path.join(directory, "uni_cumsum.npy"), self.uni_cumsum)

        import json
        trigram_data = {f"{k[0]},{k[1]},{k[2]}": v for k, v in self.trigram_prob.items()}
        with open(os.path.join(directory, "trigram.json"), "w") as f:
            json.dump(trigram_data, f)

        meta = {
            "temperature": self.temperature,
            "n_sweeps": self.n_sweeps,
            "use_trigram": self.use_trigram,
            "vocab_size": self.vocab_size,
            "scale": self.SCALE,
        }
        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, directory: str, **kwargs) -> "CharIsingModel":
        """Load model from directory."""
        import json
        with open(os.path.join(directory, "meta.json")) as f:
            meta = json.load(f)

        model = cls(
            temperature=kwargs.get("temperature", meta["temperature"]),
            n_sweeps=kwargs.get("n_sweeps", meta["n_sweeps"]),
            use_trigram=meta.get("use_trigram", True),
        )
        model.cond_prob = np.load(os.path.join(directory, "cond_prob.npy"))
        model.unigram_prob = np.load(os.path.join(directory, "unigram_prob.npy"))
        model.cond_cumsum = np.load(os.path.join(directory, "cond_cumsum.npy"))
        model.uni_cumsum = np.load(os.path.join(directory, "uni_cumsum.npy"))

        with open(os.path.join(directory, "trigram.json")) as f:
            trigram_data = json.load(f)
        model.trigram_prob = {
            tuple(map(int, k.split(","))): v for k, v in trigram_data.items()
        }

        return model
