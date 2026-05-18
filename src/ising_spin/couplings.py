"""
Integer coupling computation for the Ising Spin Language Model.

Computes J (pairwise) and h (local field) couplings from corpus
statistics using ONLY integer counting and addition.
No floating-point operations in this module.
"""

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple
import json
import numpy as np


class IsingCouplings:
    """
    Stores integer couplings for an Ising spin language model.

    Energy function (all integers):
        E(x_1, ..., x_n) = -sum_{i<j} J[i,j][x_i, x_j] - sum_i h[i][x_i]

    where J and h are pure integer matrices.

    For scalability, we use:
    - Position-independent global coupling matrix J_global (V x V, int64)
    - Sparse position-specific couplings as dict of (w, w') -> int
    - Local fields h as (seq_len x V, int64)

    Temperature is handled via integer threshold tables (precomputed),
    avoiding exp() and softmax() entirely.
    """

    def __init__(self, vocab_size: int, seq_len: int, window: int = 5):
        """
        Args:
            vocab_size: Number of tokens in vocabulary.
            seq_len: Maximum sequence length for generation.
            window: Context window for pairwise couplings.
        """
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.window = window

        # Local fields: h[i][w] = integer field strength for word w at position i
        self.h = np.zeros((seq_len, vocab_size), dtype=np.int64)

        # Position-independent global coupling: J_global[w, w'] = integer
        # This is the primary coupling used for generation (factorized)
        self.J_global = np.zeros((vocab_size, vocab_size), dtype=np.int64)

        # Position-specific corrections (sparse): J_specific[(i,j)][(w,w')] = int
        # Only stores non-zero corrections to J_global
        self.J_specific: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = {}

        # Temperature as integer inverse: beta_int = round(beta * 1000)
        self.beta_int = 1000  # default beta=1.0

    def compute_from_sequences(
        self,
        sequences: List[List[int]],
        min_count: int = 1,
        scaling: int = 1,
    ) -> "IsingCouplings":
        """
        Compute couplings from tokenized integer sequences.

        Strategy:
            - h[i][w] = count of word w at position i (integer)
            - J_global[w,w'] = count of (w, w') co-occurrence within window (integer)
            - Position-specific J only for positions with significant deviations

        Args:
            sequences: List of integer token sequences.
            min_count: Minimum co-occurrence count to store a coupling.
            scaling: Multiply all counts by this integer.

        Returns:
            self (for chaining)
        """
        # Compute local fields h[i][w] — pure integer counting
        pos_counts = Counter()  # (position, word) -> count
        global_counts = Counter()  # word -> count

        for seq in sequences:
            for i, w in enumerate(seq):
                if i < self.seq_len:
                    pos_counts[(i, w)] += 1
                global_counts[w] += 1

        for (i, w), count in pos_counts.items():
            self.h[i, w] = count * scaling

        # Fill positions with no data using global counts at reduced weight
        for w, count in global_counts.items():
            for i in range(self.seq_len):
                if self.h[i, w] == 0:
                    self.h[i, w] = count // 20  # integer division

        # Compute position-independent global pairwise couplings
        bigram_counts = Counter()  # (w, w') -> count
        for seq in sequences:
            for i, w in enumerate(seq):
                for j_offset in range(1, self.window + 1):
                    j = i + j_offset
                    if j < len(seq):
                        w2 = seq[j]
                        bigram_counts[(w, w2)] += 1

        for (w, w2), count in bigram_counts.items():
            if count >= min_count:
                self.J_global[w, w2] = count * scaling

        # Compute distance-weighted couplings (closer = stronger)
        # This gives more structure than flat J_global
        dist_counts: Dict[int, Counter] = defaultdict(Counter)
        for seq in sequences:
            for i, w in enumerate(seq):
                for j_offset in range(1, self.window + 1):
                    j = i + j_offset
                    if j < len(seq):
                        w2 = seq[j]
                        dist_counts[j_offset][(w, w2)] += 1

        # Store distance-specific couplings as sparse dicts
        self.J_by_dist: Dict[int, Dict[Tuple[int, int], int]] = {}
        for dist, counts in dist_counts.items():
            self.J_by_dist[dist] = {}
            for (w, w2), count in counts.items():
                if count >= min_count:
                    self.J_by_dist[dist][(w, w2)] = count * scaling

        return self

    def get_local_energy(
        self, state: List[int], pos: int, word: int
    ) -> int:
        """
        Compute energy contribution from a single word at a single position.
        Pure integer addition.

        E_local = h[pos][word] + sum_{j in window} coupling(word, state[j])
        """
        energy = int(self.h[pos % self.seq_len, word])

        # Forward neighbors
        for j_offset in range(1, self.window + 1):
            j = pos + j_offset
            if j < len(state):
                neighbor = state[j]
                # Use distance-specific coupling if available
                if j_offset in self.J_by_dist:
                    key = (word, neighbor)
                    if key in self.J_by_dist[j_offset]:
                        energy += self.J_by_dist[j_offset][key]
                    else:
                        energy += int(self.J_global[word, neighbor])
                else:
                    energy += int(self.J_global[word, neighbor])

        # Backward neighbors
        for j_offset in range(1, self.window + 1):
            j = pos - j_offset
            if j >= 0:
                neighbor = state[j]
                dist = j_offset
                if dist in self.J_by_dist:
                    key = (neighbor, word)
                    if key in self.J_by_dist[dist]:
                        energy += self.J_by_dist[dist][key]
                    else:
                        energy += int(self.J_global[neighbor, word])
                else:
                    energy += int(self.J_global[neighbor, word])

        return energy

    def get_energy(self, state: List[int]) -> int:
        """
        Compute total energy of a state. Pure integer addition.
        """
        energy = 0
        for i, w in enumerate(state):
            if i < self.seq_len:
                energy += int(self.h[i, w])

        for i, w in enumerate(state):
            for j_offset in range(1, self.window + 1):
                j = i + j_offset
                if j < len(state):
                    w2 = state[j]
                    if j_offset in self.J_by_dist:
                        key = (w, w2)
                        if key in self.J_by_dist[j_offset]:
                            energy += self.J_by_dist[j_offset][key]
                        else:
                            energy += int(self.J_global[w, w2])
                    else:
                        energy += int(self.J_global[w, w2])

        return energy

    def get_neighbor_words(self, word: int, top_k: int = 50) -> List[int]:
        """
        Get the top-k most strongly coupled words for a given word.
        Used to limit the proposal set in Gibbs sampling.
        Pure integer comparison.
        """
        row = self.J_global[word]
        # Get indices of top-k values (integer sorting)
        top_indices = np.argsort(row)[-top_k:]
        return [int(idx) for idx in top_indices if row[idx] > 0]

    def save(self, path: str):
        """Save couplings to disk (integer data only)."""
        np.save(f"{path}_h.npy", self.h)
        np.save(f"{path}_J_global.npy", self.J_global)

        # Save distance-specific couplings (sparse)
        j_by_dist_ser = {}
        for dist, couplings in self.J_by_dist.items():
            j_by_dist_ser[str(dist)] = {
                f"{w},{w2}": c for (w, w2), c in couplings.items()
            }
        with open(f"{path}_J_by_dist.json", "w") as f:
            json.dump(j_by_dist_ser, f)

        meta = {
            "vocab_size": self.vocab_size,
            "seq_len": self.seq_len,
            "window": self.window,
            "beta_int": self.beta_int,
        }
        with open(f"{path}_meta.json", "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> "IsingCouplings":
        """Load couplings from disk."""
        with open(f"{path}_meta.json") as f:
            meta = json.load(f)

        couplings = cls(
            vocab_size=meta["vocab_size"],
            seq_len=meta["seq_len"],
            window=meta["window"],
        )
        couplings.beta_int = meta["beta_int"]
        couplings.h = np.load(f"{path}_h.npy")
        couplings.J_global = np.load(f"{path}_J_global.npy")

        with open(f"{path}_J_by_dist.json") as f:
            j_by_dist_ser = json.load(f)
        couplings.J_by_dist = {}
        for dist_str, couplings_dict in j_by_dist_ser.items():
            dist = int(dist_str)
            couplings.J_by_dist[dist] = {}
            for key_str, count in couplings_dict.items():
                w, w2 = map(int, key_str.split(","))
                couplings.J_by_dist[dist][(w, w2)] = count

        return couplings
