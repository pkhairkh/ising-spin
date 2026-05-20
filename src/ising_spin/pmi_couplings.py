"""
Integer-only PMI coupling computation for the Ising Spin Language Model.

Replaces raw co-occurrence counts with log-floor PMI (Pointwise Mutual Information)
computed entirely via integer arithmetic and bit operations.

Key insight: PMI(x,y) = log2(P(x,y) / (P(x)*P(y)))
           ≈ floor(log2(C(x,y)*N / (marginal_x * marginal_y)))
           = bit_length(ratio) - 1

This is the "log-floor PMI" — a purely integer approximation that:
  - Preserves sign (positive/negative association)
  - Preserves ordering (monotonic with true PMI)
  - Provides natural sparsity (J=0 when |PMI| < 1)
  - Is symmetric (unlike conditional probabilities)
  - Uses only integer multiply, divide, comparison, and bit_length

References:
  - Levy & Goldberg (2014): Word2Vec ≈ SVD of PMI matrix
  - Novel: bit_length() as floor(log2()) for integer PMI (no prior work found)
"""

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple
import json
import numpy as np


class PMICouplings:
    """
    Stores integer log-floor PMI couplings for the Ising spin language model.

    Energy function (all integers):
        E(x_1, ..., x_n) = -sum_{i<j} J_PMI[x_i, x_j] * decay(|i-j|)
                          - sum_i h[i][x_i]

    where J_PMI is the log-floor PMI matrix and h is the unigram field.
    """

    def __init__(self, vocab_size: int, seq_len: int, window: int = 10):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.window = window

        # PMI coupling matrix: J_PMI[w, w'] = integer log-floor PMI
        self.J_PMI = np.zeros((vocab_size, vocab_size), dtype=np.int64)

        # Distance-weighted PMI couplings (sparse, per distance)
        self.J_by_dist: Dict[int, Dict[Tuple[int, int], int]] = {}

        # Local fields from unigram PMI
        self.h = np.zeros((seq_len, vocab_size), dtype=np.int64)

        # Hebbian memory coupling (sentence-level co-occurrence)
        self.J_Hebb = np.zeros((vocab_size, vocab_size), dtype=np.int64)

        # Raw counts for diagnostics
        self.bigram_counts: Optional[np.ndarray] = None
        self.unigram_counts: Optional[np.ndarray] = None
        self.total_tokens: int = 0

    @staticmethod
    def _log_floor_pmi(
        cooc: int, marginal_i: int, marginal_j: int, total: int
    ) -> int:
        """
        Compute log-floor PMI using only integer arithmetic and bit operations.

        PMI(i,j) = log2(C(i,j)*N / (C(i)*C(j)))

        Integer approximation:
          numerator = C(i,j) * N
          denominator = C(i) * C(j)
          sign = +1 if numerator > denominator, else -1
          ratio = max(num, denom) // min(num, denom)
          J = sign * (ratio.bit_length() - 1)

        When ratio < 2, J = 0 (natural sparsity threshold).
        When ratio >= 2, J = floor(log2(ratio)) which approximates PMI.

        Returns 0 when either marginal is 0 (no data).
        """
        if cooc == 0 or marginal_i == 0 or marginal_j == 0 or total == 0:
            return 0

        # Use Python's arbitrary precision integers to avoid overflow
        num = int(cooc) * int(total)
        denom = int(marginal_i) * int(marginal_j)

        if num == 0 or denom == 0:
            return 0

        sign = 1 if num > denom else -1
        ratio = max(num, denom) // min(num, denom)

        # bit_length() - 1 = floor(log2(ratio))
        # ratio < 2 → bit_length() - 1 = 0 → natural sparsity
        return sign * (ratio.bit_length() - 1)

    @staticmethod
    def _log_floor_pmi_capped(
        cooc: int, marginal_i: int, marginal_j: int, total: int,
        cap: int = 15
    ) -> int:
        """Log-floor PMI with a maximum magnitude (prevents outlier coupling)."""
        pmi = PMICouplings._log_floor_pmi(cooc, marginal_i, marginal_j, total)
        return max(-cap, min(cap, pmi))

    def compute_from_sequences(
        self,
        sequences: List[List[int]],
        min_count: int = 2,
        pmi_cap: int = 15,
        use_hebbian: bool = True,
        hebbian_weight: int = 1,
    ) -> "PMICouplings":
        """
        Compute PMI couplings from tokenized integer sequences.

        Strategy:
            - Count unigrams, bigrams, and co-occurrences within window
            - Compute log-floor PMI for each (w, w') pair
            - Build distance-weighted sparse coupling dict
            - Optionally compute Hebbian (sentence-level) coupling

        Args:
            sequences: List of integer token sequences.
            min_count: Minimum co-occurrence count for PMI computation.
            pmi_cap: Maximum |PMI| value (prevents outliers).
            use_hebbian: Whether to compute Hebbian sentence-level coupling.
            hebbian_weight: Integer weight for Hebbian term.

        Returns:
            self (for chaining)
        """
        V = self.vocab_size
        window = self.window

        # Step 1: Count unigrams (pure integer)
        unigram = np.zeros(V, dtype=np.int64)
        for seq in sequences:
            for w in seq:
                unigram[w] += 1

        total_tokens = int(unigram.sum())
        self.unigram_counts = unigram.copy()
        self.total_tokens = total_tokens

        # Step 2: Count windowed co-occurrences (pure integer)
        cooc_counts = Counter()  # (w, w') -> count within window
        dist_cooc: Dict[int, Counter] = defaultdict(Counter)

        for seq in sequences:
            for i, w in enumerate(seq):
                for j_offset in range(1, window + 1):
                    j = i + j_offset
                    if j < len(seq):
                        w2 = seq[j]
                        cooc_counts[(w, w2)] += 1
                        dist_cooc[j_offset][(w, w2)] += 1

        # Step 3: Compute log-floor PMI for global coupling matrix
        for (w, w2), count in cooc_counts.items():
            if count >= min_count:
                pmi = self._log_floor_pmi_capped(
                    int(count), int(unigram[w]), int(unigram[w2]),
                    total_tokens, cap=pmi_cap
                )
                self.J_PMI[w, w2] = pmi
                # Symmetrize: also set J[w2, w] (PMI is symmetric)
                self.J_PMI[w2, w] = pmi

        # Step 4: Compute distance-weighted PMI (sparse, per distance)
        for dist, counts in dist_cooc.items():
            self.J_by_dist[dist] = {}
            for (w, w2), count in counts.items():
                if count >= min_count:
                    pmi = self._log_floor_pmi_capped(
                        int(count), int(unigram[w]), int(unigram[w2]),
                        total_tokens, cap=pmi_cap
                    )
                    if pmi != 0:
                        self.J_by_dist[dist][(w, w2)] = pmi

        # Step 5: Compute local fields from unigram frequency
        # h[i][w] = integer-scaled unigram count (position-independent)
        # Use log-floor of frequency ratio as "self-PMI"
        for w in range(V):
            if unigram[w] > 0:
                # Self-information approximation: floor(log2(N/count(w)))
                ratio = total_tokens // int(unigram[w])
                if ratio >= 2:
                    # Higher frequency → lower field (less surprising)
                    self.h[:, w] = ratio.bit_length() - 1
                else:
                    self.h[:, w] = 1

        # Step 6: Compute Hebbian coupling (sentence-level co-occurrence)
        if use_hebbian:
            for seq in sequences:
                # Build set of unique words in this sentence
                words_in_seq = set(seq)
                for w in words_in_seq:
                    for w2 in words_in_seq:
                        if w != w2:
                            self.J_Hebb[w, w2] += hebbian_weight

        # Store bigram counts for diagnostics
        self.bigram_counts = np.zeros((V, V), dtype=np.int64)
        for (w, w2), count in cooc_counts.items():
            self.bigram_counts[w, w2] = count

        return self

    def get_local_energy(
        self, state: List[int], pos: int, word: int
    ) -> int:
        """
        Compute energy contribution from a single word at a single position.
        Pure integer addition.

        E_local = h[pos][word]
                + sum_{j in window} J_PMI[word, state[j]] * decay(|i-j|)
        """
        energy = int(self.h[pos % self.seq_len, word])

        for j_offset in range(1, self.window + 1):
            # Forward neighbor
            j = pos + j_offset
            if j < len(state):
                neighbor = state[j]
                # Use distance-specific PMI if available
                if j_offset in self.J_by_dist:
                    key = (word, neighbor)
                    key_rev = (neighbor, word)
                    if key in self.J_by_dist[j_offset]:
                        energy += self.J_by_dist[j_offset][key]
                    elif key_rev in self.J_by_dist[j_offset]:
                        energy += self.J_by_dist[j_offset][key_rev]
                    else:
                        energy += int(self.J_PMI[word, neighbor])
                else:
                    energy += int(self.J_PMI[word, neighbor])

            # Backward neighbor
            j = pos - j_offset
            if j >= 0:
                neighbor = state[j]
                dist = j_offset
                if dist in self.J_by_dist:
                    key = (neighbor, word)
                    key_rev = (word, neighbor)
                    if key in self.J_by_dist[dist]:
                        energy += self.J_by_dist[dist][key]
                    elif key_rev in self.J_by_dist[dist]:
                        energy += self.J_by_dist[dist][key_rev]
                    else:
                        energy += int(self.J_PMI[neighbor, word])
                else:
                    energy += int(self.J_PMI[neighbor, word])

        return energy

    def get_energy(self, state: List[int]) -> int:
        """Compute total energy of a state. Pure integer addition."""
        energy = 0
        for i, w in enumerate(state):
            if i < self.seq_len:
                energy += int(self.h[i, w])

        for i, w in enumerate(state):
            for j_offset in range(1, self.window + 1):
                j = i + j_offset
                if j < len(state):
                    w2 = state[j]
                    energy += int(self.J_PMI[w, w2])

        return energy

    def get_neighbor_words(self, word: int, top_k: int = 50) -> List[int]:
        """
        Get the top-k most strongly coupled words for a given word.
        Uses absolute PMI value for ranking.
        """
        row = np.abs(self.J_PMI[word])
        top_indices = np.argsort(row)[-top_k:]
        return [int(idx) for idx in top_indices if row[idx] > 0]

    def get_pmi_neighbors(self, word: int, positive_only: bool = True) -> List[Tuple[int, int]]:
        """
        Get (word_idx, pmi_value) pairs for strongly associated words.
        """
        row = self.J_PMI[word]
        pairs = []
        for w2 in range(self.vocab_size):
            val = int(row[w2])
            if positive_only and val > 0:
                pairs.append((w2, val))
            elif not positive_only and val != 0:
                pairs.append((w2, val))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        return pairs

    def combine_with_hebbian(
        self, alpha: int = 3, beta: int = 1
    ) -> np.ndarray:
        """
        Combine PMI and Hebbian couplings: J_total = alpha*J_PMI + beta*J_Hebb.

        Returns combined integer coupling matrix.
        """
        return alpha * self.J_PMI + beta * self.J_Hebb

    def save(self, path: str):
        """Save PMI couplings to disk (integer data only)."""
        np.save(f"{path}_J_PMI.npy", self.J_PMI)
        np.save(f"{path}_h.npy", self.h)
        np.save(f"{path}_J_Hebb.npy", self.J_Hebb)

        if self.bigram_counts is not None:
            np.save(f"{path}_bigram.npy", self.bigram_counts)
        if self.unigram_counts is not None:
            np.save(f"{path}_unigram.npy", self.unigram_counts)

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
            "total_tokens": self.total_tokens,
        }
        with open(f"{path}_meta.json", "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> "PMICouplings":
        """Load PMI couplings from disk."""
        with open(f"{path}_meta.json") as f:
            meta = json.load(f)

        couplings = cls(
            vocab_size=meta["vocab_size"],
            seq_len=meta["seq_len"],
            window=meta["window"],
        )
        couplings.total_tokens = meta["total_tokens"]
        couplings.J_PMI = np.load(f"{path}_J_PMI.npy")
        couplings.h = np.load(f"{path}_h.npy")
        couplings.J_Hebb = np.load(f"{path}_J_Hebb.npy")

        try:
            couplings.bigram_counts = np.load(f"{path}_bigram.npy")
            couplings.unigram_counts = np.load(f"{path}_unigram.npy")
        except FileNotFoundError:
            pass

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
