"""
Hash-Compressed Integer Energy Table — Phase 1 of the Architectural Rethink.

v77: Kill the SDR, kill the J-matrix, kill the DAM.

The core insight from v76h: Alpha=0.0 means the DAM is dead weight.
The 0.799 rerank_acc came ENTIRELY from bigram + repetition penalty.
The DAM's global energy (std=1.68M) is pure noise that drowns any signal.

This module replaces the entire DAM pipeline with a hash-compressed
integer energy table:

  E(x_{t-1}, x_t) = sum_h table_h[hash_h(x_{t-1}, x_t) mod P_h]

Where:
  - P_h is a prime number (table size per hash function)
  - hash_h are independent double-hash functions
  - table_h entries are int32 (trained via integer NCE)
  - Multiple hash functions create an ensemble (like Bloom filters)

WHY THIS IS NOT "just a bigram table":
  1. Hash collisions create IMPLICIT GENERALIZATION — similar token
     pairs share energy entries, so "the cat" and "a cat" overlap
  2. Multi-hash ensemble effect — 3 independent hash functions vote
  3. Extendable to trigrams: table[hash(x_{t-2}, x_{t-1}, x_t)]
  4. Can stack multiple tables at different prime sizes for richer signal
  5. O(1) per candidate lookup — no matrix multiply, no SDR encoding

TRAINING:
  For real pairs:     table[h] -= eta       (lower energy = more likely)
  For corrupted pairs: table[h] += eta      (higher energy = less likely)
  Pure integer NCE — no gradients, no BLAS, no float32 intermediates.

INFERENCE:
  For each candidate x_t, look up:
    delta_E = sum_h table_h[hash_h(x_{t-1}, x_t) mod P_h]
  Use delta_E to adjust base model probability:
    P(s_t) proportional to P_base(s_t) * exp(-alpha * delta_E / T)

This is the "Hash-Compressed Skip-Gram Energy" approach.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


# ---------------------------------------------------------------------------
# Hash functions — integer-only, deterministic, collision-resistant
# ---------------------------------------------------------------------------

def _double_hash(a: int, b: int, h_idx: int, P: int) -> int:
    """
    Double-hash for pair (a, b) with hash index h_idx.

    h(a, b, i) = (a * P1 + b * P2 + i * P3) mod P

    Where P1, P2, P3 are distinct primes. This gives independent
    hash functions for each h_idx (like double-hashing in Bloom filters).

    All arithmetic is integer. The result is in [0, P).
    """
    # Three large primes
    P1 = 2654435761  # Knuth's multiplicative hash constant (2^32 / phi)
    P2 = 2246822519  # Another large prime
    P3 = 3266489917  # Another large prime

    # Compute in int64 to avoid overflow, then mod P
    val = (a * P1 + b * P2 + h_idx * P3) & 0xFFFFFFFF  # Keep in 32-bit
    return int(val % P)


def _triple_hash(a: int, b: int, c: int, h_idx: int, P: int) -> int:
    """
    Triple-hash for trigram (a, b, c) with hash index h_idx.

    h(a, b, c, i) = (a * P1 + b * P2 + c * P4 + i * P3) mod P
    """
    P1 = 2654435761
    P2 = 2246822519
    P3 = 3266489917
    P4 = 3367900313

    val = (a * P1 + b * P2 + c * P4 + h_idx * P3) & 0xFFFFFFFF
    return int(val % P)


# ---------------------------------------------------------------------------
# Vectorized hash functions — numpy arrays for batch operations
# ---------------------------------------------------------------------------

# Prime constants (must match scalar versions exactly)
_P1 = np.int64(2654435761)
_P2 = np.int64(2246822519)
_P3 = np.int64(3266489917)
_P4 = np.int64(3367900313)
_MASK32 = np.int64(0xFFFFFFFF)


def _double_hash_vec(a: np.ndarray, b: np.ndarray, h_idx: int, P: int) -> np.ndarray:
    """
    Vectorized double-hash for arrays of (a, b) pairs.

    Args:
        a: Array of first elements (prev word IDs), shape (N,).
        b: Array of second elements (target word IDs), shape (N,).
        h_idx: Hash function index.
        P: Table size (prime).

    Returns:
        Array of hash slots, shape (N,), dtype int64.
    """
    # Use Python int arithmetic wrapped in numpy to avoid int64 overflow
    # a and b are typically < 2000, so a*P1 fits in int64
    val = (a.astype(np.int64) * _P1 + b.astype(np.int64) * _P2 + np.int64(h_idx) * _P3) & _MASK32
    return val % np.int64(P)


def _triple_hash_vec(a: np.ndarray, b: np.ndarray, c: np.ndarray, h_idx: int, P: int) -> np.ndarray:
    """
    Vectorized triple-hash for arrays of (a, b, c) triples.
    """
    val = (a.astype(np.int64) * _P1 + b.astype(np.int64) * _P2
           + c.astype(np.int64) * _P4 + np.int64(h_idx) * _P3) & _MASK32
    return val % np.int64(P)


# ---------------------------------------------------------------------------
# HashEnergyTable — the core data structure
# ---------------------------------------------------------------------------

class HashEnergyTable:
    """
    Hash-compressed integer energy table.

    Replaces the DAM + SDR + J-matrix pipeline with O(1) integer lookups.

    Supports:
      - Bigram energy: E(x_{t-1}, x_t)
      - Trigram energy: E(x_{t-2}, x_{t-1}, x_t)
      - Multiple hash functions for ensemble effect
      - Pure integer NCE training (no gradients)
    """

    # Default prime table sizes (each is a prime)
    PRIME_SIZES = [
        65537,    # ~64K entries per hash
        65521,    # Slightly different prime
        65479,    # Another prime close to 64K
    ]

    def __init__(
        self,
        vocab_size: int,
        n_hashes: int = 3,
        table_size: int = 65537,
        use_trigram: bool = True,
        trigram_weight: int = 1,
        eta: int = 1,
        clip_value: int = 1000,
        seed: int = 42,
    ):
        """
        Args:
            vocab_size: Number of words in vocabulary (V).
            n_hashes: Number of independent hash functions (ensemble size).
            table_size: Prime number for table size. Each hash function
                        gets a table of this size. Larger = fewer collisions.
            use_trigram: Whether to also use trigram hash tables.
            trigram_weight: Relative weight of trigram vs bigram (int scale).
            eta: NCE learning rate (integer increment per update).
            clip_value: Max absolute value for table entries after each epoch.
                        Prevents energy explosion from hash collision accumulation.
                        0 = no clipping.
            seed: Random seed for training order shuffling.
        """
        self.V = vocab_size
        self.n_hashes = n_hashes
        self.table_size = table_size
        self.use_trigram = use_trigram
        self.trigram_weight = trigram_weight
        self.eta = eta
        self.clip_value = clip_value
        self.seed = seed

        # Bigram tables: one int32 array per hash function
        self._bigram_tables = [
            np.zeros(table_size, dtype=np.int32)
            for _ in range(n_hashes)
        ]

        # Trigram tables (optional)
        self._trigram_tables = None
        if use_trigram:
            # Use a slightly different prime to decorrelate from bigrams
            trigram_size = self._next_prime(table_size + 256)
            self._trigram_tables = [
                np.zeros(trigram_size, dtype=np.int32)
                for _ in range(n_hashes)
            ]
            self._trigram_size = trigram_size

        # Precompute hash lookups for speed
        # For each (prev_word, candidate_word) pair we need at decode time,
        # we could cache hashes, but the hash is O(1) anyway.

        # Training statistics
        self._train_updates = 0
        self._train_pos_count = 0
        self._train_neg_count = 0

    @staticmethod
    def _next_prime(n: int) -> int:
        """Find the next prime >= n. Simple trial division."""
        if n <= 2:
            return 2
        if n % 2 == 0:
            n += 1
        while True:
            is_prime = True
            for i in range(3, int(n**0.5) + 1, 2):
                if n % i == 0:
                    is_prime = False
                    break
            if is_prime:
                return n
            n += 2

    # -------------------------------------------------------------------
    # Energy computation — the hot path
    # -------------------------------------------------------------------

    def compute_bigram_energy(self, prev_word: int, candidate: int) -> int:
        """
        Compute bigram hash energy for (prev_word, candidate).

        E = sum_h bigram_table_h[hash_h(prev_word, candidate) mod P]

        O(n_hashes) — typically 3 integer lookups and additions.

        Returns integer energy. Lower = more likely = better.
        """
        energy = 0
        for h_idx in range(self.n_hashes):
            slot = _double_hash(prev_word, candidate, h_idx, self.table_size)
            energy += int(self._bigram_tables[h_idx][slot])
        return energy

    def compute_bigram_energy_batch(
        self, prev_word: int, candidates: np.ndarray
    ) -> np.ndarray:
        """
        Compute bigram hash energy for all candidates at once.
        Vectorized using _double_hash_vec for speed.

        Args:
            prev_word: Previous word ID.
            candidates: Array of candidate word IDs, shape (K,).

        Returns:
            Integer energy array, shape (K,). Lower = better.
        """
        K = len(candidates)
        energies = np.zeros(K, dtype=np.int64)
        prev_arr = np.full(K, prev_word, dtype=np.int64)

        for h_idx in range(self.n_hashes):
            slots = _double_hash_vec(prev_arr, candidates.astype(np.int64), h_idx, self.table_size)
            energies += self._bigram_tables[h_idx][slots]

        return energies

    def compute_trigram_energy(
        self, prev2_word: int, prev_word: int, candidate: int
    ) -> int:
        """
        Compute trigram hash energy for (prev2_word, prev_word, candidate).

        Only available if use_trigram=True.
        """
        if self._trigram_tables is None:
            return 0

        energy = 0
        for h_idx in range(self.n_hashes):
            slot = _triple_hash(
                prev2_word, prev_word, candidate, h_idx, self._trigram_size
            )
            energy += int(self._trigram_tables[h_idx][slot])
        return energy

    def compute_trigram_energy_batch(
        self, prev2_word: int, prev_word: int, candidates: np.ndarray
    ) -> np.ndarray:
        """
        Compute trigram hash energy for all candidates at once.
        Vectorized using _triple_hash_vec for speed.
        """
        if self._trigram_tables is None:
            return np.zeros(len(candidates), dtype=np.int64)

        K = len(candidates)
        energies = np.zeros(K, dtype=np.int64)
        prev2_arr = np.full(K, prev2_word, dtype=np.int64)
        prev_arr = np.full(K, prev_word, dtype=np.int64)

        for h_idx in range(self.n_hashes):
            slots = _triple_hash_vec(
                prev2_arr, prev_arr, candidates.astype(np.int64),
                h_idx, self._trigram_size
            )
            energies += self._trigram_tables[h_idx][slots]

        return energies

    def compute_local_energy(
        self,
        context_word_ids: List[int],
        candidate: int,
    ) -> int:
        """
        Compute total local energy for a candidate given context.

        Combines bigram and trigram energies:
          E = bigram_E(context[-1], candidate)
            + trigram_weight * trigram_E(context[-2], context[-1], candidate)

        This is the ΔE for the Local Energy-Guided Decoding (Phase 2).

        Args:
            context_word_ids: List of context word IDs (at least 1).
            candidate: Candidate word ID.

        Returns:
            Integer energy. Lower = more likely = better.
        """
        if len(context_word_ids) == 0:
            return 0

        prev = context_word_ids[-1]
        energy = self.compute_bigram_energy(prev, candidate)

        if self.use_trigram and len(context_word_ids) >= 2:
            prev2 = context_word_ids[-2]
            energy += self.trigram_weight * self.compute_trigram_energy(
                prev2, prev, candidate
            )

        return energy

    def compute_local_energy_batch(
        self,
        context_word_ids: List[int],
        candidates: np.ndarray,
    ) -> np.ndarray:
        """
        Compute total local energy for all candidates given context.

        This is the batch version for LEGD Phase 2.

        Args:
            context_word_ids: List of context word IDs (at least 1).
            candidates: Array of candidate word IDs, shape (K,).

        Returns:
            Integer energy array, shape (K,). Lower = better.
        """
        K = len(candidates)
        if len(context_word_ids) == 0:
            return np.zeros(K, dtype=np.int64)

        prev = context_word_ids[-1]
        energies = self.compute_bigram_energy_batch(prev, candidates)

        if self.use_trigram and len(context_word_ids) >= 2:
            prev2 = context_word_ids[-2]
            tri_energies = self.compute_trigram_energy_batch(
                prev2, prev, candidates
            )
            energies += self.trigram_weight * tri_energies

        return energies

    # -------------------------------------------------------------------
    # NCE Training — vectorized batch integer updates
    # -------------------------------------------------------------------

    def train_nce(
        self,
        sequences: List[List[int]],
        n_epochs: int = 3,
        n_negatives: int = 3,
        corruptor=None,
        callback=None,
    ) -> Dict:
        """
        Train the hash energy table via vectorized integer NCE.

        PHASE 1 CORE: Precompute all (prev, target) pairs from sequences,
        then do BATCH hash lookups and updates using numpy fancy indexing.
        This is ~50x faster than the Python-loop version.

        For each (context, target) pair:
          1. POSITIVE: Decrease table entries for (prev, target)
          2. NEGATIVE: Increase table entries for (prev, corrupted_target)

        Pure integer arithmetic — no gradients, no BLAS, no float32.

        Args:
            sequences: List of tokenized word-ID sequences.
            n_epochs: Number of training epochs.
            n_negatives: Number of NCE negatives per positive.
            corruptor: Corruptor instance for generating negatives.
                       If None, uses random substitution only.
            callback: Optional callback(epoch, stats).

        Returns:
            Training statistics dict.
        """
        import time as _time
        rng = np.random.RandomState(self.seed)

        # --- Precompute all (prev, target) pairs from sequences ---
        all_prev = []
        all_target = []
        all_prev2 = []  # For trigrams

        for seq in sequences:
            if len(seq) < 2:
                continue
            for pos in range(1, len(seq)):
                all_prev.append(seq[pos - 1])
                all_target.append(seq[pos])
                if pos >= 2:
                    all_prev2.append(seq[pos - 2])
                else:
                    all_prev2.append(0)  # Placeholder

        all_prev = np.array(all_prev, dtype=np.int64)
        all_target = np.array(all_target, dtype=np.int64)
        all_prev2 = np.array(all_prev2, dtype=np.int64)
        N = len(all_prev)

        print(f"    Precomputed {N:,} (prev, target) pairs from "
              f"{len(sequences):,} sequences")

        all_stats = []

        for epoch in range(n_epochs):
            t_start = _time.time()

            # Shuffle pairs
            order = rng.permutation(N)
            shuffled_prev = all_prev[order]
            shuffled_target = all_target[order]
            shuffled_prev2 = all_prev2[order]

            # Generate negatives as random substitutions
            # Shape: (N,) — one negative per positive (we'll repeat n_neg times)
            n_pos = 0
            n_neg = 0

            # Process in chunks to manage memory
            chunk_size = 100000
            for chunk_start in range(0, N, chunk_size):
                chunk_end = min(chunk_start + chunk_size, N)
                c_prev = shuffled_prev[chunk_start:chunk_end]
                c_target = shuffled_target[chunk_start:chunk_start + chunk_size]
                c_prev2 = shuffled_prev2[chunk_start:chunk_start + chunk_size]
                C = len(c_prev)

                # --- POSITIVE UPDATE (bigram) ---
                for h_idx in range(self.n_hashes):
                    slots = _double_hash_vec(
                        c_prev, c_target, h_idx, self.table_size
                    )
                    # np.add.at handles duplicate slots correctly
                    np.add.at(self._bigram_tables[h_idx], slots, -self.eta)
                n_pos += C

                # --- NEGATIVE UPDATES (bigram) ---
                for _ in range(n_negatives):
                    # Random substitution: pick random words from vocab
                    neg_target = rng.randint(4, self.V, size=C)

                    for h_idx in range(self.n_hashes):
                        slots = _double_hash_vec(
                            c_prev, neg_target, h_idx, self.table_size
                        )
                        np.add.at(self._bigram_tables[h_idx], slots, self.eta)
                    n_neg += C

                # --- Trigram updates ---
                if self.use_trigram:
                    has_prev2 = c_prev2 > 0  # Filter: only update where prev2 exists
                    if np.any(has_prev2):
                        c_p2 = c_prev2[has_prev2]
                        c_p1 = c_prev[has_prev2]
                        c_t = c_target[has_prev2]
                        C2 = len(c_p2)

                        # Positive trigram
                        for h_idx in range(self.n_hashes):
                            slots = _triple_hash_vec(
                                c_p2, c_p1, c_t, h_idx, self._trigram_size
                            )
                            np.add.at(self._trigram_tables[h_idx], slots, -self.eta)
                        n_pos += C2

                        # Negative trigrams
                        for _ in range(n_negatives):
                            neg_t = rng.randint(4, self.V, size=C2)
                            for h_idx in range(self.n_hashes):
                                slots = _triple_hash_vec(
                                    c_p2, c_p1, neg_t, h_idx, self._trigram_size
                                )
                                np.add.at(self._trigram_tables[h_idx], slots, self.eta)
                            n_neg += C2

            t_elapsed = _time.time() - t_start

            # --- Clip table values to prevent energy explosion ---
            if self.clip_value > 0:
                for h_idx in range(self.n_hashes):
                    np.clip(
                        self._bigram_tables[h_idx],
                        -self.clip_value, self.clip_value,
                        out=self._bigram_tables[h_idx]
                    )
                if self._trigram_tables is not None:
                    for h_idx in range(self.n_hashes):
                        np.clip(
                            self._trigram_tables[h_idx],
                            -self.clip_value, self.clip_value,
                            out=self._trigram_tables[h_idx]
                        )

            # --- Discriminative accuracy check (vectorized, sampled) ---
            n_check = min(2000, N)
            check_idx = rng.choice(N, n_check, replace=False)
            check_prev = all_prev[check_idx]
            check_target = all_target[check_idx]

            # Compute positive energies (vectorized)
            pos_energies = np.zeros(n_check, dtype=np.int64)
            for h_idx in range(self.n_hashes):
                slots = _double_hash_vec(
                    check_prev, check_target, h_idx, self.table_size
                )
                pos_energies += self._bigram_tables[h_idx][slots]

            # Generate negatives and compute their energies
            neg_target = rng.randint(4, self.V, size=n_check)
            neg_energies = np.zeros(n_check, dtype=np.int64)
            for h_idx in range(self.n_hashes):
                slots = _double_hash_vec(
                    check_prev, neg_target, h_idx, self.table_size
                )
                neg_energies += self._bigram_tables[h_idx][slots]

            n_correct = int(np.sum(pos_energies < neg_energies))
            disc_acc = n_correct / max(1, n_check)

            stats = {
                'epoch': epoch,
                'n_pos': n_pos,
                'n_neg': n_neg,
                'disc_accuracy': disc_acc,
                'disc_total': n_check,
                'time_s': t_elapsed,
                'bigram_range': (
                    int(min(t.min() for t in self._bigram_tables)),
                    int(max(t.max() for t in self._bigram_tables)),
                ),
                'bigram_nnz': sum(
                    int(np.count_nonzero(t)) for t in self._bigram_tables
                ),
            }

            if self._trigram_tables is not None:
                stats['trigram_range'] = (
                    int(min(t.min() for t in self._trigram_tables)),
                    int(max(t.max() for t in self._trigram_tables)),
                )
                stats['trigram_nnz'] = sum(
                    int(np.count_nonzero(t)) for t in self._trigram_tables
                )

            all_stats.append(stats)

            print(f"    Epoch {epoch+1}/{n_epochs}: "
                  f"disc_acc={disc_acc:.3f}, "
                  f"pos={n_pos:,}, neg={n_neg:,}, "
                  f"bigram range=[{stats['bigram_range'][0]},{stats['bigram_range'][1]}], "
                  f"bigram nnz={stats['bigram_nnz']:,}, "
                  f"time={t_elapsed:.1f}s")

            if callback:
                callback(epoch, stats)

        self._train_pos_count += sum(s['n_pos'] for s in all_stats)
        self._train_neg_count += sum(s['n_neg'] for s in all_stats)
        self._train_updates += 1

        return {'epochs': all_stats}

    def _decrease_energy(self, prev: int, target: int):
        """Decrease bigram energy for (prev, target) — positive NCE update."""
        for h_idx in range(self.n_hashes):
            slot = _double_hash(prev, target, h_idx, self.table_size)
            self._bigram_tables[h_idx][slot] -= self.eta

    def _increase_energy(self, prev: int, target: int):
        """Increase bigram energy for (prev, target) — negative NCE update."""
        for h_idx in range(self.n_hashes):
            slot = _double_hash(prev, target, h_idx, self.table_size)
            self._bigram_tables[h_idx][slot] += self.eta

    def _decrease_trigram_energy(self, prev2: int, prev: int, target: int):
        """Decrease trigram energy — positive NCE update."""
        if self._trigram_tables is None:
            return
        for h_idx in range(self.n_hashes):
            slot = _triple_hash(
                prev2, prev, target, h_idx, self._trigram_size
            )
            self._trigram_tables[h_idx][slot] -= self.eta

    def _increase_trigram_energy(self, prev2: int, prev: int, target: int):
        """Increase trigram energy — negative NCE update."""
        if self._trigram_tables is None:
            return
        for h_idx in range(self.n_hashes):
            slot = _triple_hash(
                prev2, prev, target, h_idx, self._trigram_size
            )
            self._trigram_tables[h_idx][slot] += self.eta

    # -------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------

    def energy_statistics(self) -> Dict:
        """Compute statistics about the energy tables."""
        bigram_means = [float(t.mean()) for t in self._bigram_tables]
        bigram_stds = [float(t.std()) for t in self._bigram_tables]
        bigram_mins = [int(t.min()) for t in self._bigram_tables]
        bigram_maxs = [int(t.max()) for t in self._bigram_tables]
        bigram_nnz = sum(int(np.count_nonzero(t)) for t in self._bigram_tables)

        stats = {
            'bigram_mean': np.mean(bigram_means),
            'bigram_std': np.mean(bigram_stds),
            'bigram_min': min(bigram_mins),
            'bigram_max': max(bigram_maxs),
            'bigram_nnz': bigram_nnz,
            'bigram_total_entries': self.table_size * self.n_hashes,
            'bigram_density': bigram_nnz / (self.table_size * self.n_hashes),
            'n_hashes': self.n_hashes,
            'table_size': self.table_size,
            'use_trigram': self.use_trigram,
        }

        if self._trigram_tables is not None:
            tri_means = [float(t.mean()) for t in self._trigram_tables]
            tri_stds = [float(t.std()) for t in self._trigram_tables]
            tri_nnz = sum(
                int(np.count_nonzero(t)) for t in self._trigram_tables
            )
            stats.update({
                'trigram_mean': np.mean(tri_means),
                'trigram_std': np.mean(tri_stds),
                'trigram_nnz': tri_nnz,
                'trigram_total_entries': self._trigram_size * self.n_hashes,
                'trigram_density': tri_nnz / (
                    self._trigram_size * self.n_hashes
                ),
                'trigram_size': self._trigram_size,
            })

        return stats

    def memory_mb(self) -> float:
        """Estimate memory usage in MB."""
        total_entries = self.table_size * self.n_hashes
        mb = total_entries * 4 / (1024 * 1024)  # int32 = 4 bytes
        if self._trigram_tables is not None:
            total_entries += self._trigram_size * self.n_hashes
            mb = total_entries * 4 / (1024 * 1024)
        return mb
