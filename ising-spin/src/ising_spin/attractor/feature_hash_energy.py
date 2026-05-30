"""
Feature-Hashed Integer Energy Table — Level 1 Generalization.

v78: Replace single lexical hash with multi-feature hash ensemble.

The v77 HashEnergyTable hashes raw token IDs. This gives zero generalization:
learning "the cat" tells you nothing about "a dog" because they hash to
completely different slots.

FeatureHashEnergyTable adds POS (Part-of-Speech) hash tables that learn
CATEGORY-LEVEL rules, not token-specific facts:

  E = pos_weight * E_pos  +  lex_weight * E_lex

Where:
  E_pos = sum_h table_pos_h[hash(POS(prev), POS(target)) mod P_pos]
        + tri_weight * sum_h table_pos_tri_h[hash(POS(p2), POS(p1), POS(t)) mod P_pos_tri]

  E_lex = sum_h table_lex_h[hash(prev_id, target_id) mod P_lex]      # same as v77
        + tri_weight * sum_h table_lex_tri_h[hash(p2, p1, t) mod P_lex_tri]

KEY INSIGHT — Why POS tables generalize:
  With 13 POS types, there are only 13×13 = 169 possible POS bigram pairs.
  Every DET→NOUN transition ("the cat", "a dog", "this house", ...) maps
  to the SAME hash slot. Training on "the cat" automatically improves
  the score for "a dog" — without ever seeing that pair.

  The POS table learns RULES:
    DET→NOUN = strongly negative (good)
    DET→VERB = weakly positive (bad but sometimes ok: "the running...")
    NOUN→VERB = negative (good)
    NOUN→DET = strongly positive (bad: "cat the")
    PUNCT→NOUN = negative (good: ". The")
    PUNCT→PUNCT = positive (bad: "..")

  These rules apply to ALL words with those POS tags, including words
  the model has never seen in training. That's generalization.

WHY NOT JUST USE A 13×13 MATRIX?
  We could — but the hash table structure:
  1. Is consistent with the lexical table code
  2. Supports multi-hash ensemble (2-3 independent hash functions)
  3. Scales trivially if we add more features later
  4. With P=1009, we get essentially zero collisions for 169 pairs

TRAINING:
  For each (prev, target) pair from real text:
    POS positive: table_pos[hash(POS(prev), POS(target))] -= pos_eta
    POS negative: table_pos[hash(POS(prev), POS(neg))]    += pos_eta
    LEX positive: table_lex[hash(prev_id, target_id)]     -= lex_eta
    LEX negative: table_lex[hash(prev_id, neg_id)]        += lex_eta

  The POS table converges MUCH faster than the lexical table because each
  of the 169 slots receives millions of training updates.

INFERENCE:
  delta_E = pos_weight * E_pos(prev_pos, target_pos) + lex_weight * E_lex(prev_id, target_id)

  This is STILL O(1) per candidate — just a few more integer lookups.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


# ---------------------------------------------------------------------------
# Hash functions — reuse from hash_energy for consistency
# ---------------------------------------------------------------------------

# Prime constants for hashing
_P1 = np.int64(2654435761)
_P2 = np.int64(2246822519)
_P3 = np.int64(3266489917)
_P4 = np.int64(3367900313)
_MASK32 = np.int64(0xFFFFFFFF)


def _double_hash_vec(a: np.ndarray, b: np.ndarray, h_idx: int, P: int) -> np.ndarray:
    """Vectorized double-hash for arrays of (a, b) pairs."""
    val = (a.astype(np.int64) * _P1 + b.astype(np.int64) * _P2
           + np.int64(h_idx) * _P3) & _MASK32
    return val % np.int64(P)


def _triple_hash_vec(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                     h_idx: int, P: int) -> np.ndarray:
    """Vectorized triple-hash for arrays of (a, b, c) triples."""
    val = (a.astype(np.int64) * _P1 + b.astype(np.int64) * _P2
           + c.astype(np.int64) * _P4 + np.int64(h_idx) * _P3) & _MASK32
    return val % np.int64(P)


def _double_hash(a: int, b: int, h_idx: int, P: int) -> int:
    """Scalar double-hash."""
    P1, P2, P3 = 2654435761, 2246822519, 3266489917
    val = (a * P1 + b * P2 + h_idx * P3) & 0xFFFFFFFF
    return int(val % P)


def _triple_hash(a: int, b: int, c: int, h_idx: int, P: int) -> int:
    """Scalar triple-hash."""
    P1, P2, P3, P4 = 2654435761, 2246822519, 3266489917, 3367900313
    val = (a * P1 + b * P2 + c * P4 + h_idx * P3) & 0xFFFFFFFF
    return int(val % P)


# ---------------------------------------------------------------------------
# FeatureHashEnergyTable — the core data structure
# ---------------------------------------------------------------------------

class FeatureHashEnergyTable:
    """
    Feature-Hashed Integer Energy Table with POS generalization.

    Combines:
      1. POS bigram/trigram hash tables — category-level RULES
      2. Lexical bigram/trigram hash tables — token-specific FACTS

    The POS tables are the key innovation: they learn that DET→NOUN is
    good and NOUN→DET is bad, and this applies to ALL words with those
    POS tags — including words the model has never seen together.
    """

    def __init__(
        self,
        vocab_size: int,
        word_pos: np.ndarray,
        n_pos_types: int = 13,
        # POS table parameters
        n_pos_hashes: int = 2,
        pos_table_size: int = 1009,
        pos_eta: int = 3,
        pos_clip: int = 500,
        # Lexical table parameters
        n_lex_hashes: int = 3,
        lex_table_size: int = 65537,
        lex_eta: int = 1,
        lex_clip: int = 1000,
        # Trigram
        use_trigram: bool = True,
        trigram_weight: int = 1,
        # Energy combination weights
        pos_weight: float = 1.0,
        lex_weight: float = 1.0,
        # Seed
        seed: int = 42,
    ):
        """
        Args:
            vocab_size: Number of words in vocabulary (V).
            word_pos: Array of shape (V,) mapping word_id -> primary POS type.
                      Values in [0, n_pos_types).
            n_pos_types: Number of distinct POS categories.
            n_pos_hashes: Number of hash functions for POS tables.
                          Fewer needed because only 13×13 = 169 possible pairs.
            pos_table_size: Prime for POS hash table. Small is fine (1009)
                            because we only have 169 possible POS bigrams.
            pos_eta: NCE learning rate for POS tables. Higher than lexical
                     because we want POS rules to converge fast.
            pos_clip: Clip value for POS tables.
            n_lex_hashes: Number of hash functions for lexical tables.
            lex_table_size: Prime for lexical hash table (same as v77).
            lex_eta: NCE learning rate for lexical tables.
            lex_clip: Clip value for lexical tables.
            use_trigram: Whether to also use trigram hash tables.
            trigram_weight: Relative weight of trigram vs bigram.
            pos_weight: Weight for POS energy in combined score.
            lex_weight: Weight for lexical energy in combined score.
            seed: Random seed.
        """
        self.V = vocab_size
        self.word_pos = word_pos.astype(np.int32)
        self.n_pos_types = n_pos_types

        # POS table parameters
        self.n_pos_hashes = n_pos_hashes
        self.pos_table_size = pos_table_size
        self.pos_eta = pos_eta
        self.pos_clip = pos_clip

        # Lexical table parameters
        self.n_lex_hashes = n_lex_hashes
        self.lex_table_size = lex_table_size
        self.lex_eta = lex_eta
        self.lex_clip = lex_clip

        # Shared parameters
        self.use_trigram = use_trigram
        self.trigram_weight = trigram_weight
        self.pos_weight = pos_weight
        self.lex_weight = lex_weight
        self.seed = seed

        # ---- POS bigram tables ----
        self._pos_bigram_tables = [
            np.zeros(pos_table_size, dtype=np.int32)
            for _ in range(n_pos_hashes)
        ]

        # ---- POS trigram tables (optional) ----
        self._pos_trigram_tables = None
        self._pos_trigram_size = 0
        if use_trigram:
            self._pos_trigram_size = self._next_prime(pos_table_size + 256)
            self._pos_trigram_tables = [
                np.zeros(self._pos_trigram_size, dtype=np.int32)
                for _ in range(n_pos_hashes)
            ]

        # ---- Lexical bigram tables ----
        self._lex_bigram_tables = [
            np.zeros(lex_table_size, dtype=np.int32)
            for _ in range(n_lex_hashes)
        ]

        # ---- Lexical trigram tables (optional) ----
        self._lex_trigram_tables = None
        self._lex_trigram_size = 0
        if use_trigram:
            self._lex_trigram_size = self._next_prime(lex_table_size + 256)
            self._lex_trigram_tables = [
                np.zeros(self._lex_trigram_size, dtype=np.int32)
                for _ in range(n_lex_hashes)
            ]

        # Training statistics
        self._train_updates = 0

    @staticmethod
    def _next_prime(n: int) -> int:
        """Find the next prime >= n."""
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
    # Energy computation — POS component
    # -------------------------------------------------------------------

    def _compute_pos_bigram_energy_batch(
        self, prev_pos: int, candidates: np.ndarray
    ) -> np.ndarray:
        """Compute POS bigram energy for all candidates. Vectorized."""
        K = len(candidates)
        energies = np.zeros(K, dtype=np.int64)
        prev_arr = np.full(K, prev_pos, dtype=np.int64)
        cand_pos = self.word_pos[candidates].astype(np.int64)

        for h_idx in range(self.n_pos_hashes):
            slots = _double_hash_vec(
                prev_arr, cand_pos, h_idx, self.pos_table_size
            )
            energies += self._pos_bigram_tables[h_idx][slots]

        return energies

    def _compute_pos_trigram_energy_batch(
        self, prev2_pos: int, prev_pos: int, candidates: np.ndarray
    ) -> np.ndarray:
        """Compute POS trigram energy for all candidates. Vectorized."""
        if self._pos_trigram_tables is None:
            return np.zeros(len(candidates), dtype=np.int64)

        K = len(candidates)
        energies = np.zeros(K, dtype=np.int64)
        prev2_arr = np.full(K, prev2_pos, dtype=np.int64)
        prev_arr = np.full(K, prev_pos, dtype=np.int64)
        cand_pos = self.word_pos[candidates].astype(np.int64)

        for h_idx in range(self.n_pos_hashes):
            slots = _triple_hash_vec(
                prev2_arr, prev_arr, cand_pos,
                h_idx, self._pos_trigram_size
            )
            energies += self._pos_trigram_tables[h_idx][slots]

        return energies

    # -------------------------------------------------------------------
    # Energy computation — Lexical component
    # -------------------------------------------------------------------

    def _compute_lex_bigram_energy_batch(
        self, prev_word: int, candidates: np.ndarray
    ) -> np.ndarray:
        """Compute lexical bigram energy for all candidates. Vectorized."""
        K = len(candidates)
        energies = np.zeros(K, dtype=np.int64)
        prev_arr = np.full(K, prev_word, dtype=np.int64)

        for h_idx in range(self.n_lex_hashes):
            slots = _double_hash_vec(
                prev_arr, candidates.astype(np.int64),
                h_idx, self.lex_table_size
            )
            energies += self._lex_bigram_tables[h_idx][slots]

        return energies

    def _compute_lex_trigram_energy_batch(
        self, prev2_word: int, prev_word: int, candidates: np.ndarray
    ) -> np.ndarray:
        """Compute lexical trigram energy for all candidates. Vectorized."""
        if self._lex_trigram_tables is None:
            return np.zeros(len(candidates), dtype=np.int64)

        K = len(candidates)
        energies = np.zeros(K, dtype=np.int64)
        prev2_arr = np.full(K, prev2_word, dtype=np.int64)
        prev_arr = np.full(K, prev_word, dtype=np.int64)

        for h_idx in range(self.n_lex_hashes):
            slots = _triple_hash_vec(
                prev2_arr, prev_arr, candidates.astype(np.int64),
                h_idx, self._lex_trigram_size
            )
            energies += self._lex_trigram_tables[h_idx][slots]

        return energies

    # -------------------------------------------------------------------
    # Energy computation — combined (the hot path for LEGD)
    # -------------------------------------------------------------------

    def compute_local_energy(
        self,
        context_word_ids: List[int],
        candidate: int,
    ) -> int:
        """
        Compute total local energy for a candidate given context.

        E = pos_weight * E_pos + lex_weight * E_lex

        Where:
          E_pos = E_pos_bigram(context[-1], candidate)
                + tri_weight * E_pos_trigram(context[-2], context[-1], candidate)
          E_lex = E_lex_bigram(context[-1], candidate)
                + tri_weight * E_lex_trigram(context[-2], context[-1], candidate)

        Returns integer energy. Lower = more likely = better.
        """
        if len(context_word_ids) == 0:
            return 0

        prev = context_word_ids[-1]
        prev_pos = int(self.word_pos[prev])
        cand_pos = int(self.word_pos[candidate]) if candidate < self.V else 0

        # POS bigram energy
        pos_energy = 0
        for h_idx in range(self.n_pos_hashes):
            slot = _double_hash(prev_pos, cand_pos, h_idx, self.pos_table_size)
            pos_energy += int(self._pos_bigram_tables[h_idx][slot])

        # Lexical bigram energy
        lex_energy = 0
        for h_idx in range(self.n_lex_hashes):
            slot = _double_hash(prev, candidate, h_idx, self.lex_table_size)
            lex_energy += int(self._lex_bigram_tables[h_idx][slot])

        # Trigrams
        if self.use_trigram and len(context_word_ids) >= 2:
            prev2 = context_word_ids[-2]
            prev2_pos = int(self.word_pos[prev2])

            # POS trigram
            if self._pos_trigram_tables is not None:
                for h_idx in range(self.n_pos_hashes):
                    slot = _triple_hash(
                        prev2_pos, prev_pos, cand_pos,
                        h_idx, self._pos_trigram_size
                    )
                    pos_energy += self.trigram_weight * int(
                        self._pos_trigram_tables[h_idx][slot]
                    )

            # Lexical trigram
            if self._lex_trigram_tables is not None:
                for h_idx in range(self.n_lex_hashes):
                    slot = _triple_hash(
                        prev2, prev, candidate,
                        h_idx, self._lex_trigram_size
                    )
                    lex_energy += self.trigram_weight * int(
                        self._lex_trigram_tables[h_idx][slot]
                    )

        # Weighted combination
        # Use integer approximation: scale weights to avoid float
        # pos_weight and lex_weight are floats, but the energy is integer
        total = int(self.pos_weight * pos_energy + self.lex_weight * lex_energy)
        return total

    def compute_local_energy_batch(
        self,
        context_word_ids: List[int],
        candidates: np.ndarray,
    ) -> np.ndarray:
        """
        Compute total local energy for all candidates given context.
        Vectorized batch version for LEGD Phase 2.

        Returns integer energy array. Lower = better.
        """
        K = len(candidates)
        if len(context_word_ids) == 0:
            return np.zeros(K, dtype=np.int64)

        prev = context_word_ids[-1]
        prev_pos = int(self.word_pos[prev])

        # POS bigram energy
        pos_energies = self._compute_pos_bigram_energy_batch(
            prev_pos, candidates
        )

        # Lexical bigram energy
        lex_energies = self._compute_lex_bigram_energy_batch(
            prev, candidates
        )

        # Trigrams
        if self.use_trigram and len(context_word_ids) >= 2:
            prev2 = context_word_ids[-2]
            prev2_pos = int(self.word_pos[prev2])

            pos_energies += self.trigram_weight * self._compute_pos_trigram_energy_batch(
                prev2_pos, prev_pos, candidates
            )
            lex_energies += self.trigram_weight * self._compute_lex_trigram_energy_batch(
                prev2, prev, candidates
            )

        # Weighted combination
        combined = (self.pos_weight * pos_energies.astype(np.float64)
                    + self.lex_weight * lex_energies.astype(np.float64))

        return combined.astype(np.int64)

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
        Train both POS and lexical hash tables via vectorized integer NCE.

        For each (prev, target) pair from real text:
          POS: table_pos[hash(POS(prev), POS(target))]  -= pos_eta  (positive)
               table_pos[hash(POS(prev), POS(neg))]     += pos_eta  (negative)
          LEX: table_lex[hash(prev, target)]             -= lex_eta  (positive)
               table_lex[hash(prev, neg)]                += lex_eta  (negative)

        The POS table converges MUCH faster because it has only 169
        possible bigram patterns — each slot receives millions of updates.
        """
        import time as _time
        rng = np.random.RandomState(self.seed)

        # --- Precompute all (prev, target) pairs from sequences ---
        all_prev = []
        all_target = []
        all_prev2 = []

        for seq in sequences:
            if len(seq) < 2:
                continue
            for pos in range(1, len(seq)):
                all_prev.append(seq[pos - 1])
                all_target.append(seq[pos])
                if pos >= 2:
                    all_prev2.append(seq[pos - 2])
                else:
                    all_prev2.append(0)

        all_prev = np.array(all_prev, dtype=np.int64)
        all_target = np.array(all_target, dtype=np.int64)
        all_prev2 = np.array(all_prev2, dtype=np.int64)
        N = len(all_prev)

        # Precompute POS types for prev and target
        all_prev_pos = self.word_pos[all_prev].astype(np.int64)
        all_target_pos = self.word_pos[all_target].astype(np.int64)
        all_prev2_pos = self.word_pos[all_prev2].astype(np.int64)

        print(f"    Precomputed {N:,} (prev, target) pairs from "
              f"{len(sequences):,} sequences")
        print(f"    POS types: {self.n_pos_types} categories, "
              f"{self.n_pos_types}x{self.n_pos_types}={self.n_pos_types**2} possible pairs")

        all_stats = []

        for epoch in range(n_epochs):
            t_start = _time.time()

            # Shuffle pairs
            order = rng.permutation(N)
            shuffled_prev = all_prev[order]
            shuffled_target = all_target[order]
            shuffled_prev2 = all_prev2[order]
            shuffled_prev_pos = all_prev_pos[order]
            shuffled_target_pos = all_target_pos[order]
            shuffled_prev2_pos = all_prev2_pos[order]

            n_pos_updates = 0
            n_neg_updates = 0

            # Process in chunks
            chunk_size = 100000
            for chunk_start in range(0, N, chunk_size):
                chunk_end = min(chunk_start + chunk_size, N)
                c_prev = shuffled_prev[chunk_start:chunk_end]
                c_target = shuffled_target[chunk_start:chunk_end]
                c_prev2 = shuffled_prev2[chunk_start:chunk_end]
                c_prev_pos = shuffled_prev_pos[chunk_start:chunk_end]
                c_target_pos = shuffled_target_pos[chunk_start:chunk_end]
                c_prev2_pos = shuffled_prev2_pos[chunk_start:chunk_end]
                C = len(c_prev)

                # ===== POSITIVE UPDATES =====

                # POS bigram positive
                for h_idx in range(self.n_pos_hashes):
                    slots = _double_hash_vec(
                        c_prev_pos, c_target_pos, h_idx, self.pos_table_size
                    )
                    np.add.at(self._pos_bigram_tables[h_idx], slots, -self.pos_eta)

                # Lexical bigram positive
                for h_idx in range(self.n_lex_hashes):
                    slots = _double_hash_vec(
                        c_prev, c_target, h_idx, self.lex_table_size
                    )
                    np.add.at(self._lex_bigram_tables[h_idx], slots, -self.lex_eta)

                n_pos_updates += C

                # ===== NEGATIVE UPDATES =====

                for _ in range(n_negatives):
                    # Random substitution for negatives
                    neg_target = rng.randint(4, self.V, size=C)
                    neg_target_pos = self.word_pos[neg_target].astype(np.int64)

                    # POS bigram negative
                    for h_idx in range(self.n_pos_hashes):
                        slots = _double_hash_vec(
                            c_prev_pos, neg_target_pos,
                            h_idx, self.pos_table_size
                        )
                        np.add.at(
                            self._pos_bigram_tables[h_idx], slots, self.pos_eta
                        )

                    # Lexical bigram negative
                    for h_idx in range(self.n_lex_hashes):
                        slots = _double_hash_vec(
                            c_prev, neg_target, h_idx, self.lex_table_size
                        )
                        np.add.at(
                            self._lex_bigram_tables[h_idx], slots, self.lex_eta
                        )

                    n_neg_updates += C

                # ===== TRIGRAM UPDATES =====
                if self.use_trigram:
                    has_prev2 = c_prev2 > 0
                    if np.any(has_prev2):
                        c_p2 = c_prev2[has_prev2]
                        c_p1 = c_prev[has_prev2]
                        c_t = c_target[has_prev2]
                        c_p2_pos = c_prev2_pos[has_prev2]
                        c_p1_pos = c_prev_pos[has_prev2]
                        c_t_pos = c_target_pos[has_prev2]
                        C2 = len(c_p2)

                        # POS trigram positive
                        if self._pos_trigram_tables is not None:
                            for h_idx in range(self.n_pos_hashes):
                                slots = _triple_hash_vec(
                                    c_p2_pos, c_p1_pos, c_t_pos,
                                    h_idx, self._pos_trigram_size
                                )
                                np.add.at(
                                    self._pos_trigram_tables[h_idx],
                                    slots, -self.pos_eta
                                )

                        # Lexical trigram positive
                        if self._lex_trigram_tables is not None:
                            for h_idx in range(self.n_lex_hashes):
                                slots = _triple_hash_vec(
                                    c_p2, c_p1, c_t,
                                    h_idx, self._lex_trigram_size
                                )
                                np.add.at(
                                    self._lex_trigram_tables[h_idx],
                                    slots, -self.lex_eta
                                )

                        n_pos_updates += C2

                        # Trigram negatives
                        for _ in range(n_negatives):
                            neg_t = rng.randint(4, self.V, size=C2)
                            neg_t_pos = self.word_pos[neg_t].astype(np.int64)

                            if self._pos_trigram_tables is not None:
                                for h_idx in range(self.n_pos_hashes):
                                    slots = _triple_hash_vec(
                                        c_p2_pos, c_p1_pos, neg_t_pos,
                                        h_idx, self._pos_trigram_size
                                    )
                                    np.add.at(
                                        self._pos_trigram_tables[h_idx],
                                        slots, self.pos_eta
                                    )

                            if self._lex_trigram_tables is not None:
                                for h_idx in range(self.n_lex_hashes):
                                    slots = _triple_hash_vec(
                                        c_p2, c_p1, neg_t,
                                        h_idx, self._lex_trigram_size
                                    )
                                    np.add.at(
                                        self._lex_trigram_tables[h_idx],
                                        slots, self.lex_eta
                                    )

                            n_neg_updates += C2

            t_elapsed = _time.time() - t_start

            # --- Clip table values ---
            if self.pos_clip > 0:
                for h_idx in range(self.n_pos_hashes):
                    np.clip(
                        self._pos_bigram_tables[h_idx],
                        -self.pos_clip, self.pos_clip,
                        out=self._pos_bigram_tables[h_idx]
                    )
                if self._pos_trigram_tables is not None:
                    for h_idx in range(self.n_pos_hashes):
                        np.clip(
                            self._pos_trigram_tables[h_idx],
                            -self.pos_clip, self.pos_clip,
                            out=self._pos_trigram_tables[h_idx]
                        )

            if self.lex_clip > 0:
                for h_idx in range(self.n_lex_hashes):
                    np.clip(
                        self._lex_bigram_tables[h_idx],
                        -self.lex_clip, self.lex_clip,
                        out=self._lex_bigram_tables[h_idx]
                    )
                if self._lex_trigram_tables is not None:
                    for h_idx in range(self.n_lex_hashes):
                        np.clip(
                            self._lex_trigram_tables[h_idx],
                            -self.lex_clip, self.lex_clip,
                            out=self._lex_trigram_tables[h_idx]
                        )

            # --- Discriminative accuracy check ---
            n_check = min(2000, N)
            check_idx = rng.choice(N, n_check, replace=False)
            check_prev = all_prev[check_idx]
            check_target = all_target[check_idx]
            check_prev_pos = all_prev_pos[check_idx]
            check_target_pos = all_target_pos[check_idx]

            # POS discrimination
            pos_e_real = np.zeros(n_check, dtype=np.int64)
            for h_idx in range(self.n_pos_hashes):
                slots = _double_hash_vec(
                    check_prev_pos, check_target_pos,
                    h_idx, self.pos_table_size
                )
                pos_e_real += self._pos_bigram_tables[h_idx][slots]

            neg_target = rng.randint(4, self.V, size=n_check)
            neg_target_pos = self.word_pos[neg_target].astype(np.int64)
            pos_e_neg = np.zeros(n_check, dtype=np.int64)
            for h_idx in range(self.n_pos_hashes):
                slots = _double_hash_vec(
                    check_prev_pos, neg_target_pos,
                    h_idx, self.pos_table_size
                )
                pos_e_neg += self._pos_bigram_tables[h_idx][slots]

            pos_disc_acc = float(np.sum(pos_e_real < pos_e_neg)) / n_check

            # Lexical discrimination
            lex_e_real = np.zeros(n_check, dtype=np.int64)
            for h_idx in range(self.n_lex_hashes):
                slots = _double_hash_vec(
                    check_prev, check_target, h_idx, self.lex_table_size
                )
                lex_e_real += self._lex_bigram_tables[h_idx][slots]

            lex_e_neg = np.zeros(n_check, dtype=np.int64)
            for h_idx in range(self.n_lex_hashes):
                slots = _double_hash_vec(
                    check_prev, neg_target, h_idx, self.lex_table_size
                )
                lex_e_neg += self._lex_bigram_tables[h_idx][slots]

            lex_disc_acc = float(np.sum(lex_e_real < lex_e_neg)) / n_check

            # Combined discrimination
            combined_real = self.pos_weight * pos_e_real.astype(np.float64) + self.lex_weight * lex_e_real.astype(np.float64)
            combined_neg = self.pos_weight * pos_e_neg.astype(np.float64) + self.lex_weight * lex_e_neg.astype(np.float64)
            combined_disc_acc = float(np.sum(combined_real < combined_neg)) / n_check

            # POS table analysis — show the top learned rules
            pos_rules = self._analyze_pos_rules(top_k=5)

            stats = {
                'epoch': epoch,
                'n_pos_updates': n_pos_updates,
                'n_neg_updates': n_neg_updates,
                'pos_disc_accuracy': pos_disc_acc,
                'lex_disc_accuracy': lex_disc_acc,
                'combined_disc_accuracy': combined_disc_acc,
                'time_s': t_elapsed,
                'pos_bigram_range': (
                    int(min(t.min() for t in self._pos_bigram_tables)),
                    int(max(t.max() for t in self._pos_bigram_tables)),
                ),
                'pos_bigram_nnz': sum(
                    int(np.count_nonzero(t)) for t in self._pos_bigram_tables
                ),
                'lex_bigram_range': (
                    int(min(t.min() for t in self._lex_bigram_tables)),
                    int(max(t.max() for t in self._lex_bigram_tables)),
                ),
                'lex_bigram_nnz': sum(
                    int(np.count_nonzero(t)) for t in self._lex_bigram_tables
                ),
                'top_pos_rules': pos_rules,
            }

            if self._pos_trigram_tables is not None:
                stats['pos_trigram_nnz'] = sum(
                    int(np.count_nonzero(t)) for t in self._pos_trigram_tables
                )
            if self._lex_trigram_tables is not None:
                stats['lex_trigram_nnz'] = sum(
                    int(np.count_nonzero(t)) for t in self._lex_trigram_tables
                )

            all_stats.append(stats)

            print(f"    Epoch {epoch+1}/{n_epochs}: "
                  f"pos_disc={pos_disc_acc:.3f}, lex_disc={lex_disc_acc:.3f}, "
                  f"combined={combined_disc_acc:.3f}, "
                  f"pos_range=[{stats['pos_bigram_range'][0]},{stats['pos_bigram_range'][1]}], "
                  f"lex_range=[{stats['lex_bigram_range'][0]},{stats['lex_bigram_range'][1]}], "
                  f"time={t_elapsed:.1f}s")

            # Print top POS rules
            print(f"      Top POS rules (lower = more likely):")
            for rule in pos_rules[:5]:
                print(f"        {rule['from']} -> {rule['to']}: "
                      f"energy={rule['avg_energy']:.0f}")

            if callback:
                callback(epoch, stats)

        self._train_updates += 1

        return {'epochs': all_stats}

    def _analyze_pos_rules(self, top_k: int = 10) -> List[Dict]:
        """Analyze what rules the POS table has learned."""
        from ..vocabulary.pos import IDX2POS

        # Average energy per POS pair across hash tables
        pair_energies = {}
        for t1 in range(self.n_pos_types):
            for t2 in range(self.n_pos_types):
                total = 0
                count = 0
                for h_idx in range(self.n_pos_hashes):
                    slot = _double_hash(t1, t2, h_idx, self.pos_table_size)
                    val = self._pos_bigram_tables[h_idx][slot]
                    total += val
                    count += 1
                avg = total / max(1, count)
                name1 = IDX2POS.get(t1, f"T{t1}")
                name2 = IDX2POS.get(t2, f"T{t2}")
                pair_energies[(name1, name2)] = avg

        # Sort by energy (lowest = most likely transition)
        sorted_pairs = sorted(pair_energies.items(), key=lambda x: x[1])

        rules = []
        for (from_pos, to_pos), energy in sorted_pairs[:top_k]:
            rules.append({
                'from': from_pos,
                'to': to_pos,
                'avg_energy': energy,
            })

        # Also add the worst rules
        for (from_pos, to_pos), energy in sorted_pairs[-top_k:]:
            rules.append({
                'from': from_pos,
                'to': to_pos,
                'avg_energy': energy,
            })

        return rules

    # -------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------

    def energy_statistics(self) -> Dict:
        """Compute statistics about the energy tables."""
        pos_means = [float(t.mean()) for t in self._pos_bigram_tables]
        pos_stds = [float(t.std()) for t in self._pos_bigram_tables]
        lex_means = [float(t.mean()) for t in self._lex_bigram_tables]
        lex_stds = [float(t.std()) for t in self._lex_bigram_tables]

        stats = {
            # POS tables
            'pos_bigram_mean': np.mean(pos_means),
            'pos_bigram_std': np.mean(pos_stds),
            'pos_bigram_min': int(min(t.min() for t in self._pos_bigram_tables)),
            'pos_bigram_max': int(max(t.max() for t in self._pos_bigram_tables)),
            'pos_bigram_nnz': sum(int(np.count_nonzero(t)) for t in self._pos_bigram_tables),
            'pos_bigram_total_entries': self.pos_table_size * self.n_pos_hashes,
            'pos_bigram_density': sum(
                int(np.count_nonzero(t)) for t in self._pos_bigram_tables
            ) / (self.pos_table_size * self.n_pos_hashes),

            # Lexical tables
            'lex_bigram_mean': np.mean(lex_means),
            'lex_bigram_std': np.mean(lex_stds),
            'lex_bigram_min': int(min(t.min() for t in self._lex_bigram_tables)),
            'lex_bigram_max': int(max(t.max() for t in self._lex_bigram_tables)),
            'lex_bigram_nnz': sum(int(np.count_nonzero(t)) for t in self._lex_bigram_tables),
            'lex_bigram_total_entries': self.lex_table_size * self.n_lex_hashes,
            'lex_bigram_density': sum(
                int(np.count_nonzero(t)) for t in self._lex_bigram_tables
            ) / (self.lex_table_size * self.n_lex_hashes),

            # Architecture
            'n_pos_hashes': self.n_pos_hashes,
            'pos_table_size': self.pos_table_size,
            'n_lex_hashes': self.n_lex_hashes,
            'lex_table_size': self.lex_table_size,
            'pos_weight': self.pos_weight,
            'lex_weight': self.lex_weight,
            'use_trigram': self.use_trigram,
        }

        if self._pos_trigram_tables is not None:
            stats['pos_trigram_nnz'] = sum(
                int(np.count_nonzero(t)) for t in self._pos_trigram_tables
            )
        if self._lex_trigram_tables is not None:
            stats['lex_trigram_nnz'] = sum(
                int(np.count_nonzero(t)) for t in self._lex_trigram_tables
            )

        return stats

    def memory_mb(self) -> float:
        """Estimate memory usage in MB."""
        total = self.pos_table_size * self.n_pos_hashes  # POS bigram
        total += self.lex_table_size * self.n_lex_hashes  # Lex bigram
        if self._pos_trigram_tables is not None:
            total += self._pos_trigram_size * self.n_pos_hashes
        if self._lex_trigram_tables is not None:
            total += self._lex_trigram_size * self.n_lex_hashes
        return total * 4 / (1024 * 1024)  # int32 = 4 bytes

    def get_pos_transition_matrix(self) -> np.ndarray:
        """
        Return the average POS transition energy as a 13x13 matrix.

        This is useful for visualizing what rules the model has learned.
        Lower values = more likely transitions.
        """
        matrix = np.zeros((self.n_pos_types, self.n_pos_types), dtype=np.float64)
        for t1 in range(self.n_pos_types):
            for t2 in range(self.n_pos_types):
                total = 0
                for h_idx in range(self.n_pos_hashes):
                    slot = _double_hash(t1, t2, h_idx, self.pos_table_size)
                    total += self._pos_bigram_tables[h_idx][slot]
                matrix[t1, t2] = total / max(1, self.n_pos_hashes)
        return matrix
