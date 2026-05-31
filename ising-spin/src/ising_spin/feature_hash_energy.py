"""
Feature-Hashed Integer Energy Table with POS generalization and skip-grams.

Three layers of energy signal:

  1. POS tables — category-level RULES that generalize across words
     With 13 POS types, only 169 possible bigram pairs. Training on
     "the cat" (DET->NOUN) automatically improves "a dog" (DET->NOUN).

  2. Lexical tables — token-specific FACTS for fine-grained distinctions
     Same as v77: hash(prev_id, target_id) -> energy.
     Memorizes which specific word pairs are likely.

  3. Skip-gram tables — structural DEPENDENCIES beyond adjacent tokens
     hash(x_{t-2}, x_t) captures subject-verb patterns ("cat...chased").
     hash(POS(x_{t-2}), POS(x_t)) captures DET->VERB patterns.

Combined energy at each generation step:

  E = pos_weight * E_pos(prev_pos, cand_pos)
    + lex_weight * E_lex(prev_id, cand_id)
    + skip_weight * E_skip(context[-2], candidate)
    + tri_weight  * (pos_trigram + lex_trigram)

All tables trained simultaneously via integer NCE:
  Real pair: table[hash] -= eta  (lower energy = more likely)
  Fake pair: table[hash] += eta  (higher energy = less likely)

O(1) per candidate. Pure integer arithmetic. No neural nets.
"""

import numpy as np
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Hash functions — deterministic, collision-resistant, integer-only
# ---------------------------------------------------------------------------

_P1 = np.int64(2654435761)
_P2 = np.int64(2246822519)
_P3 = np.int64(3266489917)
_P4 = np.int64(3367900313)
_MASK32 = np.int64(0xFFFFFFFF)


def _hash2_vec(a: np.ndarray, b: np.ndarray, h_idx: int, P: int) -> np.ndarray:
    """Vectorized double-hash for (a, b) pairs."""
    val = (a.astype(np.int64) * _P1 + b.astype(np.int64) * _P2
           + np.int64(h_idx) * _P3) & _MASK32
    return val % np.int64(P)


def _hash3_vec(a: np.ndarray, b: np.ndarray, c: np.ndarray,
               h_idx: int, P: int) -> np.ndarray:
    """Vectorized triple-hash for (a, b, c) triples."""
    val = (a.astype(np.int64) * _P1 + b.astype(np.int64) * _P2
           + c.astype(np.int64) * _P4 + np.int64(h_idx) * _P3) & _MASK32
    return val % np.int64(P)


def _hash2(a: int, b: int, h_idx: int, P: int) -> int:
    """Scalar double-hash."""
    val = (a * 2654435761 + b * 2246822519 + h_idx * 3266489917) & 0xFFFFFFFF
    return int(val % P)


def _hash3(a: int, b: int, c: int, h_idx: int, P: int) -> int:
    """Scalar triple-hash."""
    val = (a * 2654435761 + b * 2246822519 + c * 3367900313
           + h_idx * 3266489917) & 0xFFFFFFFF
    return int(val % P)


def _next_prime(n: int) -> int:
    """Find the next prime >= n."""
    if n <= 2:
        return 2
    if n % 2 == 0:
        n += 1
    while True:
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                break
        else:
            return n
        n += 2


# ---------------------------------------------------------------------------
# FeatureHashEnergyTable
# ---------------------------------------------------------------------------

class FeatureHashEnergyTable:
    """
    Feature-Hashed Integer Energy Table.

    Three signal layers:
      1. POS tables: category-level rules (generalize across words)
      2. Lexical tables: token-specific facts
      3. Skip-gram tables: structural dependencies (skip one token)

    All trained via integer NCE. All O(1) per candidate lookup.
    """

    def __init__(
        self,
        vocab_size: int,
        word_pos: np.ndarray,
        n_pos_types: int = 13,
        # POS tables
        n_pos_hashes: int = 2,
        pos_table_size: int = 1009,
        pos_eta: int = 3,
        pos_clip: int = 500,
        # Lexical tables
        n_lex_hashes: int = 3,
        lex_table_size: int = 65537,
        lex_eta: int = 1,
        lex_clip: int = 1000,
        # Skip-gram tables
        use_skip: bool = True,
        n_skip_hashes: int = 2,
        skip_table_size: int = 65537,
        skip_eta: int = 1,
        skip_clip: int = 800,
        # Trigram
        use_trigram: bool = True,
        trigram_weight: int = 1,
        # Energy combination weights
        pos_weight: float = 1.0,
        lex_weight: float = 1.0,
        skip_weight: float = 0.5,
        # Seed
        seed: int = 42,
    ):
        self.V = vocab_size
        self.word_pos = word_pos.astype(np.int32)
        self.n_pos_types = n_pos_types

        self.n_pos_hashes = n_pos_hashes
        self.pos_table_size = pos_table_size
        self.pos_eta = pos_eta
        self.pos_clip = pos_clip

        self.n_lex_hashes = n_lex_hashes
        self.lex_table_size = lex_table_size
        self.lex_eta = lex_eta
        self.lex_clip = lex_clip

        self.use_skip = use_skip
        self.n_skip_hashes = n_skip_hashes
        self.skip_table_size = skip_table_size
        self.skip_eta = skip_eta
        self.skip_clip = skip_clip

        self.use_trigram = use_trigram
        self.trigram_weight = trigram_weight

        self.pos_weight = pos_weight
        self.lex_weight = lex_weight
        self.skip_weight = skip_weight
        self.seed = seed

        # POS bigram + trigram tables
        self._pos_bi = [np.zeros(pos_table_size, dtype=np.int32) for _ in range(n_pos_hashes)]
        self._pos_tri = None
        self._pos_tri_size = 0
        if use_trigram:
            self._pos_tri_size = _next_prime(pos_table_size + 256)
            self._pos_tri = [np.zeros(self._pos_tri_size, dtype=np.int32) for _ in range(n_pos_hashes)]

        # Lexical bigram + trigram tables
        self._lex_bi = [np.zeros(lex_table_size, dtype=np.int32) for _ in range(n_lex_hashes)]
        self._lex_tri = None
        self._lex_tri_size = 0
        if use_trigram:
            self._lex_tri_size = _next_prime(lex_table_size + 256)
            self._lex_tri = [np.zeros(self._lex_tri_size, dtype=np.int32) for _ in range(n_lex_hashes)]

        # Skip-gram tables (hash context[-2], candidate)
        self._skip_bi = None
        self._skip_pos_bi = None
        if use_skip:
            self._skip_bi = [np.zeros(skip_table_size, dtype=np.int32) for _ in range(n_skip_hashes)]
            self._skip_pos_bi = [np.zeros(pos_table_size, dtype=np.int32) for _ in range(n_skip_hashes)]

    # -------------------------------------------------------------------
    # Energy computation — the hot path
    # -------------------------------------------------------------------

    def compute_local_energy_batch(
        self,
        context_word_ids: List[int],
        candidates: np.ndarray,
    ) -> np.ndarray:
        """
        Compute total local energy for all candidates given context.

        E = pos_weight * E_pos + lex_weight * E_lex + skip_weight * E_skip

        Returns integer energy array. Lower = more likely = better.
        """
        K = len(candidates)
        if len(context_word_ids) == 0:
            return np.zeros(K, dtype=np.int64)

        prev = context_word_ids[-1]
        prev_pos = int(self.word_pos[prev])
        cand_pos = self.word_pos[candidates].astype(np.int64)

        # --- POS bigram ---
        pos_e = np.zeros(K, dtype=np.int64)
        prev_pos_arr = np.full(K, prev_pos, dtype=np.int64)
        for h in range(self.n_pos_hashes):
            slots = _hash2_vec(prev_pos_arr, cand_pos, h, self.pos_table_size)
            pos_e += self._pos_bi[h][slots]

        # --- Lexical bigram ---
        lex_e = np.zeros(K, dtype=np.int64)
        prev_arr = np.full(K, prev, dtype=np.int64)
        for h in range(self.n_lex_hashes):
            slots = _hash2_vec(prev_arr, candidates.astype(np.int64), h, self.lex_table_size)
            lex_e += self._lex_bi[h][slots]

        # --- Skip-gram (context[-2], candidate) ---
        skip_e = np.zeros(K, dtype=np.int64)
        if self.use_skip and self._skip_bi is not None and len(context_word_ids) >= 2:
            prev2 = context_word_ids[-2]
            prev2_pos = int(self.word_pos[prev2])
            prev2_arr = np.full(K, prev2, dtype=np.int64)
            prev2_pos_arr = np.full(K, prev2_pos, dtype=np.int64)

            for h in range(self.n_skip_hashes):
                slots = _hash2_vec(prev2_arr, candidates.astype(np.int64), h, self.skip_table_size)
                skip_e += self._skip_bi[h][slots]
                # POS skip
                slots_pos = _hash2_vec(prev2_pos_arr, cand_pos, h, self.pos_table_size)
                skip_e += self._skip_pos_bi[h][slots_pos]

        # --- Trigrams ---
        if self.use_trigram and len(context_word_ids) >= 2:
            prev2 = context_word_ids[-2]
            prev2_pos = int(self.word_pos[prev2])

            # POS trigram
            if self._pos_tri is not None:
                prev2_pos_arr = np.full(K, prev2_pos, dtype=np.int64)
                for h in range(self.n_pos_hashes):
                    slots = _hash3_vec(prev2_pos_arr, prev_pos_arr, cand_pos, h, self._pos_tri_size)
                    pos_e += self.trigram_weight * self._pos_tri[h][slots]

            # Lexical trigram
            if self._lex_tri is not None:
                prev2_arr = np.full(K, prev2, dtype=np.int64)
                for h in range(self.n_lex_hashes):
                    slots = _hash3_vec(prev2_arr, prev_arr, candidates.astype(np.int64), h, self._lex_tri_size)
                    lex_e += self.trigram_weight * self._lex_tri[h][slots]

        # --- Weighted combination ---
        combined = (self.pos_weight * pos_e.astype(np.float64)
                    + self.lex_weight * lex_e.astype(np.float64)
                    + self.skip_weight * skip_e.astype(np.float64))
        return combined.astype(np.int64)

    def compute_local_energy(
        self,
        context_word_ids: List[int],
        candidate: int,
    ) -> int:
        """Scalar version for single candidate."""
        if len(context_word_ids) == 0:
            return 0

        prev = context_word_ids[-1]
        prev_pos = int(self.word_pos[prev])
        cand_pos = int(self.word_pos[candidate]) if candidate < self.V else 0

        # POS bigram
        pos_e = sum(int(self._pos_bi[h][_hash2(prev_pos, cand_pos, h, self.pos_table_size)])
                    for h in range(self.n_pos_hashes))

        # Lexical bigram
        lex_e = sum(int(self._lex_bi[h][_hash2(prev, candidate, h, self.lex_table_size)])
                    for h in range(self.n_lex_hashes))

        # Skip-gram
        skip_e = 0
        if self.use_skip and self._skip_bi is not None and len(context_word_ids) >= 2:
            prev2 = context_word_ids[-2]
            prev2_pos = int(self.word_pos[prev2])
            for h in range(self.n_skip_hashes):
                skip_e += int(self._skip_bi[h][_hash2(prev2, candidate, h, self.skip_table_size)])
                skip_e += int(self._skip_pos_bi[h][_hash2(prev2_pos, cand_pos, h, self.pos_table_size)])

        # Trigrams
        if self.use_trigram and len(context_word_ids) >= 2:
            prev2 = context_word_ids[-2]
            prev2_pos = int(self.word_pos[prev2])
            if self._pos_tri is not None:
                for h in range(self.n_pos_hashes):
                    pos_e += self.trigram_weight * int(self._pos_tri[h][_hash3(prev2_pos, prev_pos, cand_pos, h, self._pos_tri_size)])
            if self._lex_tri is not None:
                for h in range(self.n_lex_hashes):
                    lex_e += self.trigram_weight * int(self._lex_tri[h][_hash3(prev2, prev, candidate, h, self._lex_tri_size)])

        return int(self.pos_weight * pos_e + self.lex_weight * lex_e + self.skip_weight * skip_e)

    # -------------------------------------------------------------------
    # NCE Training — vectorized batch integer updates
    # -------------------------------------------------------------------

    def train_nce(
        self,
        sequences: List[List[int]],
        n_epochs: int = 3,
        n_negatives: int = 3,
        callback=None,
    ) -> Dict:
        """
        Train all energy tables via vectorized integer NCE.

        For each (prev, target) pair:
          Positive: table[hash(real_pair)] -= eta
          Negative: table[hash(fake_pair)] += eta
        """
        import time as _time
        rng = np.random.RandomState(self.seed)

        # Precompute all pairs
        all_prev, all_target, all_prev2 = [], [], []
        for seq in sequences:
            if len(seq) < 2:
                continue
            for pos in range(1, len(seq)):
                all_prev.append(seq[pos - 1])
                all_target.append(seq[pos])
                all_prev2.append(seq[pos - 2] if pos >= 2 else 0)

        all_prev = np.array(all_prev, dtype=np.int64)
        all_target = np.array(all_target, dtype=np.int64)
        all_prev2 = np.array(all_prev2, dtype=np.int64)
        N = len(all_prev)

        all_prev_pos = self.word_pos[all_prev].astype(np.int64)
        all_target_pos = self.word_pos[all_target].astype(np.int64)
        all_prev2_pos = self.word_pos[all_prev2].astype(np.int64)

        print(f"    {N:,} training pairs, {self.n_pos_types}x{self.n_pos_types}={self.n_pos_types**2} POS patterns", flush=True)

        all_stats = []

        for epoch in range(n_epochs):
            t0 = _time.time()

            order = rng.permutation(N)
            sp = all_prev[order]
            st = all_target[order]
            sp2 = all_prev2[order]
            sp_pos = all_prev_pos[order]
            st_pos = all_target_pos[order]
            sp2_pos = all_prev2_pos[order]

            chunk = 100000
            for c0 in range(0, N, chunk):
                c1 = min(c0 + chunk, N)
                cp = sp[c0:c1]; ct = st[c0:c1]; cp2 = sp2[c0:c1]
                cpp = sp_pos[c0:c1]; ctp = st_pos[c0:c1]; cp2p = sp2_pos[c0:c1]
                C = len(cp)

                # Positive updates
                for h in range(self.n_pos_hashes):
                    np.add.at(self._pos_bi[h], _hash2_vec(cpp, ctp, h, self.pos_table_size), -self.pos_eta)
                for h in range(self.n_lex_hashes):
                    np.add.at(self._lex_bi[h], _hash2_vec(cp, ct, h, self.lex_table_size), -self.lex_eta)

                # Skip-gram positive (context[-2], target)
                if self.use_skip and self._skip_bi is not None:
                    has_p2 = cp2 > 0
                    if np.any(has_p2):
                        cp2v = cp2[has_p2]; ctv = ct[has_p2]; cp2pv = cp2p[has_p2]; ctpv = ctp[has_p2]
                        for h in range(self.n_skip_hashes):
                            np.add.at(self._skip_bi[h], _hash2_vec(cp2v, ctv, h, self.skip_table_size), -self.skip_eta)
                            np.add.at(self._skip_pos_bi[h], _hash2_vec(cp2pv, ctpv, h, self.pos_table_size), -self.skip_eta)

                # Negative updates
                for _ in range(n_negatives):
                    neg = rng.randint(4, self.V, size=C)
                    neg_pos = self.word_pos[neg].astype(np.int64)

                    for h in range(self.n_pos_hashes):
                        np.add.at(self._pos_bi[h], _hash2_vec(cpp, neg_pos, h, self.pos_table_size), self.pos_eta)
                    for h in range(self.n_lex_hashes):
                        np.add.at(self._lex_bi[h], _hash2_vec(cp, neg, h, self.lex_table_size), self.lex_eta)

                    if self.use_skip and self._skip_bi is not None and np.any(has_p2 if C > 0 else False):
                        for h in range(self.n_skip_hashes):
                            np.add.at(self._skip_bi[h], _hash2_vec(cp2v, neg[has_p2] if C == len(neg) else rng.randint(4, self.V, size=len(cp2v)), h, self.skip_table_size), self.skip_eta)

                # Trigram updates
                if self.use_trigram:
                    has_p2 = cp2 > 0
                    if np.any(has_p2):
                        cp2v = cp2[has_p2]; cpv = cp[has_p2]; ctv = ct[has_p2]
                        cp2pv = cp2p[has_p2]; cppv = cpp[has_p2]; ctpv = ctp[has_p2]
                        if self._pos_tri is not None:
                            for h in range(self.n_pos_hashes):
                                np.add.at(self._pos_tri[h], _hash3_vec(cp2pv, cppv, ctpv, h, self._pos_tri_size), -self.pos_eta)
                        if self._lex_tri is not None:
                            for h in range(self.n_lex_hashes):
                                np.add.at(self._lex_tri[h], _hash3_vec(cp2v, cpv, ctv, h, self._lex_tri_size), -self.lex_eta)

                        for _ in range(n_negatives):
                            neg_t = rng.randint(4, self.V, size=len(cp2v))
                            neg_tp = self.word_pos[neg_t].astype(np.int64)
                            if self._pos_tri is not None:
                                for h in range(self.n_pos_hashes):
                                    np.add.at(self._pos_tri[h], _hash3_vec(cp2pv, cppv, neg_tp, h, self._pos_tri_size), self.pos_eta)
                            if self._lex_tri is not None:
                                for h in range(self.n_lex_hashes):
                                    np.add.at(self._lex_tri[h], _hash3_vec(cp2v, cpv, neg_t, h, self._lex_tri_size), self.lex_eta)

            t_elapsed = _time.time() - t0

            # Clip
            self._clip_tables()

            # Discriminative accuracy
            n_check = min(2000, N)
            ci = rng.choice(N, n_check, replace=False)
            cp = all_prev[ci]; ct = all_target[ci]; cpp = all_prev_pos[ci]; ctp = all_target_pos[ci]
            neg = rng.randint(4, self.V, size=n_check)
            neg_pos = self.word_pos[neg].astype(np.int64)

            pos_e = sum(self._pos_bi[h][_hash2_vec(cpp, ctp, h, self.pos_table_size)] for h in range(self.n_pos_hashes))
            neg_e = sum(self._pos_bi[h][_hash2_vec(cpp, neg_pos, h, self.pos_table_size)] for h in range(self.n_pos_hashes))
            pos_d = float(np.sum(pos_e < neg_e)) / n_check

            lex_pe = sum(self._lex_bi[h][_hash2_vec(cp, ct, h, self.lex_table_size)] for h in range(self.n_lex_hashes))
            lex_ne = sum(self._lex_bi[h][_hash2_vec(cp, neg, h, self.lex_table_size)] for h in range(self.n_lex_hashes))
            lex_d = float(np.sum(lex_pe < lex_ne)) / n_check

            comb_real = self.pos_weight * pos_e.astype(np.float64) + self.lex_weight * lex_pe.astype(np.float64)
            comb_neg = self.pos_weight * neg_e.astype(np.float64) + self.lex_weight * lex_ne.astype(np.float64)
            comb_d = float(np.sum(comb_real < comb_neg)) / n_check

            stats = {
                'epoch': epoch,
                'pos_disc': pos_d,
                'lex_disc': lex_d,
                'combined_disc': comb_d,
                'time_s': t_elapsed,
            }
            all_stats.append(stats)

            print(f"    Epoch {epoch+1}/{n_epochs}: "
                  f"pos_disc={pos_d:.3f}, lex_disc={lex_d:.3f}, "
                  f"combined={comb_d:.3f}, time={t_elapsed:.1f}s", flush=True)

            # Show top POS rules
            rules = self._top_pos_rules(5)
            print(f"      Top rules (low=likely):", flush=True)
            for r in rules:
                print(f"        {r['from']} -> {r['to']}: {r['energy']:.0f}", flush=True)

            if callback:
                callback(epoch, stats)

        return {'epochs': all_stats}

    def _clip_tables(self):
        """Clip all table values to prevent energy explosion."""
        for h in range(self.n_pos_hashes):
            np.clip(self._pos_bi[h], -self.pos_clip, self.pos_clip, out=self._pos_bi[h])
        for h in range(self.n_lex_hashes):
            np.clip(self._lex_bi[h], -self.lex_clip, self.lex_clip, out=self._lex_bi[h])
        if self._pos_tri is not None:
            for h in range(self.n_pos_hashes):
                np.clip(self._pos_tri[h], -self.pos_clip, self.pos_clip, out=self._pos_tri[h])
        if self._lex_tri is not None:
            for h in range(self.n_lex_hashes):
                np.clip(self._lex_tri[h], -self.lex_clip, self.lex_clip, out=self._lex_tri[h])
        if self._skip_bi is not None:
            for h in range(self.n_skip_hashes):
                np.clip(self._skip_bi[h], -self.skip_clip, self.skip_clip, out=self._skip_bi[h])
                np.clip(self._skip_pos_bi[h], -self.skip_clip, self.skip_clip, out=self._skip_pos_bi[h])

    def _top_pos_rules(self, k: int = 5) -> List[Dict]:
        """Return the k most and least likely POS transitions."""
        from .vocabulary import IDX2POS
        pairs = {}
        for t1 in range(self.n_pos_types):
            for t2 in range(self.n_pos_types):
                total = sum(int(self._pos_bi[h][_hash2(t1, t2, h, self.pos_table_size)])
                            for h in range(self.n_pos_hashes))
                pairs[(IDX2POS.get(t1, "X"), IDX2POS.get(t2, "X"))] = total / max(1, self.n_pos_hashes)

        sorted_p = sorted(pairs.items(), key=lambda x: x[1])
        rules = [{'from': p[0], 'to': p[1], 'energy': e} for p, e in sorted_p[:k]]
        rules += [{'from': p[0], 'to': p[1], 'energy': e} for p, e in sorted_p[-k:]]
        return rules

    def get_pos_matrix(self) -> np.ndarray:
        """Return 13x13 POS transition energy matrix for visualization."""
        matrix = np.zeros((self.n_pos_types, self.n_pos_types), dtype=np.float64)
        for t1 in range(self.n_pos_types):
            for t2 in range(self.n_pos_types):
                total = sum(int(self._pos_bi[h][_hash2(t1, t2, h, self.pos_table_size)])
                            for h in range(self.n_pos_hashes))
                matrix[t1, t2] = total / max(1, self.n_pos_hashes)
        return matrix

    def memory_mb(self) -> float:
        """Estimate memory usage in MB."""
        total = (self.pos_table_size * self.n_pos_hashes +
                 self.lex_table_size * self.n_lex_hashes +
                 self._pos_tri_size * self.n_pos_hashes +
                 self._lex_tri_size * self.n_lex_hashes)
        if self._skip_bi is not None:
            total += self.skip_table_size * self.n_skip_hashes * 2  # skip_bi + skip_pos_bi
        return total * 4 / (1024 * 1024)

    def statistics(self) -> Dict:
        """Return table statistics."""
        return {
            'pos_nnz': sum(int(np.count_nonzero(t)) for t in self._pos_bi),
            'lex_nnz': sum(int(np.count_nonzero(t)) for t in self._lex_bi),
            'pos_range': (int(min(t.min() for t in self._pos_bi)),
                          int(max(t.max() for t in self._pos_bi))),
            'lex_range': (int(min(t.min() for t in self._lex_bi)),
                          int(max(t.max() for t in self._lex_bi))),
            'memory_mb': self.memory_mb(),
            'pos_weight': self.pos_weight,
            'lex_weight': self.lex_weight,
            'skip_weight': self.skip_weight,
        }
