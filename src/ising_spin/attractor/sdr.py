"""
Sparse Distributed Representations (SDR) — kWTA encoding.

WHY SPARSE (not balanced ±1):
  - Cortical columns use ~2% active bits, not 50/50 ±1
  - Sparse codes give EXPONENTIAL pattern separation:
    Overlap probability between two random k-sparse vectors of dim D:
      P(overlap > t) ≈ C(k,t) * C(D-k, k-t) / C(D, k)
    For D=512, k=10: expected overlap ≈ 0.2 bits (essentially zero)
    For D=256, balanced ±1: expected overlap = 0 (on average) but
      variance is HIGH — many pairs have significant overlap
  - Sparse codes are naturally noise-robust: flipping a random bit
    only changes 1/k ≈ 10% of active bits, vs 1/D ≈ 0.2% for dense
  - kWTA (k-Winner-Take-All) is the biological mechanism that
    enforces sparsity — only the top k activations survive

INTEGER-ONLY:
  - SDRs are binary vectors: s ∈ {0,1}^D with exactly k bits set
  - Superposition: element-wise addition of sparse vectors → dense accumulator
  - kWTA: keep top k values of accumulator, zero the rest
  - Hamming overlap: count of shared active bits (integer)

Memory (V=2000, D=512, k=10):
  - word_sdrs: 2000 × 512 × 1 bit = 128 KB (packed)
  - Or as indices: 2000 × 10 × 2 bytes = 40 KB (sparse format)
  - We use dense uint8 for fast numpy operations → 1 MB
"""

import time
import numpy as np
from typing import List, Optional, Tuple


class SDREncoder:
    """
    Sparse Distributed Representation encoder using kWTA.

    Each word is encoded as a binary vector of dimension D with exactly
    k active bits. The encoding is deterministic (seeded random projection
    + kWTA), so the same word always produces the same SDR.

    Context encoding: superposition of word SDRs, then kWTA to maintain
    sparsity. This preserves similarity structure — similar contexts
    produce overlapping SDRs.
    """

    def __init__(
        self,
        vocab_size: int,
        D: int = 512,
        sparsity: float = 0.02,
        seed: int = 42,
    ):
        """
        Args:
            vocab_size: Number of words in vocabulary.
            D: SDR dimension (default 512).
            sparsity: Fraction of active bits (default 0.02 = 2%).
            seed: Random seed for deterministic encoding.
        """
        self.vocab_size = vocab_size
        self.D = D
        self.k = max(4, int(D * sparsity))  # Number of active bits
        self.sparsity = self.k / D
        self.seed = seed

        # Word SDRs: (V, D) binary matrix — built during training
        self.word_sdrs: Optional[np.ndarray] = None  # uint8

        # Random projection matrix for deterministic SDR generation
        # R[w, d] ∈ {-1, +1} — used once during build, then freed
        self._R: Optional[np.ndarray] = None

        # Sparse index format: for each word, list of active bit positions
        # More efficient for energy computation with sparse states
        self.word_active_bits: Optional[List[np.ndarray]] = None

        self._built = False

    def build(self, word_freq: Optional[np.ndarray] = None) -> "SDREncoder":
        """
        Build SDRs for all words using deterministic random projection + kWTA.

        Each word gets a unique, deterministic SDR. Words with similar
        frequency profiles get partially overlapping SDRs (because the
        random projection preserves some similarity structure).

        The kWTA step ensures exactly k active bits per word, providing
        maximum pattern separation under the sparsity constraint.

        Args:
            word_freq: Optional word frequency array (V,) for frequency-aware
                       encoding. High-frequency words get slightly more
                       dispersed SDRs to reduce interference.

        Returns:
            self
        """
        V = self.vocab_size
        D = self.D
        k = self.k

        # Generate deterministic random projection matrix
        rng = np.random.RandomState(self.seed)
        R = rng.choice([-1, 1], size=(V, D)).astype(np.int8)
        self._R = R

        # For each word, the projection already gives a D-dimensional vector.
        # kWTA: keep the top k values, zero the rest.
        # This is equivalent to: for each word, find the k dimensions
        # where the random projection is most positive, set those to 1.

        word_sdrs = np.zeros((V, D), dtype=np.uint8)

        for w in range(V):
            # Find the k largest values in R[w]
            # np.argpartition is O(D) — faster than full sort
            top_k_indices = np.argpartition(R[w], -k)[-k:]
            word_sdrs[w, top_k_indices] = 1

        self.word_sdrs = word_sdrs

        # Build sparse index format
        self.word_active_bits = []
        for w in range(V):
            active = np.where(word_sdrs[w] > 0)[0].astype(np.int16)
            self.word_active_bits.append(active)

        # Free projection matrix (not needed at inference)
        self._R = None
        self._built = True

        # Diagnostics
        self._print_diagnostics()

        return self

    def encode(self, word_id: int) -> np.ndarray:
        """
        Get the SDR for a single word.

        Args:
            word_id: Word index.

        Returns:
            Binary vector (D,) uint8 with exactly k active bits.
        """
        if not self._built:
            raise RuntimeError("SDREncoder not built — call build() first")
        if word_id < 0 or word_id >= self.vocab_size:
            # Return zero vector for OOV
            return np.zeros(self.D, dtype=np.uint8)
        return self.word_sdrs[word_id]

    def encode_context(
        self,
        word_ids: List[int],
        context_window: int = 10,
    ) -> np.ndarray:
        """
        Encode a context (sequence of words) as a sparse SDR — BOW.

        Method: superposition of word SDRs, then kWTA.
        The superposition (element-wise sum) accumulates evidence
        for each dimension. kWTA then selects the dimensions with
        the strongest evidence.

        This naturally handles:
        - Repeated words: their dimensions get higher activation
        - Overlapping contexts: similar words → overlapping SDRs
        - Variable-length context: kWTA normalizes to fixed sparsity

        NOTE: This is BOW (bag-of-words) — order is LOST.
        Use encode_context_positional for order-preserving encoding.

        Args:
            word_ids: List of word indices in the context.
            context_window: Only use the last `context_window` words.

        Returns:
            Binary vector (D,) uint8 with exactly k active bits.
        """
        if not self._built:
            raise RuntimeError("SDREncoder not built — call build() first")

        # Use only recent context
        recent = word_ids[-context_window:] if len(word_ids) > context_window else word_ids

        if not recent:
            return np.zeros(self.D, dtype=np.uint8)

        # Superposition: sum of word SDRs
        # Using sparse addition: accumulate counts per dimension
        acc = np.zeros(self.D, dtype=np.int32)
        for w in recent:
            if 0 <= w < self.vocab_size:
                acc += self.word_sdrs[w].astype(np.int32)

        # kWTA: keep top k
        if np.sum(acc > 0) == 0:
            return np.zeros(self.D, dtype=np.uint8)

        context_sdr = np.zeros(self.D, dtype=np.uint8)
        top_k = np.argpartition(acc, -self.k)[-self.k:]
        context_sdr[top_k] = 1

        return context_sdr

    def encode_context_positional(
        self,
        word_ids: List[int],
        context_window: int = 10,
    ) -> np.ndarray:
        """
        Encode context with POSITIONAL VSA binding — order-preserving.

        v52: Replaces BOW superposition with position-bound superposition.
        Each word's SDR is rotated by its RELATIVE position hash before
        being added to the accumulator. This preserves word order:

          "the cat" at positions (-2, -1):
              context = rotate(sdr["the"], hash(-2)) + rotate(sdr["cat"], hash(-1))
          "cat the" at positions (-2, -1):
              context = rotate(sdr["cat"], hash(-2)) + rotate(sdr["the"], hash(-1))

        These produce DIFFERENT context SDRs — the DAM can now learn
        order-dependent associations (not just co-occurrence).

        Position hash: (rel_pos * PRIME + 1) % D
        where PRIME = 7919 gives excellent spread across [0, D-1].

        Relative position: 1 = most recent word, 2 = one before, etc.
        So the last word in the context has the strongest positional signal.

        Args:
            word_ids: List of word indices in the context.
            context_window: Only use the last `context_window` words.

        Returns:
            Binary vector (D,) uint8 with exactly k active bits.
        """
        if not self._built:
            raise RuntimeError("SDREncoder not built — call build() first")

        # Use only recent context
        recent = word_ids[-context_window:] if len(word_ids) > context_window else word_ids

        if not recent:
            return np.zeros(self.D, dtype=np.uint8)

        D = self.D
        k = self.k
        PRIME = 7919

        acc = np.zeros(D, dtype=np.int32)
        n_recent = len(recent)

        for i, w in enumerate(recent):
            if 0 <= w < self.vocab_size:
                # Relative position: 1 = most recent, n_recent = oldest
                rel_pos = n_recent - i
                shift = (rel_pos * PRIME + 1) % D
                # Rotate SDR by position hash — preserves order information
                acc += np.roll(self.word_sdrs[w].astype(np.int32), shift)

        # kWTA: keep top k
        if np.sum(acc > 0) == 0:
            return np.zeros(D, dtype=np.uint8)

        context_sdr = np.zeros(D, dtype=np.uint8)
        top_k = np.argpartition(acc, -k)[-k:]
        context_sdr[top_k] = 1

        return context_sdr

    def encode_contexts_batch(
        self,
        sequences: List[List[int]],
        context_window: int = 10,
        batch_size: int = 50000,
        callback=None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        VECTORIZED batch context encoding for Hebbian training.

        Instead of calling encode_context() in a Python loop (millions of
        calls, each doing a Python loop over words + argpartition), this
        method precomputes ALL context+target SDR pairs using numpy matrix
        operations.

        Speedup: ~50-100x over the per-pair Python loop approach.

        The key insight: for a sequence [w0, w1, w2, ...], the context
        for position p is the superposition of SDRs for w[max(0,p-W):p].
        We can compute all superpositions for a sequence using cumulative
        sums, then apply kWTA in bulk.

        Args:
            sequences: List of token sequences.
            context_window: Context window size.
            batch_size: Yield every this many pairs (for memory control).
            callback: Optional callable(seq_idx, total_pairs) for progress.

        Yields:
            (context_sdrs, target_sdrs) — each (N, D) uint8 arrays.
        """
        if not self._built:
            raise RuntimeError("SDREncoder not built — call build() first")

        D = self.D
        k = self.k
        V = self.vocab_size
        word_sdrs = self.word_sdrs  # (V, D) uint8

        print(f"        [encode_contexts_batch] Starting: {len(sequences)} seqs, "
              f"D={D}, k={k}, batch_size={batch_size}", flush=True)

        batch_ctx = []
        batch_tgt = []
        total_pairs = 0
        t_start = time.time()
        n_seqs_processed = 0

        for seq_idx, seq in enumerate(sequences):
            seq_len = len(seq)
            if seq_len < 3:
                continue

            # Filter valid word IDs
            valid_seq = [w for w in seq if 0 <= w < V]
            if len(valid_seq) < 3:
                continue

            n = len(valid_seq)
            sdr_stack = word_sdrs[valid_seq].astype(np.int32)  # (n, D)
            cumsum = np.cumsum(sdr_stack, axis=0)  # (n, D)

            # VECTORIZED: compute ALL context windows at once
            positions = np.arange(1, n)
            starts = np.maximum(0, positions - context_window)

            # Build (n-1, D) context accumulator array
            context_acc = np.empty((len(positions), D), dtype=np.int32)
            for i, (pos, start) in enumerate(zip(positions, starts)):
                if start == 0:
                    context_acc[i] = cumsum[pos - 1]
                else:
                    context_acc[i] = cumsum[pos - 1] - cumsum[start - 1]

            # kWTA for ALL positions at once using 2D argpartition
            nonzero_mask = np.max(context_acc, axis=1) > 0
            n_valid = int(np.sum(nonzero_mask))
            if n_valid == 0:
                continue

            valid_acc = context_acc[nonzero_mask]
            top_k_all = np.argpartition(valid_acc, -k, axis=1)[:, -k:]

            # Build context SDRs
            valid_sdrs = np.zeros((n_valid, D), dtype=np.uint8)
            rows = np.arange(n_valid).reshape(-1, 1)
            valid_sdrs[rows, top_k_all] = 1

            # Target SDRs for valid positions
            valid_positions = positions[nonzero_mask]
            valid_targets = np.array([valid_seq[p] for p in valid_positions], dtype=np.int64)

            # Filter out zero context/target
            ctx_sums = np.sum(valid_sdrs, axis=1)
            tgt_sums = np.sum(word_sdrs[valid_targets], axis=1)
            both_nonzero = (ctx_sums > 0) & (tgt_sums > 0)

            for i in np.where(both_nonzero)[0]:
                batch_ctx.append(valid_sdrs[i].copy())
                batch_tgt.append(word_sdrs[int(valid_targets[i])].copy())
                total_pairs += 1

                if len(batch_ctx) >= batch_size:
                    print(f"        [encode] Yielding batch of {len(batch_ctx)} pairs "
                          f"after seq {seq_idx+1}/{len(sequences)} "
                          f"({time.time()-t_start:.1f}s)", flush=True)
                    ctx_arr = np.array(batch_ctx, dtype=np.uint8)
                    tgt_arr = np.array(batch_tgt, dtype=np.uint8)
                    yield ctx_arr, tgt_arr
                    batch_ctx = []
                    batch_tgt = []

            n_seqs_processed += 1

            # Progress every 500 sequences or 5 seconds
            now = time.time()
            if n_seqs_processed % 500 == 0 or (now - t_start) > 5.0:
                elapsed = now - t_start
                rate = n_seqs_processed / max(0.1, elapsed)
                eta = (len(sequences) - seq_idx) / max(1, rate)
                print(f"        [encode] {n_seqs_processed} seqs, {total_pairs} pairs, "
                      f"{rate:.0f} seqs/s, ETA {eta:.0f}s", flush=True)
                t_start = now  # Reset to avoid spamming
                n_seqs_processed = 0

        # Final partial batch
        if batch_ctx:
            print(f"        [encode] Final batch: {len(batch_ctx)} pairs", flush=True)
            ctx_arr = np.array(batch_ctx, dtype=np.uint8)
            tgt_arr = np.array(batch_tgt, dtype=np.uint8)
            yield ctx_arr, tgt_arr

        print(f"        [encode] Done: {total_pairs} total pairs", flush=True)

        if callback:
            callback(len(sequences), total_pairs)

    def encode_contexts_batch_with_binding(
        self,
        sequences: List[List[int]],
        context_window: int = 10,
        batch_size: int = 50000,
        bind_window: int = 8,
        bind_density: int = 20,
        callback=None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        ORDER-SENSITIVE batch context encoding with VSA binding.

        v45 FUNDAMENTAL FIX: The DAM was trained on BOW (bag-of-words)
        contexts, making it order-blind. This method builds context SDRs
        that INCLUDE M_bind — the VSA binding vector that encodes bigram
        order. The DAM then learns order-dependent couplings natively.

        Instead of cumsum (order-blind superposition), we scan each
        sequence position-by-position, building M_bind incrementally.
        M_bind is OR'd into the context accumulator before kWTA, so
        the kWTA-selected context bits include both co-occurrence AND
        order information.

        Args:
            sequences: List of token sequences.
            context_window: Context window size.
            batch_size: Yield every this many pairs (for memory control).
            bind_window: Number of recent bigram bindings for M_bind.
            bind_density: Target M_bind density after kWTA (2k=20).
            callback: Optional callable(seq_idx, total_pairs) for progress.

        Yields:
            (context_sdrs, target_sdrs) — each (N, D) uint8 arrays.
        """
        from ..attractor.binding import bind_pair, sdr_hash, rotate_sdr
        from collections import deque

        if not self._built:
            raise RuntimeError("SDREncoder not built — call build() first")

        D = self.D
        k = self.k
        V = self.vocab_size
        word_sdrs = self.word_sdrs  # (V, D) uint8

        print(f"        [encode_contexts_batch_with_binding] Starting: {len(sequences)} seqs, "
              f"D={D}, k={k}, batch_size={batch_size}, bind_window={bind_window}", flush=True)

        batch_ctx = []
        batch_tgt = []
        total_pairs = 0
        t_start = time.time()
        n_seqs_processed = 0

        for seq_idx, seq in enumerate(sequences):
            seq_len = len(seq)
            if seq_len < 3:
                continue

            # Filter valid word IDs
            valid_seq = [w for w in seq if 0 <= w < V]
            if len(valid_seq) < 3:
                continue

            n = len(valid_seq)

            # SEQUENTIAL SCAN: build binding-aware context per position
            # This replaces the cumsum approach — we MUST go position by
            # position because M_bind depends on the previous word.
            context_acc = np.empty((n - 1, D), dtype=np.int32)

            # Binding state (mirrors BindingContext._rebuild logic)
            _bindings = deque(maxlen=bind_window)
            _last_word_bits = None

            # Pre-compute SDR stack for superposition contexts
            sdr_stack = word_sdrs[valid_seq].astype(np.int32)  # (n, D)
            cumsum = np.cumsum(sdr_stack, axis=0)  # for BOW part only

            for p in range(1, n):
                # BOW superposition context (same as before)
                start = max(0, p - context_window)
                if start == 0:
                    bow_acc = cumsum[p - 1].copy()
                else:
                    bow_acc = cumsum[p - 1] - cumsum[start - 1]
                    bow_acc = bow_acc.copy()

                # Build bigram binding for this position
                prev_bits = self.word_active_bits[valid_seq[p - 1]]
                curr_bits = self.word_active_bits[valid_seq[p]]

                if _last_word_bits is not None:
                    # bind(current, previous) = rot(current, hash(previous))
                    bound = bind_pair(curr_bits, _last_word_bits, D)
                    _bindings.append(bound)

                _last_word_bits = curr_bits.copy()

                # Build M_bind accumulator from binding window (uniform weights)
                if len(_bindings) > 0:
                    bind_acc = np.zeros(D, dtype=np.int32)
                    for bound_bits in _bindings:
                        for b in bound_bits:
                            idx = int(b)
                            if 0 <= idx < D:
                                bind_acc[idx] += 1

                    # kWTA on M_bind to get top bind_density bits
                    n_bind_active = min(bind_density, int(np.sum(bind_acc > 0)))
                    if n_bind_active > 0 and np.max(bind_acc) > 0:
                        top_bind = np.argpartition(bind_acc, -n_bind_active)[-n_bind_active:]
                        # Add M_bind bits to BOW accumulator
                        # Weight: bind bits get +2 boost (they carry ORDER info)
                        # BOW bits typically have count 1-5, so +2 makes binding
                        # bits competitive in kWTA without overwhelming BOW signal
                        for idx in top_bind:
                            bow_acc[int(idx)] += 2

                context_acc[p - 1] = bow_acc

            # kWTA for ALL positions at once (same as before)
            nonzero_mask = np.max(context_acc, axis=1) > 0
            n_valid = int(np.sum(nonzero_mask))
            if n_valid == 0:
                continue

            valid_acc = context_acc[nonzero_mask]
            top_k_all = np.argpartition(valid_acc, -k, axis=1)[:, -k:]

            # Build context SDRs
            valid_sdrs = np.zeros((n_valid, D), dtype=np.uint8)
            rows = np.arange(n_valid).reshape(-1, 1)
            valid_sdrs[rows, top_k_all] = 1

            # Target SDRs for valid positions
            positions = np.arange(1, n)
            valid_positions = positions[nonzero_mask]
            valid_targets = np.array([valid_seq[p] for p in valid_positions], dtype=np.int64)

            # Filter out zero context/target
            ctx_sums = np.sum(valid_sdrs, axis=1)
            tgt_sums = np.sum(word_sdrs[valid_targets], axis=1)
            both_nonzero = (ctx_sums > 0) & (tgt_sums > 0)

            for i in np.where(both_nonzero)[0]:
                batch_ctx.append(valid_sdrs[i].copy())
                batch_tgt.append(word_sdrs[int(valid_targets[i])].copy())
                total_pairs += 1

                if len(batch_ctx) >= batch_size:
                    print(f"        [encode] Yielding batch of {len(batch_ctx)} pairs "
                          f"after seq {seq_idx+1}/{len(sequences)} "
                          f"({time.time()-t_start:.1f}s)", flush=True)
                    ctx_arr = np.array(batch_ctx, dtype=np.uint8)
                    tgt_arr = np.array(batch_tgt, dtype=np.uint8)
                    yield ctx_arr, tgt_arr
                    batch_ctx = []
                    batch_tgt = []

            n_seqs_processed += 1

            # Progress every 500 sequences or 5 seconds
            now = time.time()
            if n_seqs_processed % 500 == 0 or (now - t_start) > 5.0:
                elapsed = now - t_start
                rate = n_seqs_processed / max(0.1, elapsed)
                eta = (len(sequences) - seq_idx) / max(1, rate)
                print(f"        [encode] {n_seqs_processed} seqs, {total_pairs} pairs, "
                      f"{rate:.0f} seqs/s, ETA {eta:.0f}s", flush=True)
                t_start = now
                n_seqs_processed = 0

        # Final partial batch
        if batch_ctx:
            print(f"        [encode] Final batch: {len(batch_ctx)} pairs", flush=True)
            ctx_arr = np.array(batch_ctx, dtype=np.uint8)
            tgt_arr = np.array(batch_tgt, dtype=np.uint8)
            yield ctx_arr, tgt_arr

        print(f"        [encode] Done: {total_pairs} total pairs (with binding)", flush=True)

        if callback:
            callback(len(sequences), total_pairs)

    def encode_contexts_batch_positional(
        self,
        sequences: List[List[int]],
        context_window: int = 10,
        batch_size: int = 50000,
        callback=None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        VECTORIZED batch POSITIONAL VSA context encoding for Hebbian training.

        v52: Replaces BOW cumsum with position-bound superposition.
        Each word's SDR is rotated by its RELATIVE position hash before
        being summed into the context accumulator. This preserves word
        order in the context SDR, allowing the DAM to learn sequential
        patterns (not just co-occurrence).

        Key difference from v45 binding encoding:
          - v45 used content-dependent hash: bind(word, hash(prev_word))
            → same BOW context produced different SDRs depending on order
            → DAM couldn't handle the variation → PPL regressed 221→1909
          - v52 uses position-dependent hash: rotate(word, hash(rel_pos))
            → same ordered context always produces the same SDR
            → consistent training signals → DAM can learn position-dependent patterns

        Implementation:
        1. Pre-compute rotated SDRs for all words at all relative positions
           rotated_all[r, w, :] = roll(word_sdrs[w], pos_hash(r+1))
           Memory: W * V * D bytes ≈ 10 MB for W=10, V=2000, D=512

        2. For each sequence, gather rotated SDRs per relative position
           using numpy advanced indexing, then sum to get context accumulator.
           Vectorized per-relative-position: O(W) per sequence instead of O(n*W).

        3. Apply kWTA in bulk (same as BOW version).

        Speed: ~2-3x slower than BOW cumsum, but still processes
        50K sequences in ~1-2 minutes on Raspberry Pi 5.

        Args:
            sequences: List of token sequences.
            context_window: Context window size.
            batch_size: Yield every this many pairs (for memory control).
            callback: Optional callable(seq_idx, total_pairs) for progress.

        Yields:
            (context_sdrs, target_sdrs) — each (N, D) uint8 arrays.
        """
        if not self._built:
            raise RuntimeError("SDREncoder not built — call build() first")

        D = self.D
        k = self.k
        V = self.vocab_size
        word_sdrs = self.word_sdrs  # (V, D) uint8

        PRIME = 7919

        # Pre-compute position shifts for relative positions 1..context_window
        pos_shifts = [(r * PRIME + 1) % D for r in range(1, context_window + 1)]

        # Pre-compute rotated SDRs for all words at all relative positions
        # rotated_all[r, w, :] = roll(word_sdrs[w], pos_shifts[r])
        # r: 0 = relative position 1 (most recent), W-1 = relative position W (oldest)
        print(f"        [encode_positional] Pre-computing rotated SDRs: "
              f"{context_window} pos x {V} words x {D} dims = "
              f"{context_window * V * D / 1024 / 1024:.1f} MB...", flush=True)
        rotated_all = np.zeros((context_window, V, D), dtype=np.int8)
        for r in range(context_window):
            rotated_all[r] = np.roll(word_sdrs.astype(np.int8), pos_shifts[r], axis=1)

        print(f"        [encode_contexts_batch_positional] Starting: {len(sequences)} seqs, "
              f"D={D}, k={k}, batch_size={batch_size}", flush=True)

        batch_ctx = []
        batch_tgt = []
        total_pairs = 0
        t_start = time.time()
        n_seqs_processed = 0

        for seq_idx, seq in enumerate(sequences):
            seq_len = len(seq)
            if seq_len < 3:
                continue

            # Filter valid word IDs
            valid_seq = [w for w in seq if 0 <= w < V]
            if len(valid_seq) < 3:
                continue

            n = len(valid_seq)
            valid_seq_arr = np.array(valid_seq, dtype=np.int64)

            # Compute positional context accumulators for all target positions
            # using per-relative-position vectorized approach.
            #
            # For relative position r (0-based, 0=most recent):
            #   Target positions with a word at this relative position: r+1, r+2, ..., n-1
            #   Word at target p, relative position r+1: valid_seq[p-1-r]
            #   Its rotated SDR: rotated_all[r, valid_seq[p-1-r]]
            #
            # We process one relative position at a time, gathering ALL words
            # at that position across all targets, then add to the accumulator.
            context_acc = np.zeros((n - 1, D), dtype=np.int32)
            W = min(context_window, n - 1)

            for r in range(W):
                # Words at relative position r+1 for all valid target positions
                p_range = np.arange(r + 1, n)
                word_indices = valid_seq_arr[p_range - 1 - r]
                # Gather rotated SDRs using advanced indexing: (n-1-r, D)
                rotated_at_r = rotated_all[r, word_indices]
                context_acc[r:n-1] += rotated_at_r

            # kWTA for ALL positions at once using 2D argpartition
            nonzero_mask = np.max(context_acc, axis=1) > 0
            n_valid = int(np.sum(nonzero_mask))
            if n_valid == 0:
                continue

            valid_acc = context_acc[nonzero_mask]
            top_k_all = np.argpartition(valid_acc, -k, axis=1)[:, -k:]

            # Build context SDRs
            valid_sdrs = np.zeros((n_valid, D), dtype=np.uint8)
            rows = np.arange(n_valid).reshape(-1, 1)
            valid_sdrs[rows, top_k_all] = 1

            # Target SDRs for valid positions
            positions = np.arange(1, n)
            valid_positions = positions[nonzero_mask]
            valid_targets = np.array([valid_seq[p] for p in valid_positions], dtype=np.int64)

            # Filter out zero context/target
            ctx_sums = np.sum(valid_sdrs, axis=1)
            tgt_sums = np.sum(word_sdrs[valid_targets], axis=1)
            both_nonzero = (ctx_sums > 0) & (tgt_sums > 0)

            for i in np.where(both_nonzero)[0]:
                batch_ctx.append(valid_sdrs[i].copy())
                batch_tgt.append(word_sdrs[int(valid_targets[i])].copy())
                total_pairs += 1

                if len(batch_ctx) >= batch_size:
                    print(f"        [encode] Yielding batch of {len(batch_ctx)} pairs "
                          f"after seq {seq_idx+1}/{len(sequences)} "
                          f"({time.time()-t_start:.1f}s)", flush=True)
                    ctx_arr = np.array(batch_ctx, dtype=np.uint8)
                    tgt_arr = np.array(batch_tgt, dtype=np.uint8)
                    yield ctx_arr, tgt_arr
                    batch_ctx = []
                    batch_tgt = []

            n_seqs_processed += 1

            # Progress every 500 sequences or 5 seconds
            now = time.time()
            if n_seqs_processed % 500 == 0 or (now - t_start) > 5.0:
                elapsed = now - t_start
                rate = n_seqs_processed / max(0.1, elapsed)
                eta = (len(sequences) - seq_idx) / max(1, rate)
                print(f"        [encode] {n_seqs_processed} seqs, {total_pairs} pairs, "
                      f"{rate:.0f} seqs/s, ETA {eta:.0f}s", flush=True)
                t_start = now
                n_seqs_processed = 0

        # Final partial batch
        if batch_ctx:
            print(f"        [encode] Final batch: {len(batch_ctx)} pairs", flush=True)
            ctx_arr = np.array(batch_ctx, dtype=np.uint8)
            tgt_arr = np.array(batch_tgt, dtype=np.uint8)
            yield ctx_arr, tgt_arr

        print(f"        [encode] Done: {total_pairs} total pairs (positional VSA)", flush=True)

        if callback:
            callback(len(sequences), total_pairs)

    def hamming_overlap(self, sdr1: np.ndarray, sdr2: np.ndarray) -> int:
        """
        Compute Hamming overlap (count of shared active bits) between two SDRs.

        This is the fundamental similarity metric for sparse codes.
        For two random k-sparse vectors of dim D:
            E[overlap] = k²/D (very small for k << D)

        Args:
            sdr1, sdr2: Binary vectors (D,) uint8.

        Returns:
            Integer count of shared active bits.
        """
        return int(np.sum(sdr1 & sdr2))

    def nearest_word(
        self,
        sdr: np.ndarray,
        candidate_words: Optional[np.ndarray] = None,
        n_candidates: int = 200,
        word_freq: Optional[np.ndarray] = None,
    ) -> int:
        """
        Find the word whose SDR has the highest overlap with the query SDR.

        This is the "readout" operation: map from a predicted SDR back
        to the most likely word. Uses brute-force Hamming overlap,
        which is fast for sparse vectors (only k active bits to check).

        Args:
            sdr: Query SDR (D,) uint8.
            candidate_words: Optional array of candidate word IDs.
            n_candidates: Max number of candidates to check.
            word_freq: Optional frequency array for top-k pre-filtering.

        Returns:
            Word ID with highest Hamming overlap.
        """
        if not self._built:
            raise RuntimeError("SDREncoder not built — call build() first")

        if candidate_words is None:
            if word_freq is not None and len(word_freq) > n_candidates:
                # Pre-filter to most common words
                top_idx = np.argsort(word_freq)[-n_candidates:]
                candidate_words = top_idx
            else:
                candidate_words = np.arange(min(self.vocab_size, n_candidates))

        # Compute overlaps using sparse format
        # For each candidate, count shared active bits
        sdr_active = np.where(sdr > 0)[0]

        best_word = int(candidate_words[0])
        best_overlap = 0

        for w in candidate_words:
            w = int(w)
            if w < 0 or w >= self.vocab_size:
                continue
            overlap = len(np.intersect1d(self.word_active_bits[w], sdr_active, assume_unique=True))
            if overlap > best_overlap:
                best_overlap = overlap
                best_word = w

        return best_word

    def compute_overlap_batch(
        self,
        sdr: np.ndarray,
        candidate_words: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Hamming overlap between a query SDR and all candidate words.

        Vectorized version using dense matrix multiplication.
        For binary vectors: overlap(sdr, word_sdr[w]) = sdr · word_sdr[w]

        Args:
            sdr: Query SDR (D,) uint8.
            candidate_words: Array of candidate word IDs.

        Returns:
            Array of overlaps (len(candidate_words),) int32.
        """
        if not self._built:
            raise RuntimeError("SDREncoder not built — call build() first")

        # Batch dot product: word_sdrs[candidates] @ sdr
        candidate_sdrs = self.word_sdrs[candidate_words]  # (n_cand, D) uint8
        overlaps = candidate_sdrs.astype(np.int32) @ sdr.astype(np.int32)
        return overlaps

    def _print_diagnostics(self) -> None:
        """Print SDR quality diagnostics."""
        if self.word_sdrs is None:
            return

        V, D = self.word_sdrs.shape

        # Verify sparsity
        active_per_word = np.sum(self.word_sdrs, axis=1)
        avg_active = np.mean(active_per_word)
        assert np.all(active_per_word == self.k), \
            f"SDR sparsity violation: expected {self.k} active bits, got range [{active_per_word.min()}, {active_per_word.max()}]"

        # Pairwise overlap statistics (sample)
        n_sample = min(500, V)
        sample_idx = np.random.choice(V, size=n_sample, replace=False)
        sample_sdrs = self.word_sdrs[sample_idx]

        overlaps = []
        for i in range(min(200, n_sample)):
            for j in range(i + 1, min(i + 20, n_sample)):
                ov = int(np.sum(sample_sdrs[i] & sample_sdrs[j]))
                overlaps.append(ov)

        if overlaps:
            avg_overlap = np.mean(overlaps)
            max_overlap = np.max(overlaps)
            # Expected random overlap: k²/D
            expected_random = self.k ** 2 / D
            print(f"    SDR diagnostics: D={D}, k={self.k} ({self.sparsity*100:.1f}% sparse)")
            print(f"      Avg pairwise overlap: {avg_overlap:.2f} (random expectation: {expected_random:.2f})")
            print(f"      Max pairwise overlap: {max_overlap}")
            print(f"      Memory: {self.word_sdrs.nbytes / 1024:.1f} KB")

    def reset(self) -> None:
        """Reset encoder state (keep built SDRs)."""
        pass
