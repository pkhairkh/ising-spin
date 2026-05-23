"""
Sparse Long-Range Word-Word Coupling for Ising Spin Glass LM — v16.1
=====================================================================

v16.1 FIX: Memory-efficient incremental COO construction.
v16 had OOM on Pi 16GB because it collected ALL (target, context) pairs
into giant numpy arrays before building the sparse matrix:
  27M tokens × 30 offsets ≈ 810M pairs × 16 bytes = 13GB intermediates!

v16.1 FIX:
  - Process each batch, convert to CSR immediately, add to accumulator
  - Never hold more than one batch of pairs in memory
  - Peak intermediate memory: ~1 batch × 30 offsets instead of all batches
  - Also fixed boundary detection bug (boundary_within_d was computed but unused)

KEY DIFFERENCE from v15:
  v15: context → cluster histogram → H2W cluster→word → per-word (DILUTED)
  v16: context → direct word→word PMI → per-word (NOT diluted)

Expected energy range:
  5 active context words × PMI 5 × decay 0.5 × weight 800 = 10,000
  This COMPETES with recall (±32,000) — can actually change rankings!

INTEGER-ONLY ARITHMETIC
------------------------
- PMI values: int16 (Q3: × 8)
- Positional decay: int16 (Q8: 0-256)
- Energy accumulation: int32/int64
- ZERO floating-point in the hot path
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import time


def build_decay_table(window: int = 30) -> np.ndarray:
    """Build integer positional decay table. decay(d) = max(1, 256 >> (d // 4))"""
    table = np.zeros(window + 1, dtype=np.int16)
    for d in range(window + 1):
        table[d] = max(1, 256 >> (d // 4))
    return table


class LongRangeCouplingLayer:
    """
    v16.1: Sparse Long-Range Word-Word Coupling Layer.
    
    Computes per-word energy from long-range context (30 tokens)
    using direct word→word PMI couplings. No cluster routing.
    """
    
    def __init__(
        self,
        vocab_size: int = 4000,
        window: int = 30,
        top_k: int = 200,
        longrange_weight: int = 800,
        pmi_cap: int = 64,
        min_count: int = 5,
        confidence_min_count: int = 10,
        min_confidence_q8: int = 128,
        enabled: bool = True,
    ):
        self.vocab_size = vocab_size
        self.window = window
        self.top_k = top_k
        self.longrange_weight = longrange_weight
        self.pmi_cap = pmi_cap
        self.min_count = min_count
        self.confidence_min_count = confidence_min_count
        self.min_confidence_q8 = min_confidence_q8
        self.enabled = enabled
        
        self.J_lr: Dict[int, Dict[int, int]] = {}
        self.decay_table = build_decay_table(window)
        self._built = False
        self._context_buffer: List[int] = []
        
        self._diag = {
            'lr_hits': 0,
            'lr_energy_sum': 0,
            'lr_zero_count': 0,
            'lr_total_candidates': 0,
            'confidence_sum': 0,
            'confidence_count': 0,
        }
    
    def build(
        self,
        sequences: List[List[int]],
        vocab_size: int,
        word_freq: np.ndarray,
    ) -> None:
        """
        Build the long-range coupling matrix from training sequences.
        
        v16.1: Memory-efficient incremental CSR construction.
        Each batch is converted to CSR and summed into an accumulator
        immediately — never holds all pairs in memory at once.
        """
        print(f"\n  [v16.1] Building Long-Range Coupling Layer...")
        print(f"  [v16.1]   vocab_size={vocab_size}, window={self.window}")
        print(f"  [v16.1]   top_k={self.top_k}, longrange_weight={self.longrange_weight}")
        print(f"  [v16.1]   pmi_cap={self.pmi_cap}, min_count={self.min_count}")
        
        self.vocab_size = vocab_size
        t0 = time.time()
        
        # ─── Phase 1: Word frequencies ───────────────────────────
        print("  [LR16] Phase 1: Counting word frequencies...")
        wf = word_freq.copy() if len(word_freq) >= vocab_size else np.zeros(vocab_size, dtype=np.int64)
        total_tokens = int(wf.sum())
        print(f"  [LR16]   Total tokens: {total_tokens:,}")
        print(f"  [LR16]   Non-zero words: {np.count_nonzero(wf):,}")
        
        # ─── Phase 2: Count co-occurrences INCREMENTALLY ─────────
        # v16.1 FIX: Process each batch, convert to CSR, add to accumulator.
        # Peak memory: one batch of pairs + CSR accumulator.
        print("  [LR16] Phase 2: Counting co-occurrences (incremental CSR)...")
        
        import scipy.sparse as sp
        
        cooc_accumulator = None  # Will be a CSR matrix
        total_pairs = 0
        batch_size = 50000
        
        for batch_start in range(0, len(sequences), batch_size):
            batch_end = min(batch_start + batch_size, len(sequences))
            t_batch = time.time()
            
            # Concatenate sequences with -1 separators
            parts = []
            for seq in sequences[batch_start:batch_end]:
                parts.append(seq)
                parts.append([-1])
            
            concat = np.concatenate([np.array(p, dtype=np.int64) for p in parts])
            n_concat = len(concat)
            
            # Build boundary positions: True where we should NOT cross
            is_boundary = np.zeros(n_concat, dtype=bool)
            is_boundary[0] = True
            is_boundary[concat == -1] = True
            
            # Pre-compute cumulative boundary flag for efficient range checks.
            # cbf[i] = number of boundaries in [0, i] (inclusive)
            # Then: boundary between positions i and i+d (exclusive of i)
            #       iff cbf[i+d] - cbf[i] > 0
            cbf = np.cumsum(is_boundary.astype(np.int32))
            
            # Collect pairs for THIS BATCH ONLY
            batch_targets = []
            batch_contexts = []
            
            for d in range(1, min(self.window + 1, n_concat)):
                # target at position i+d, context at position i
                ctx = concat[:n_concat - d]
                tgt = concat[d:]
                
                # Valid: both in vocab range
                valid = (ctx >= 0) & (ctx < vocab_size) & (tgt >= 0) & (tgt < vocab_size)
                
                if d > 1:
                    # Check no boundary between position i and position i+d
                    # Boundary in (i, i+d] iff cbf[i+d] - cbf[i] > 0
                    boundary_between = cbf[d:] - cbf[:n_concat - d]
                    has_boundary = boundary_between > 0
                    valid = valid & ~has_boundary
                else:
                    # d=1: just check that position i+1 is not a boundary start
                    # (i.e., not a separator position)
                    valid = valid & ~is_boundary[d:]
                
                if valid.any():
                    batch_targets.append(tgt[valid])
                    batch_contexts.append(ctx[valid])
            
            # Free concat immediately
            del concat, is_boundary, cbf
            
            if batch_targets:
                bt = np.concatenate(batch_targets)
                bc = np.concatenate(batch_contexts)
                del batch_targets, batch_contexts
                
                n_pairs = len(bt)
                total_pairs += n_pairs
                
                # Build COO for this batch and convert to CSR immediately
                batch_coo = sp.coo_matrix(
                    (np.ones(n_pairs, dtype=np.int64), (bt, bc)),
                    shape=(vocab_size, vocab_size)
                )
                batch_csr = batch_coo.tocsr()
                del batch_coo, bt, bc
                
                # Accumulate into global CSR
                if cooc_accumulator is None:
                    cooc_accumulator = batch_csr
                else:
                    cooc_accumulator = cooc_accumulator + batch_csr
                del batch_csr
            
            elapsed = time.time() - t_batch
            if batch_start > 0 or batch_end < len(sequences):
                print(f"  [LR16]     Batch {batch_start//batch_size + 1}: "
                      f"seqs {batch_start}-{batch_end}, "
                      f"pairs={total_pairs:,}, "
                      f"elapsed={elapsed:.1f}s")
        
        if cooc_accumulator is None:
            print("  [LR16]   No co-occurrences found!")
            self._built = True
            return
        
        n_nz = cooc_accumulator.nnz
        print(f"  [LR16]   Total pairs processed: {total_pairs:,}")
        print(f"  [LR16]   Non-zero co-occurrence pairs (after dedup): {n_nz:,}")
        
        # ─── Phase 3: Compute PMI and keep top-K per target ──────
        print("  [LR16] Phase 3: Computing PMI and sparsifying...")
        
        import math
        N = total_tokens
        self.J_lr = {}
        
        for target_word in range(vocab_size):
            if wf[target_word] < self.min_count:
                continue
            
            row_start = cooc_accumulator.indptr[target_word]
            row_end = cooc_accumulator.indptr[target_word + 1]
            
            if row_start == row_end:
                continue
            
            context_words = cooc_accumulator.indices[row_start:row_end]
            context_counts = cooc_accumulator.data[row_start:row_end]
            
            pmi_scores = {}
            for idx in range(len(context_words)):
                cw = int(context_words[idx])
                count = int(context_counts[idx])
                
                if count < self.min_count or wf[cw] < self.min_count:
                    continue
                
                ratio_num = count * N
                ratio_den = int(wf[cw]) * int(wf[target_word])
                
                if ratio_den == 0 or ratio_num <= 0:
                    continue
                
                ratio = ratio_num / ratio_den
                
                if ratio <= 1.0:  # PMI ≤ 0 → not predictive
                    continue
                
                pmi_float = math.log2(ratio)
                pmi_q3 = int(pmi_float * 8)
                pmi_q3 = min(self.pmi_cap, pmi_q3)
                
                pmi_scores[cw] = pmi_q3
            
            if pmi_scores:
                top_items = sorted(pmi_scores.items(), key=lambda x: -x[1])[:self.top_k]
                self.J_lr[target_word] = dict(top_items)
        
        # Free the big CSR matrix — we don't need it anymore
        del cooc_accumulator
        
        # ─── Phase 4: Build reverse index for fast inference ─────
        print("  [LR16] Phase 4: Building reverse index for inference...")
        
        self.J_reverse: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for target_word, context_dict in self.J_lr.items():
            for context_word, pmi_q3 in context_dict.items():
                self.J_reverse[context_word].append((target_word, pmi_q3))
        
        for cw in self.J_reverse:
            self.J_reverse[cw].sort(key=lambda x: -x[1])
        
        self.J_reverse = dict(self.J_reverse)
        
        # ─── Statistics ──────────────────────────────────────────
        n_targets = len(self.J_lr)
        n_entries = sum(len(v) for v in self.J_lr.values())
        n_reverse_keys = len(self.J_reverse)
        
        all_pmi = []
        for target_dict in self.J_lr.values():
            all_pmi.extend(target_dict.values())
        
        if all_pmi:
            pmi_arr = np.array(all_pmi, dtype=np.int32)
            print(f"  [LR16]   PMI distribution: min={pmi_arr.min()}, max={pmi_arr.max()}, "
                  f"mean={pmi_arr.mean():.1f}, median={np.median(pmi_arr):.1f}")
        
        mem_mb = n_entries * 8 / (1024 * 1024)
        print(f"  [LR16]   Target words with couplings: {n_targets:,}")
        print(f"  [LR16]   Total coupling entries: {n_entries:,}")
        print(f"  [LR16]   Reverse index keys: {n_reverse_keys:,}")
        print(f"  [LR16]   Memory: ~{mem_mb:.1f} MB")
        
        t_build = time.time() - t0
        print(f"  [LR16]   Build time: {t_build:.1f}s")
        print(f"  [v16.1] Long-Range Coupling Layer built successfully.")
        
        self._built = True
    
    def reset(self):
        """Reset context buffer for a new sequence."""
        self._context_buffer = []
    
    def update_context(self, word_id: int):
        """Add a word to the context buffer."""
        self._context_buffer.append(word_id)
        if len(self._context_buffer) > self.window:
            self._context_buffer = self._context_buffer[-self.window:]
    
    def compute_energy(
        self,
        candidate_words: np.ndarray,
        context_words: List[int],
        recall_ngram_level: int = 5,
        recall_ngram_count: int = 100,
    ) -> Tuple[np.ndarray, int]:
        """
        Compute long-range coupling energy for candidate words.
        
        Returns (energies, confidence_q8).
        """
        n_candidates = len(candidate_words)
        energies = np.zeros(n_candidates, dtype=np.int64)
        
        if not self._built or len(context_words) == 0:
            return energies, 256
        
        # Use REVERSE index: iterate over context words (≤30)
        # and accumulate energy for their associated targets.
        candidate_set = set(int(w) for w in candidate_words)
        candidate_energy = {}
        
        n_ctx = len(context_words)
        
        for i, ctx_word in enumerate(context_words):
            dist = n_ctx - i
            if dist < 0 or dist >= len(self.decay_table):
                decay = 1
            else:
                decay = int(self.decay_table[dist])
            
            if ctx_word in self.J_reverse:
                for target_word, pmi_q3 in self.J_reverse[ctx_word]:
                    if target_word in candidate_set:
                        contribution = pmi_q3 * decay  # Q3 × Q8 = Q11
                        if target_word in candidate_energy:
                            candidate_energy[target_word] += contribution
                        else:
                            candidate_energy[target_word] = contribution
        
        # Scale: longrange_weight × Q11 / 2048
        # Give a BONUS to words predicted by long-range context.
        # No penalty for unmatched words (recall already handles that).
        # No centering (that was causing problems).
        # v16.1: Cap at 4× weight so LR can actually compete with recall.
        # With 5 context words × PMI 5 × decay 0.5 × weight 800:
        #   raw ≈ 25600, scaled ≈ 10000, cap = 3200 → most signal preserved
        # Recall swings ±32000, so ±3200 LR = 10% of recall — significant!
        scale_divisor = 2048
        
        lr_hits = 0
        lr_energy_sum = 0
        
        max_lr = self.longrange_weight * 4  # 4× cap: allows LR to matter
        
        for i, w in enumerate(candidate_words):
            w_int = int(w)
            if w_int in candidate_energy:
                raw = candidate_energy[w_int]
                scaled = (raw * self.longrange_weight) // scale_divisor
                # Cap to prevent extreme values (but allow up to 4× weight)
                scaled = min(max_lr, max(-max_lr, scaled))
                energies[i] = -scaled  # Negative = more likely
                lr_hits += 1
                lr_energy_sum += abs(scaled)
        
        # Recall confidence
        ngram_level_q8 = min(256, recall_ngram_level * 51)
        count_ratio = min(256, (recall_ngram_count * 256) // max(1, self.confidence_min_count))
        confidence_q8 = max(self.min_confidence_q8, (ngram_level_q8 * count_ratio) >> 8)
        
        # Diagnostics
        self._diag['lr_hits'] += lr_hits
        self._diag['lr_energy_sum'] += lr_energy_sum
        self._diag['lr_zero_count'] += (n_candidates - lr_hits)
        self._diag['lr_total_candidates'] += n_candidates
        self._diag['confidence_sum'] += confidence_q8
        self._diag['confidence_count'] += 1
        
        return energies, confidence_q8
    
    def get_diagnostics(self) -> Dict:
        """Return diagnostic counters."""
        return dict(self._diag)
