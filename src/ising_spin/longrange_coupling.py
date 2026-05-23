"""
Sparse Long-Range Word-Word Coupling for Ising Spin Glass LM — v16
====================================================================

v16 ARCHITECTURE: DIRECT WORD-WORD COUPLING (NO CLUSTERS)
For each target word, store top-200 context words with their PMI.
At inference: sum PMI contributions from last 30 tokens → per-word energy.

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
    v16: Sparse Long-Range Word-Word Coupling Layer.
    
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
        
        Uses scipy.sparse COO construction for fast co-occurrence counting.
        """
        print(f"\n  [v16] Building Long-Range Coupling Layer...")
        print(f"  [v16]   vocab_size={vocab_size}, window={self.window}")
        print(f"  [v16]   top_k={self.top_k}, longrange_weight={self.longrange_weight}")
        print(f"  [v16]   pmi_cap={self.pmi_cap}, min_count={self.min_count}")
        
        self.vocab_size = vocab_size
        t0 = time.time()
        
        # ─── Phase 1: Word frequencies ───────────────────────────
        print("  [LR16] Phase 1: Counting word frequencies...")
        wf = word_freq.copy() if len(word_freq) >= vocab_size else np.zeros(vocab_size, dtype=np.int64)
        total_tokens = int(wf.sum())
        print(f"  [LR16]   Total tokens: {total_tokens:,}")
        print(f"  [LR16]   Non-zero words: {np.count_nonzero(wf):,}")
        
        # ─── Phase 2: Count co-occurrences using COO sparse ──────
        # Key insight: build all (target, context) pairs as COO arrays,
        # then let scipy sum duplicates automatically.
        print("  [LR16] Phase 2: Counting co-occurrences (COO sparse)...")
        
        import scipy.sparse as sp
        
        # Collect all (target, context) pairs across all sequences and offsets
        all_targets = []
        all_contexts = []
        
        # Process sequences in batches to manage memory
        batch_size = 50000
        
        for batch_start in range(0, len(sequences), batch_size):
            batch_end = min(batch_start + batch_size, len(sequences))
            if batch_start > 0:
                print(f"  [LR16]     Processing sequence {batch_start}...")
            
            # Concatenate sequences with -1 separators
            parts = []
            seq_starts = []
            pos = 0
            for seq in sequences[batch_start:batch_end]:
                seq_starts.append(pos)
                parts.append(seq)
                parts.append([-1])  # Separator
                pos += len(seq) + 1
            
            concat = np.concatenate([np.array(p, dtype=np.int64) for p in parts])
            
            # Build boundary mask: True where we should NOT cross
            is_boundary = np.zeros(len(concat), dtype=bool)
            is_boundary[0] = True
            is_boundary[concat == -1] = True
            
            # For each offset d=1..window, collect valid (target, context) pairs
            for d in range(1, min(self.window + 1, len(concat))):
                # target at position i+d, context at position i
                ctx = concat[:len(concat)-d]
                tgt = concat[d:]
                
                # Valid: both in vocab range, no boundary between them
                valid = (ctx >= 0) & (ctx < vocab_size) & (tgt >= 0) & (tgt < vocab_size)
                
                if d > 1:
                    # Check no boundary between i and i+d
                    # A boundary at position j means we can't have a pair crossing it
                    # Use cumulative boundary detection
                    # For offset d: check that none of positions i+1..i+d have boundary
                    # Pre-compute: for each position, is there a boundary within d positions before?
                    boundary_within_d = np.zeros(len(concat), dtype=bool)
                    for dd in range(1, d + 1):
                        boundary_within_d[dd:] |= is_boundary[:-dd] if dd < len(is_boundary) else False
                    # Simpler: check that position i+d doesn't have boundary flag
                    # (meaning there's a separator right before it)
                    valid = valid & ~is_boundary[d:]
                
                if valid.any():
                    all_targets.append(tgt[valid])
                    all_contexts.append(ctx[valid])
        
        if not all_targets:
            print("  [LR16]   No co-occurrences found!")
            self._built = True
            return
        
        # Concatenate all pairs
        all_targets = np.concatenate(all_targets)
        all_contexts = np.concatenate(all_contexts)
        
        print(f"  [LR16]   Total pairs before dedup: {len(all_targets):,}")
        
        # Build sparse COO matrix — scipy automatically sums duplicates!
        cooc_sparse = sp.coo_matrix(
            (np.ones(len(all_targets), dtype=np.int64), (all_targets, all_contexts)),
            shape=(vocab_size, vocab_size)
        )
        # Convert to CSR for efficient row access
        cooc_csr = cooc_sparse.tocsr()
        
        n_nz = cooc_csr.nnz
        print(f"  [LR16]   Non-zero co-occurrence pairs: {n_nz:,}")
        
        # ─── Phase 3: Compute PMI and keep top-K per target ──────
        print("  [LR16] Phase 3: Computing PMI and sparsifying...")
        
        import math
        N = total_tokens
        self.J_lr = {}
        
        for target_word in range(vocab_size):
            if wf[target_word] < self.min_count:
                continue
            
            row_start = cooc_csr.indptr[target_word]
            row_end = cooc_csr.indptr[target_word + 1]
            
            if row_start == row_end:
                continue
            
            context_words = cooc_csr.indices[row_start:row_end]
            context_counts = cooc_csr.data[row_start:row_end]
            
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
        print(f"  [v16] Long-Range Coupling Layer built successfully.")
        
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
        # Simple approach: give a BONUS to words predicted by long-range context.
        # No penalty for unmatched words (recall already handles that).
        # No centering (that was causing problems).
        # The key is keeping the weight small enough that LR is a perturbation
        # that helps select between recall's top candidates, not override recall.
        scale_divisor = 2048
        
        lr_hits = 0
        lr_energy_sum = 0
        
        for i, w in enumerate(candidate_words):
            w_int = int(w)
            if w_int in candidate_energy:
                raw = candidate_energy[w_int]
                scaled = (raw * self.longrange_weight) // scale_divisor
                # Cap to prevent extreme values
                max_lr = self.longrange_weight
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
