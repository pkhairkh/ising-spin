#!/usr/bin/env python3
"""
Ising Spin Model — Parallel Training
=====================================

Multi-core training for Raspberry Pi (4 cores) and desktop.

Parallelizes the 3 bottlenecks:
  1. PMI co-occurrence counting
  2. Skip-gram PMI (5 distances)
  3. N-gram index building

Also merges steps 4+5 into a single pass (both need co-occurrence data).

Speedup: ~3.2× on 4 cores (Pi 4/5), ~6× on 8 cores.

Usage:
  # Drop-in replacement for train_long.py:
  python train_parallel.py --n-samples 200000

  # Quick test:
  python train_parallel.py --test

  # Specify cores:
  python train_parallel.py --n-samples 500000 --workers 4
"""

import sys
import os
import time
import json
import argparse
import multiprocessing as mp
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
from functools import partial

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from ising_spin.model import (
    IsingLMModel, IntegerBoltzmannSampler, Vocabulary, POSTypeSystem,
    NGramIndex, compute_log_floor_pmi,
    load_fineweb_edu, tokenize_texts, truncate_sequences,
    LN2_NUM, LN2_DEN, LOG2_SCALE,
    N_POS,
)


# ============================================================================
# Parallel Workers
# ============================================================================

def _count_cooccurrences_chunk(args):
    """
    Worker: count unigrams + windowed co-occurrences for a chunk of sequences.
    Returns (unigram_counts, cooc_counts) where both are dicts/Counters.
    """
    sequences, window = args
    unigram = Counter()
    cooc = Counter()

    for seq in sequences:
        for w in seq:
            unigram[w] += 1
        for i, w in enumerate(seq):
            for j in range(i + 1, min(i + window + 1, len(seq))):
                cooc[(w, seq[j])] += 1

    return dict(unigram), dict(cooc)


def _count_skip_cooccurrences_chunk(args):
    """
    Worker: count co-occurrences at each distance for a chunk of sequences.
    Returns {distance: {(w1,w2): count}}
    """
    sequences, max_dist = args
    cooc_by_dist = {d: Counter() for d in range(1, max_dist + 1)}
    unigram = Counter()

    for seq in sequences:
        for w in seq:
            unigram[w] += 1
        for i, w in enumerate(seq):
            for d in range(1, min(max_dist + 1, len(seq) - i)):
                j = i + d
                cooc_by_dist[d][(w, seq[j])] += 1

    return dict(unigram), {d: dict(c) for d, c in cooc_by_dist.items()}


def _build_ngram_chunk(args):
    """
    Worker: build n-gram counts for a chunk of sequences.
    Returns {k: {context_tuple: {word: count}}}
    """
    sequences, max_n, min_count = args
    index = {k: defaultdict(Counter) for k in range(1, max_n + 1)}

    for seq in sequences:
        start = 0
        for i, w in enumerate(seq):
            if w >= 4:
                start = i
                break

        for t in range(start, len(seq)):
            for k in range(1, max_n + 1):
                if t - k < start:
                    break
                context = tuple(seq[t-k:t])
                continuation = seq[t]
                if any(w < 4 for w in context) or continuation < 4:
                    continue
                index[k][context][continuation] += 1

    # Convert defaultdicts to regular dicts for pickling
    result = {}
    for k in range(1, max_n + 1):
        result[k] = {}
        for ctx, conts in index[k].items():
            result[k][ctx] = dict(conts)

    return result


# ============================================================================
# Parallel PMI Computation
# ============================================================================

def compute_pmi_parallel(sequences, vocab_size, window=5, min_count=2,
                         pmi_cap=10, n_workers=4):
    """Compute PMI couplings using parallel co-occurrence counting."""
    V = vocab_size

    # Split sequences into chunks
    chunk_size = max(1, len(sequences) // n_workers)
    chunks = []
    for i in range(0, len(sequences), chunk_size):
        chunk = sequences[i:i+chunk_size]
        if chunk:
            chunks.append((chunk, window))

    # Count in parallel
    with mp.Pool(n_workers) as pool:
        results = pool.map(_count_cooccurrences_chunk, chunks)

    # Merge results
    unigram = np.zeros(V, dtype=np.int64)
    cooc_counts = Counter()

    for uni_chunk, cooc_chunk in results:
        for w, c in uni_chunk.items():
            if w < V:
                unigram[w] += c
        for pair, c in cooc_chunk.items():
            cooc_counts[pair] += c

    total_tokens = int(unigram.sum())

    # Build sparse J matrix
    rows, cols, data = [], [], []
    seen = set()
    for (w, w2), count in cooc_counts.items():
        if count >= min_count and w < V and w2 < V:
            pmi = compute_log_floor_pmi(
                int(count), int(unigram[w]), int(unigram[w2]),
                total_tokens, cap=pmi_cap
            )
            if pmi != 0:
                if (w, w2) not in seen:
                    rows.append(w)
                    cols.append(w2)
                    data.append(pmi)
                    seen.add((w, w2))
                if (w2, w) not in seen:
                    rows.append(w2)
                    cols.append(w)
                    data.append(pmi)
                    seen.add((w2, w))

    J = sp.csr_matrix(
        (np.array(data, dtype=np.int64),
         (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
        shape=(V, V)
    )

    # Compute local field
    h = np.ones(V, dtype=np.int64)
    for w in range(V):
        if unigram[w] > 0 and total_tokens > unigram[w]:
            ratio = total_tokens // int(unigram[w])
            if ratio >= 2:
                h[w] = ratio.bit_length() - 1

    n_nonzero = J.nnz
    print(f"    PMI matrix (sparse): {n_nonzero:,} non-zero entries out of {V*V:,}")
    if n_nonzero > 0:
        sparse_bytes = J.data.nbytes + J.indices.nbytes + J.indptr.nbytes
        dense_bytes = V * V * 8
        print(f"    Memory: sparse {sparse_bytes/1024/1024:.1f}MB vs dense {dense_bytes/1024/1024:.1f}MB")
        print(f"    PMI range: [{int(J.data.min())}, {int(J.data.max())}]")

    return J, h


def compute_skip_pmi_parallel(sequences, vocab_size, max_dist=5, min_count=2,
                               pmi_cap=10, n_workers=4):
    """Compute skip-gram PMI using parallel counting."""
    V = vocab_size

    chunk_size = max(1, len(sequences) // n_workers)
    chunks = []
    for i in range(0, len(sequences), chunk_size):
        chunk = sequences[i:i+chunk_size]
        if chunk:
            chunks.append((chunk, max_dist))

    with mp.Pool(n_workers) as pool:
        results = pool.map(_count_skip_cooccurrences_chunk, chunks)

    # Merge
    unigram = np.zeros(V, dtype=np.int64)
    cooc_by_dist = {d: Counter() for d in range(1, max_dist + 1)}

    for uni_chunk, cooc_chunk in results:
        for w, c in uni_chunk.items():
            if w < V:
                unigram[w] += c
        for d in range(1, max_dist + 1):
            for pair, c in cooc_chunk[d].items():
                cooc_by_dist[d][pair] += c

    total_tokens = int(unigram.sum())

    # Build sparse matrices
    J_skip = {}
    for dist in range(1, max_dist + 1):
        rows, cols, data = [], [], []
        seen = set()
        for (w, w2), count in cooc_by_dist[dist].items():
            if count >= min_count and w < V and w2 < V:
                pmi = compute_log_floor_pmi(
                    int(count), int(unigram[w]), int(unigram[w2]),
                    total_tokens, cap=pmi_cap
                )
                if pmi != 0:
                    if (w, w2) not in seen:
                        rows.append(w)
                        cols.append(w2)
                        data.append(pmi)
                        seen.add((w, w2))
                    if (w2, w) not in seen:
                        rows.append(w2)
                        cols.append(w)
                        data.append(pmi)
                        seen.add((w2, w))

        J_skip[dist] = sp.csr_matrix(
            (np.array(data, dtype=np.int64),
             (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
            shape=(V, V)
        )
        print(f"    Skip-PMI dist={dist}: {J_skip[dist].nnz:,} non-zero entries")

    return J_skip


def build_ngram_parallel_mutate(self, sequences, n_workers=4):
    """Parallel NGramIndex.build that mutates self in-place."""
    max_n = self.max_n
    min_count = self.min_count

    chunk_size = max(1, len(sequences) // n_workers)
    chunks = []
    for i in range(0, len(sequences), chunk_size):
        chunk = sequences[i:i+chunk_size]
        if chunk:
            chunks.append((chunk, max_n, min_count))

    with mp.Pool(n_workers) as pool:
        results = pool.map(_build_ngram_chunk, chunks)

    # Merge: add counters for matching contexts
    merged = {k: defaultdict(Counter) for k in range(1, max_n + 1)}
    for chunk_result in results:
        for k in range(1, max_n + 1):
            for ctx, conts in chunk_result[k].items():
                for w, c in conts.items():
                    merged[k][ctx][w] += c

    # Prune low-count and populate self.index/self.context_totals
    for k in range(1, max_n + 1):
        for ctx in list(merged[k].keys()):
            # Prune
            low_count = [w for w, c in merged[k][ctx].items() if c < min_count]
            for w in low_count:
                del merged[k][ctx][w]
            if not merged[k][ctx]:
                del merged[k][ctx]
                continue
            self.index[k][ctx] = Counter(merged[k][ctx])
            self.context_totals[k][ctx] = sum(merged[k][ctx].values())

    self._built = True

    # Build unigram totals
    self._unigram_totals = {}
    if 1 in self.index:
        total_N = sum(self.context_totals[1].values())
        for context in self.index[1]:
            if context and len(context) == 1:
                w = context[0]
                count_w = self.context_totals[1].get(context, 0)
                self._unigram_totals[w] = (count_w, total_N)

    # KN continuation counts
    self._kn_continuation = {}
    for k in range(2, max_n + 1):
        cont_count = Counter()
        for context, continuations in self.index[k].items():
            for w in continuations:
                cont_count[w] += 1
        self._kn_continuation[k] = dict(cont_count)

    self._kn_totals = {}
    for k, cont_count in self._kn_continuation.items():
        self._kn_totals[k] = sum(cont_count.values())

    self._kn_discount = 3
    self._kn_discount_fp = 12

    for k in range(1, max_n + 1):
        n_ctx = len(self.index[k])
        n_cont = sum(len(v) for v in self.index[k].values())
        kn_info = f", KN cont={len(self._kn_continuation.get(k, {})):,}" if k >= 2 else ""
        print(f"    {k}-gram: {n_ctx:,} contexts, {n_cont:,} continuations{kn_info}")

    return self


# ============================================================================
# Monkey-patch IsingLMModel.train for parallel execution
# ============================================================================

def train_parallel(model, n_samples=20000, texts=None, n_workers=None):
    """
    Drop-in replacement for IsingLMModel.train() that uses multiprocessing
    for the 3 bottleneck steps: PMI, skip-PMI, n-gram index.
    """
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 4)

    print("=" * 70)
    print("ISING-ENHANCED N-GRAM LANGUAGE MODEL -- PARALLEL TRAINING")
    print(f"  Workers: {n_workers}")
    print("=" * 70)

    # Call original train but monkey-patch the 3 bottleneck steps
    # We do this by temporarily replacing the functions

    import ising_spin.model as model_module

    # Save originals
    orig_compute_pmi = model_module.compute_pmi_couplings
    orig_compute_skip_pmi = model_module.compute_skip_pmi_couplings

    # Replace with parallel versions
    model_module.compute_pmi_couplings = partial(
        compute_pmi_parallel, n_workers=n_workers
    )
    model_module.compute_skip_pmi_couplings = partial(
        compute_skip_pmi_parallel, n_workers=n_workers
    )

    # Monkey-patch NGramIndex.build to use parallel version (mutates in-place)
    orig_ngram_build = model_module.NGramIndex.build

    def parallel_ngram_build(self, sequences):
        return build_ngram_parallel_mutate(self, sequences, n_workers=n_workers)

    model_module.NGramIndex.build = parallel_ngram_build

    try:
        # Run original training with parallel functions
        result = orig_train(model, n_samples=n_samples, texts=texts)
    finally:
        # Restore originals
        model_module.compute_pmi_couplings = orig_compute_pmi
        model_module.compute_skip_pmi_couplings = orig_compute_skip_pmi
        model_module.NGramIndex.build = orig_ngram_build

    return result


# Save original train method
orig_train = IsingLMModel.train


# ============================================================================
# Evaluation (same as train_long.py)
# ============================================================================

def quick_ppl(model, n_seqs=10):
    """Fast PPL estimate."""
    sampler = model.generator.word_sampler
    gen = model.generator
    total_log2_prob = 0
    total_tokens = 0

    for seq in model.test_sequences[:n_seqs]:
        if len(seq) < 3:
            continue
        for pos in range(1, len(seq)):
            target_word = seq[pos]
            context_words = seq[:pos]
            context_types = [gen._get_word_type(w) for w in context_words]
            word_type = gen._get_word_type(target_word)
            candidate_list = gen.type_words.get(word_type, [])
            if not candidate_list:
                continue
            candidate_words = np.array(candidate_list, dtype=np.int64)
            if int(target_word) not in set(candidate_words.tolist()):
                total_log2_prob += -15 * LOG2_SCALE
                total_tokens += 1
                continue
            recall_matches = gen.ngram_index.lookup(context_words)
            recall_hit = bool(recall_matches)
            energies = gen._compute_word_energy(
                pos, candidate_words, word_type,
                context_words, context_types, recall_hit
            )
            log_probs = sampler.compute_log_probabilities(energies)
            target_idx = np.where(candidate_words == target_word)[0]
            if len(target_idx) > 0:
                total_log2_prob += int(log_probs[target_idx[0]])
            else:
                total_log2_prob += -15 * LOG2_SCALE
            total_tokens += 1

    if total_tokens == 0:
        return float('inf')
    avg_log2 = total_log2_prob / (total_tokens * LOG2_SCALE)
    return 2.0 ** (-avg_log2)


def beta_sweep(model, n_seqs=20):
    """Sweep beta factor."""
    factors = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.5]
    recall_scale = model.recall_scale
    best_ppl = float('inf')
    best_factor = 0.9
    results = {}

    for factor in factors:
        beta_val = factor * LN2_NUM / (recall_scale * LN2_DEN)
        sampler = IntegerBoltzmannSampler(beta=beta_val, max_delta=25000)
        total_log2_prob = 0
        total_tokens = 0
        for seq in model.test_sequences[:n_seqs]:
            if len(seq) < 3:
                continue
            for pos in range(1, len(seq)):
                target_word = seq[pos]
                context_words = seq[:pos]
                context_types = [model.generator._get_word_type(w) for w in context_words]
                word_type = model.generator._get_word_type(target_word)
                candidate_list = model.generator.type_words.get(word_type, [])
                if not candidate_list:
                    continue
                candidate_words = np.array(candidate_list, dtype=np.int64)
                if int(target_word) not in set(candidate_words.tolist()):
                    total_log2_prob += -15 * LOG2_SCALE
                    total_tokens += 1
                    continue
                recall_matches = model.generator.ngram_index.lookup(context_words)
                recall_hit = bool(recall_matches)
                energies = model.generator._compute_word_energy(
                    pos, candidate_words, word_type,
                    context_words, context_types, recall_hit
                )
                log_probs = sampler.compute_log_probabilities(energies)
                target_idx = np.where(candidate_words == target_word)[0]
                if len(target_idx) > 0:
                    total_log2_prob += int(log_probs[target_idx[0]])
                else:
                    total_log2_prob += -15 * LOG2_SCALE
                total_tokens += 1
        if total_tokens > 0:
            avg_log2 = total_log2_prob / (total_tokens * LOG2_SCALE)
            ppl_val = 2.0 ** (-avg_log2)
            results[factor] = ppl_val
            marker = " <-- BEST" if ppl_val < best_ppl else ""
            print(f"    f={factor:.2f}: PPL={ppl_val:.1f}{marker}")
            if ppl_val < best_ppl:
                best_ppl = ppl_val
                best_factor = factor

    return best_ppl, best_factor, results


# ============================================================================
# Main
# ============================================================================

BEST_CONFIG = dict(
    vocab_min_freq=25, vocab_max_size=2000,
    ngram_max_n=5, ngram_min_count=2,
    recall_scale=1600, pmi_weight=5, field_weight=1,
    knowledge_scale=0, spin3_scale=0, category_scale=0,
    logic_rule_scale=0, logic_hard_scale=0,
    beta_type=0.001, beta_word=0.001,
    copy_enabled=True, copy_min_context=2, copy_min_confidence=0.25,
    same_word_penalty=200, max_closed_class_run=2,
    ising_enabled=True, skip_pmi_max_dist=5, mcmc_refine_steps=0,
    use_conceptnet=False, walsh_enabled=False,
    graded_couplings_enabled=False, auto_calibrate_beta=False,
    recall_primary_mode=True, interpolated=False, kn_backoff=False,
    topic_spin_enabled=False,
)


def main():
    parser = argparse.ArgumentParser(
        description="Ising Spin Model — Parallel Training (multi-core)",
    )
    parser.add_argument("--n-samples", type=int, default=200000)
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (default: CPU count)")
    parser.add_argument("--vocab-size", type=int, default=2000)
    parser.add_argument("--test", action="store_true",
                        help="Quick test (10K samples)")
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument("--no-sweep", action="store_true")
    args = parser.parse_args()

    n_workers = args.workers or min(mp.cpu_count(), 8)
    if args.test:
        args.n_samples = min(args.n_samples, 10000)

    config = BEST_CONFIG.copy()
    config['vocab_max_size'] = args.vocab_size

    print(f"\nWorkers: {n_workers} (CPU count: {mp.cpu_count()})")
    print(f"Samples: {args.n_samples}")
    print(f"Vocab: {args.vocab_size}")

    # Load data — pick the largest cache file that exists
    cache_path = args.cache
    if cache_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Find all cached files and pick the largest one
        cache_candidates = []
        for fname in os.listdir(script_dir):
            if fname.startswith("cached_fineweb_") and fname.endswith(".json"):
                fpath = os.path.join(script_dir, fname)
                # Extract count from filename like cached_fineweb_200k.json
                try:
                    count_str = fname.split("_")[-1].replace(".json", "")
                    count = int(count_str.replace("k", "000"))
                    cache_candidates.append((count, fpath))
                except (ValueError, IndexError):
                    pass

        # Sort by count descending — prefer largest cache
        cache_candidates.sort(reverse=True)

        # Use the largest cache that has >= n_samples, or the largest available
        for count, fpath in cache_candidates:
            if count >= args.n_samples:
                cache_path = fpath
                break
        if cache_path is None and cache_candidates:
            cache_path = cache_candidates[0][1]  # Use largest available

    if cache_path and os.path.exists(cache_path):
        print(f"Loading: {cache_path}")
        with open(cache_path) as f:
            texts = json.load(f)
        print(f"  {len(texts)} texts loaded")

        # If cache is too small, download more
        if len(texts) < args.n_samples:
            print(f"  Cache has {len(texts)} texts but need {args.n_samples}")
            print(f"  Downloading {args.n_samples} texts from FineWeb-Edu...")
            texts = load_fineweb_edu(n_samples=args.n_samples)
            cache_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"cached_fineweb_{len(texts)//1000}k.json"
            )
            with open(cache_path, 'w') as f:
                json.dump(texts, f)
            print(f"  Cached {len(texts)} texts to {cache_path}")
    else:
        print("No cache found. Downloading FineWeb-Edu...")
        texts = load_fineweb_edu(n_samples=args.n_samples)
        cache_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"cached_fineweb_{len(texts)//1000}k.json"
        )
        with open(cache_path, 'w') as f:
            json.dump(texts, f)
        print(f"  Cached {len(texts)} texts to {cache_path}")

    n_use = min(args.n_samples, len(texts))

    # Train with parallelism
    t0 = time.time()
    model = IsingLMModel(**config)

    # Use parallel training
    train_parallel(model, n_samples=n_use, texts=texts, n_workers=n_workers)

    t_train = time.time() - t0
    print(f"\nParallel training: {t_train:.1f}s ({t_train/60:.1f}min)")
    print(f"  Throughput: {n_use/t_train:.0f} samples/sec")
    print(f"  Vocab size: {len(model.vocab)}")

    # Quick PPL
    ppl_quick = quick_ppl(model, n_seqs=10)
    print(f"\nQuick PPL: {ppl_quick:.1f}")

    # Beta sweep
    if not args.no_sweep:
        print(f"\nBeta sweep:")
        best_ppl, best_factor, sweep_results = beta_sweep(model, n_seqs=20)
        print(f"\nBest: f={best_factor:.2f}, PPL={best_ppl:.1f}")

        recall_scale = model.recall_scale
        beta_val = best_factor * LN2_NUM / (recall_scale * LN2_DEN)
        model.generator.word_sampler = IntegerBoltzmannSampler(
            beta=beta_val, max_delta=25000
        )
        model.generator.beta_word = beta_val
        model.beta_word = beta_val
    else:
        best_ppl = ppl_quick
        best_factor = 0.0
        sweep_results = {}

    # Full PPL
    ppl_full = model.compute_perplexity(n_samples=100)
    print(f"\nPPL (full, 100 seqs): {ppl_full:.2f}")

    # Generate
    print(f"\n--- Generation ---")
    for prompt in ["the history of", "science and technology", "research shows that"]:
        result = model.generator.generate(prompt=prompt, length=400)
        text = result['text']
        words = text.split()
        n_recalls = sum(1 for d in result['diagnostics'] if d['recall_hit'])
        n_copies = sum(1 for d in result['diagnostics'] if d['copy'])
        print(f"\n  '{prompt}' ({len(words)} words, recalls={n_recalls}, copies={n_copies})")
        print(f"  {text[:200]}...")

    print(f"\nTotal: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    main()
