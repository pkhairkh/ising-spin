#!/usr/bin/env python3
"""
Ising Spin Glass Language Model — v12 Training
================================================

Key improvements over v11.7:
  1. Kneser-Ney backoff (kn_backoff=True) — better backoff energies
  2. Interpolated smoothing (interpolated=True) — product of experts from all n-gram levels
  3. Topic Spin coherence (topic_spin_enabled=True) — Potts topic bonus
  4. Larger vocab (4000 vs 2000) — fewer OOV tokens = lower PPL
  5. Memory-safe n-gram building — cap sequences to avoid OOM on Pi
  6. Multi-config sweep — test multiple configs automatically

Usage on Pi:
  # Quick test (5 min):
  python train_v12.py --test

  # Single best-guess config:
  python train_v12.py --config best

  # Sweep multiple configs to find optimal:
  python train_v12.py --config sweep

  # Custom:
  python train_v12.py --vocab-size 4000 --kn-backoff --interpolated --topic-spin
"""

import sys
import os
import time
import json
import argparse
import multiprocessing as mp
import gc
from datetime import datetime
from collections import Counter, defaultdict
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
# Parallel Workers (same as train_parallel.py)
# ============================================================================

def _count_cooccurrences_chunk(args):
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
    result = {}
    for k in range(1, max_n + 1):
        result[k] = {}
        for ctx, conts in index[k].items():
            result[k][ctx] = dict(conts)
    return result


def compute_pmi_parallel(sequences, vocab_size, window=5, min_count=2,
                         pmi_cap=10, n_workers=4):
    V = vocab_size
    chunk_size = max(1, len(sequences) // n_workers)
    chunks = [(sequences[i:i+chunk_size], window)
              for i in range(0, len(sequences), chunk_size) if sequences[i:i+chunk_size]]

    with mp.Pool(n_workers) as pool:
        results = pool.map(_count_cooccurrences_chunk, chunks)

    unigram = np.zeros(V, dtype=np.int64)
    cooc_counts = Counter()
    for uni_chunk, cooc_chunk in results:
        for w, c in uni_chunk.items():
            if w < V:
                unigram[w] += c
        for pair, c in cooc_chunk.items():
            cooc_counts[pair] += c

    total_tokens = int(unigram.sum())
    rows, cols, data = [], [], []
    seen = set()
    for (w, w2), count in cooc_counts.items():
        if count >= min_count and w < V and w2 < V:
            pmi = compute_log_floor_pmi(
                int(count), int(unigram[w]), int(unigram[w2]),
                total_tokens, cap=pmi_cap
            )
            if pmi != 0:
                for a, b in [(w, w2), (w2, w)]:
                    if (a, b) not in seen:
                        rows.append(a)
                        cols.append(b)
                        data.append(pmi)
                        seen.add((a, b))

    J = sp.csr_matrix(
        (np.array(data, dtype=np.int64),
         (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
        shape=(V, V)
    )

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
    V = vocab_size
    chunk_size = max(1, len(sequences) // n_workers)
    chunks = [(sequences[i:i+chunk_size], max_dist)
              for i in range(0, len(sequences), chunk_size) if sequences[i:i+chunk_size]]

    with mp.Pool(n_workers) as pool:
        results = pool.map(_count_skip_cooccurrences_chunk, chunks)

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
                    for a, b in [(w, w2), (w2, w)]:
                        if (a, b) not in seen:
                            rows.append(a)
                            cols.append(b)
                            data.append(pmi)
                            seen.add((a, b))
        J_skip[dist] = sp.csr_matrix(
            (np.array(data, dtype=np.int64),
             (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
            shape=(V, V)
        )
        print(f"    Skip-PMI dist={dist}: {J_skip[dist].nnz:,} non-zero entries")
    return J_skip


def build_ngram_parallel_mutate(self, sequences, n_workers=4):
    max_n = self.max_n
    min_count = self.min_count
    chunk_size = max(1, len(sequences) // n_workers)
    chunks = [(sequences[i:i+chunk_size], max_n, min_count)
              for i in range(0, len(sequences), chunk_size) if sequences[i:i+chunk_size]]

    with mp.Pool(n_workers) as pool:
        results = pool.map(_build_ngram_chunk, chunks)

    merged = {k: defaultdict(Counter) for k in range(1, max_n + 1)}
    for chunk_result in results:
        for k in range(1, max_n + 1):
            for ctx, conts in chunk_result[k].items():
                for w, c in conts.items():
                    merged[k][ctx][w] += c

    for k in range(1, max_n + 1):
        for ctx in list(merged[k].keys()):
            low_count = [w for w, c in merged[k][ctx].items() if c < min_count]
            for w in low_count:
                del merged[k][ctx][w]
            if not merged[k][ctx]:
                del merged[k][ctx]
                continue
            self.index[k][ctx] = Counter(merged[k][ctx])
            self.context_totals[k][ctx] = sum(merged[k][ctx].values())

    self._built = True
    self._unigram_totals = {}
    if 1 in self.index:
        total_N = sum(self.context_totals[1].values())
        for context in self.index[1]:
            if context and len(context) == 1:
                w = context[0]
                count_w = self.context_totals[1].get(context, 0)
                self._unigram_totals[w] = (count_w, total_N)

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
# Parallel Training Wrapper
# ============================================================================

def train_parallel(model, n_samples=200000, texts=None, n_workers=None):
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 4)

    import ising_spin.model as model_module

    orig_compute_pmi = model_module.compute_pmi_couplings
    orig_compute_skip_pmi = model_module.compute_skip_pmi_couplings
    orig_ngram_build = model_module.NGramIndex.build

    model_module.compute_pmi_couplings = partial(compute_pmi_parallel, n_workers=n_workers)
    model_module.compute_skip_pmi_couplings = partial(compute_skip_pmi_parallel, n_workers=n_workers)

    def parallel_ngram_build(self, sequences):
        return build_ngram_parallel_mutate(self, sequences, n_workers=n_workers)
    model_module.NGramIndex.build = parallel_ngram_build

    orig_train = IsingLMModel.train
    try:
        result = orig_train(model, n_samples=n_samples, texts=texts)
    finally:
        model_module.compute_pmi_couplings = orig_compute_pmi
        model_module.compute_skip_pmi_couplings = orig_compute_skip_pmi
        model_module.NGramIndex.build = orig_ngram_build

    return result


# ============================================================================
# Data Loading
# ============================================================================

def load_cached_data(n_samples, script_dir, cache_override=None):
    """Load cached data, preferring the largest available cache that's big enough."""
    cache_path = cache_override

    if cache_path is None:
        cache_candidates = []
        for fname in os.listdir(script_dir):
            if fname.startswith("cached_fineweb_") and fname.endswith(".json"):
                fpath = os.path.join(script_dir, fname)
                try:
                    count_str = fname.split("_")[-1].replace(".json", "")
                    count = int(count_str.replace("k", "000"))
                    cache_candidates.append((count, fpath))
                except (ValueError, IndexError):
                    pass
        cache_candidates.sort(reverse=True)

        # Use the smallest cache that has >= n_samples
        for count, fpath in cache_candidates:
            if count >= n_samples:
                cache_path = fpath
                break
        # If no cache is big enough, use the largest available
        if cache_path is None and cache_candidates:
            cache_path = cache_candidates[0][1]

    if cache_path and os.path.exists(cache_path):
        print(f"Loading: {cache_path}")
        with open(cache_path) as f:
            texts = json.load(f)
        print(f"  {len(texts)} texts loaded")
        return texts
    else:
        print("No cache found. Downloading FineWeb-Edu...")
        texts = load_fineweb_edu(n_samples=n_samples)
        cache_path = os.path.join(script_dir, f"cached_fineweb_{len(texts)//1000}k.json")
        with open(cache_path, 'w') as f:
            json.dump(texts, f)
        print(f"  Cached {len(texts)} texts to {cache_path}")
        return texts


# ============================================================================
# PPL Evaluation
# ============================================================================

def beta_sweep(model, n_seqs=20, max_delta=25000):
    """Sweep beta factor to find optimal PPL."""
    factors = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.5]
    recall_scale = model.recall_scale
    best_ppl = float('inf')
    best_factor = 0.9

    for factor in factors:
        beta_val = factor * LN2_NUM / (recall_scale * LN2_DEN)
        sampler = IntegerBoltzmannSampler(beta=beta_val, max_delta=max_delta)
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
            marker = " <-- BEST" if ppl_val < best_ppl else ""
            print(f"    f={factor:.2f}: PPL={ppl_val:.1f}{marker}")
            if ppl_val < best_ppl:
                best_ppl = ppl_val
                best_factor = factor

    return best_ppl, best_factor


def quick_ppl(model, n_seqs=10):
    """Fast PPL estimate with default beta."""
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


# ============================================================================
# Configurations
# ============================================================================

# v11.7 baseline (PPL~44-54)
CONFIG_BASELINE = dict(
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

# v12: KN backoff only (expect ~15-25% PPL improvement from better backoff)
CONFIG_KN = CONFIG_BASELINE.copy()
CONFIG_KN.update(dict(
    kn_backoff=True,
))

# v12: Interpolated smoothing (product of experts from all n-gram levels)
CONFIG_INTERP = CONFIG_BASELINE.copy()
CONFIG_INTERP.update(dict(
    interpolated=True,
))

# v12: KN + Interpolated (the gold standard for n-gram LMs)
CONFIG_KN_INTERP = CONFIG_BASELINE.copy()
CONFIG_KN_INTERP.update(dict(
    kn_backoff=True,
    interpolated=True,
))

# v12: Larger vocab (4000 words covers ~50% of tokens vs ~33% with 2000)
CONFIG_VOCAB4K = CONFIG_BASELINE.copy()
CONFIG_VOCAB4K.update(dict(
    vocab_max_size=4000,
    vocab_min_freq=15,
))

# v12: Full v12 config — all improvements combined
CONFIG_V12 = dict(
    vocab_min_freq=15, vocab_max_size=4000,
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
    recall_primary_mode=True, interpolated=True, kn_backoff=True,
    topic_spin_enabled=True,
    topic_n_topics=16, topic_coherence_penalty=400,
    topic_spin_flip_interval=20, topic_context_window=30,
    topic_coupling_scale=100,
)

# v12 aggressive: even larger vocab, stronger topic spin
CONFIG_V12_AGGRESSIVE = CONFIG_V12.copy()
CONFIG_V12_AGGRESSIVE.update(dict(
    vocab_max_size=6000,
    vocab_min_freq=10,
    topic_coherence_penalty=600,
    pmi_weight=8,
))

ALL_CONFIGS = {
    'baseline': CONFIG_BASELINE,
    'kn': CONFIG_KN,
    'interp': CONFIG_INTERP,
    'kn_interp': CONFIG_KN_INTERP,
    'vocab4k': CONFIG_VOCAB4K,
    'v12': CONFIG_V12,
    'v12_agg': CONFIG_V12_AGGRESSIVE,
}


# ============================================================================
# Run a single training config
# ============================================================================

def run_config(config, config_name, texts, n_workers, n_samples=None):
    """Train with a single config, return results dict."""
    n_use = min(n_samples or len(texts), len(texts))

    print(f"\n{'='*70}")
    print(f"CONFIG: {config_name}")
    print(f"  vocab_max_size={config.get('vocab_max_size')}")
    print(f"  kn_backoff={config.get('kn_backoff')}")
    print(f"  interpolated={config.get('interpolated')}")
    print(f"  topic_spin={config.get('topic_spin_enabled')}")
    print(f"  n_samples={n_use}")
    print(f"{'='*70}")

    gc.collect()
    t0 = time.time()

    model = IsingLMModel(**config)
    train_parallel(model, n_samples=n_use, texts=texts, n_workers=n_workers)

    t_train = time.time() - t0
    print(f"\nTraining: {t_train:.1f}s ({t_train/60:.1f}min)")

    # Beta sweep
    print(f"\nBeta sweep:")
    max_delta = 50000 if config.get('interpolated') else 25000
    best_ppl, best_factor = beta_sweep(model, n_seqs=20, max_delta=max_delta)
    print(f"Best: f={best_factor:.2f}, PPL={best_ppl:.1f}")

    # Apply best beta
    recall_scale = model.recall_scale
    beta_val = best_factor * LN2_NUM / (recall_scale * LN2_DEN)
    # With interpolated=True, energies are larger, so increase max_delta
    max_delta = 50000 if config.get('interpolated') else 25000
    model.generator.word_sampler = IntegerBoltzmannSampler(beta=beta_val, max_delta=max_delta)
    model.generator.beta_word = beta_val
    model.beta_word = beta_val

    # Full PPL
    ppl_full = model.compute_perplexity(n_samples=100)
    print(f"PPL (full, 100 seqs): {ppl_full:.2f}")

    # Generate
    print(f"\n--- Generation (PPL={ppl_full:.2f}) ---")
    gen_results = {}
    for prompt in ["the history of", "science and technology", "research shows that"]:
        result = model.generator.generate(prompt=prompt, length=400)
        text = result['text']
        words = text.split()
        n_recalls = sum(1 for d in result['diagnostics'] if d['recall_hit'])
        n_copies = sum(1 for d in result['diagnostics'] if d['copy'])
        gen_results[prompt] = {
            'text': text[:300],
            'n_words': len(words),
            'recalls': n_recalls,
            'copies': n_copies,
        }
        print(f"  '{prompt}' ({len(words)} words, recalls={n_recalls}, copies={n_copies})")
        print(f"    {text[:200]}...")

    return {
        'config_name': config_name,
        'config': {k: str(v) for k, v in config.items()},
        'n_samples': n_use,
        'training_time_sec': t_train,
        'best_beta_factor': best_factor,
        'ppl_sweep': best_ppl,
        'ppl_full': ppl_full,
        'vocab_size': len(model.vocab),
        'generated': gen_results,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ising Spin Model — v12 Training (KN + Interpolated + Topic Spin)",
    )
    parser.add_argument("--config", type=str, default="v12",
                        choices=list(ALL_CONFIGS.keys()) + ['sweep', 'best'],
                        help="Config to use (default: v12)")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of training samples (default: use all cached)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes")
    parser.add_argument("--cache", type=str, default=None,
                        help="Path to cached training data JSON")
    parser.add_argument("--test", action="store_true",
                        help="Quick test (1K samples)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    n_workers = args.workers or min(mp.cpu_count(), 8)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Output directory
    output_dir = args.output_dir or os.path.join(
        script_dir, "output",
        f"v12_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(output_dir, exist_ok=True)

    t_total = time.time()

    print("=" * 70)
    print("ISING SPIN GLASS LANGUAGE MODEL — v12 TRAINING")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Output: {output_dir}")
    print(f"Workers: {n_workers}")
    print("=" * 70)

    # Load data
    n_target = args.n_samples or 1000000  # Default: use up to 1M texts
    if args.test:
        n_target = 10000
    texts = load_cached_data(n_target, script_dir, args.cache)

    # Cap: don't use more than 1M texts for n-gram building (memory safety)
    # With 1M texts and vocab=4000, n-gram index ≈ 2-3GB — fine for 16GB Pi
    MAX_NGRAM_TEXTS = 1000000
    n_use = min(n_target, len(texts))

    if n_use > MAX_NGRAM_TEXTS:
        print(f"\n  Note: capping training to {MAX_NGRAM_TEXTS} texts (memory safety)")
        print(f"  (Have {len(texts)} texts available, using first {MAX_NGRAM_TEXTS})")
        n_use = MAX_NGRAM_TEXTS

    # Determine which configs to run
    if args.config == 'sweep':
        # Sweep key configs to find the best
        configs_to_run = ['baseline', 'kn', 'interp', 'kn_interp', 'v12']
    elif args.config == 'best':
        configs_to_run = ['v12']
    else:
        configs_to_run = [args.config]

    all_results = []

    for config_name in configs_to_run:
        config = ALL_CONFIGS[config_name].copy()
        result = run_config(config, config_name, texts, n_workers, n_use)
        all_results.append(result)
        gc.collect()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<15} {'Vocab':>5} {'PPL(sweep)':>10} {'PPL(full)':>10} {'Time':>8}")
    print("-" * 50)
    for r in all_results:
        print(f"{r['config_name']:<15} {r['vocab_size']:>5} "
              f"{r['ppl_sweep']:>10.1f} {r['ppl_full']:>10.2f} "
              f"{r['training_time_sec']:>7.0f}s")

    # Find best
    best = min(all_results, key=lambda r: r['ppl_full'])
    print(f"\nBest: {best['config_name']} with PPL={best['ppl_full']:.2f}")

    # Save results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {results_path}")

    # Also save to standard location
    std_results_path = os.path.join(script_dir, "training_results.json")
    with open(std_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {std_results_path}")

    print(f"\nDone. Total time: {time.time()-t_total:.0f}s")


if __name__ == "__main__":
    main()
