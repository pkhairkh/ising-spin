#!/usr/bin/env python3
"""
v8.1 Thorough Training — FineWeb-Edu at scale.
Optimized for recall-primary mode (skip PMI, skip knowledge, skip graded).
"""

import sys
import os
import time
import math
import gc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ising_spin.model import IsingLMModel, IntegerBoltzmannSampler


def train_and_eval(n_samples, vocab_max_size=8000, vocab_min_freq=5,
                   ngram_min_count=2, recall_scale=800, label=""):
    """Train and evaluate a model with given parameters."""
    print("\n" + "=" * 70)
    print(f"TRAINING: {label}")
    print("=" * 70)

    t0 = time.time()

    model = IsingLMModel(
        vocab_min_freq=vocab_min_freq,
        vocab_max_size=vocab_max_size,
        ngram_max_n=5,
        ngram_min_count=ngram_min_count,
        pmi_window=5,
        pmi_min_count=2,
        pmi_cap=10,

        # v8.0 Recall-primary
        recall_scale=recall_scale,
        pmi_weight=0,
        field_weight=1,
        knowledge_scale=0,
        spin3_scale=0,
        category_scale=0,
        logic_rule_scale=0,
        logic_hard_scale=0,

        beta_type=0.001,
        beta_word=0.001,
        copy_enabled=True,
        copy_min_context=2,
        copy_min_confidence=0.25,
        same_word_penalty=500,
        max_closed_class_run=2,
        ising_enabled=False,
        skip_pmi_max_dist=5,
        mcmc_refine_steps=0,
        use_conceptnet=False,
        walsh_enabled=False,
        graded_couplings_enabled=False,
        recall_primary_mode=True,
        auto_calibrate_beta=True,
    )

    model.train(n_samples=n_samples)
    t_train = time.time() - t0
    print(f"\nTraining time: {t_train:.1f}s")

    # PPL evaluation
    ppl = model.compute_perplexity(n_samples=100)
    print(f"  PPL: {ppl:.2f}")

    # Generation test
    prompts = ["the", "science", "research", "students", "education",
                "he", "they", "water", "school", "city",
                "the study", "in the", "of the", "a new"]
    print(f"\n  Generation samples:")
    for p in prompts:
        r = model.generator.generate(prompt=p, length=12)
        print(f"    {p:>14} -> {r['text']}")

    # Stats
    gen = model.generator
    ngram_sizes = {}
    for k in gen.ngram_index.index:
        ngram_sizes[k] = len(gen.ngram_index.index[k])
    total_contexts = sum(ngram_sizes.values())
    print(f"\n  Vocab: {model.vocab_size:,}")
    print(f"  N-gram contexts: {total_contexts:,}  {ngram_sizes}")
    print(f"  β_word: {gen.beta_word:.6f}")

    result = {
        'label': label,
        'n_samples': n_samples,
        'vocab_size': model.vocab_size,
        'ppl': ppl,
        'train_time': t_train,
        'ngram_contexts': total_contexts,
        'ngram_sizes': ngram_sizes,
    }

    del model
    gc.collect()

    return result


def main():
    print("=" * 70)
    print("v8.1 THOROUGH TRAINING — FineWeb-Edu at Scale")
    print("=" * 70)
    print("\nRecall-Primary Architecture:")
    print("  E = log₂(1/P) * scale → correct Boltzmann energy")
    print("  β ≈ 0.85*ln(2)/scale → empirically optimal")
    print("  All other layers DISABLED (they hurt PPL)")
    print("  Optimized: skip PMI, skip knowledge, skip graded couplings")
    print()

    results = []

    # ======================================================================
    # Experiment 1: Same config as v8.0 (20K) — baseline reference
    # ======================================================================
    r = train_and_eval(
        n_samples=20000, vocab_max_size=8000, vocab_min_freq=3,
        ngram_min_count=1, recall_scale=800,
        label="20K/8K-vocab/min1 (v8.0 baseline)"
    )
    results.append(r)

    # ======================================================================
    # Experiment 2: 50K, same vocab
    # ======================================================================
    r = train_and_eval(
        n_samples=50000, vocab_max_size=8000, vocab_min_freq=5,
        ngram_min_count=2, recall_scale=800,
        label="50K/8K-vocab/min2"
    )
    results.append(r)

    # ======================================================================
    # Experiment 3: 100K, same vocab
    # ======================================================================
    r = train_and_eval(
        n_samples=100000, vocab_max_size=8000, vocab_min_freq=5,
        ngram_min_count=2, recall_scale=800,
        label="100K/8K-vocab/min2"
    )
    results.append(r)

    # ======================================================================
    # Experiment 4: 200K, larger vocab
    # ======================================================================
    r = train_and_eval(
        n_samples=200000, vocab_max_size=12000, vocab_min_freq=5,
        ngram_min_count=2, recall_scale=800,
        label="200K/12K-vocab/min2"
    )
    results.append(r)

    # ======================================================================
    # Summary
    # ======================================================================
    print("\n" + "=" * 70)
    print("TRAINING SCALE COMPARISON — v8.1")
    print("=" * 70)
    print(f"\n  {'Config':<40} {'PPL':>8} {'Vocab':>7} {'N-gram ctx':>12} {'Time':>8}")
    print(f"  {'-'*40} {'-'*8} {'-'*7} {'-'*12} {'-'*8}")
    for r in results:
        print(f"  {r['label']:<40} {r['ppl']:>8.1f} {r['vocab_size']:>7,} {r['ngram_contexts']:>12,} {r['train_time']:>7.0f}s")

    best = min(results, key=lambda r: r['ppl'])
    print(f"\n  Best: {best['label']} with PPL = {best['ppl']:.1f}")


if __name__ == "__main__":
    main()
