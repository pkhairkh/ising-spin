#!/usr/bin/env python3
"""
Ising Spin Glass Language Model — v8.1 Thorough Training.

Scale up from 20K to 200K+ samples from FineWeb-Edu.
Recall-Primary Architecture: E = log₂(1/P) * scale IS the Boltzmann energy.

Key optimizations for large-scale training:
- Increased vocab size (8000 → 16000)
- Higher min_count for n-gram pruning (memory efficiency)
- Larger training set for better n-gram coverage
- β calibrated from recall energy
"""

import sys
import os
import time
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ising_spin.model import IsingLMModel, IntegerBoltzmannSampler


def main():
    # ======================================================================
    # Training scale experiments
    # ======================================================================
    configs = [
        # (n_samples, vocab_size, ngram_min_count, label)
        (50000,   12000, 2, "50K samples, 12K vocab"),
        (100000,  16000, 2, "100K samples, 16K vocab"),
        (200000,  20000, 3, "200K samples, 20K vocab"),
    ]

    results = []

    for n_samples, vocab_size, ngram_min_count, label in configs:
        print("\n" + "=" * 70)
        print(f"TRAINING: {label}")
        print("=" * 70)

        t0 = time.time()

        model = IsingLMModel(
            # Vocabulary — scaled up
            vocab_min_freq=3,
            vocab_max_size=vocab_size,

            # N-gram settings — optimized for larger data
            ngram_max_n=5,
            ngram_min_count=ngram_min_count,
            pmi_window=5,
            pmi_min_count=2,
            pmi_cap=10,

            # Energy scales — v8.0: Recall is PRIMARY
            recall_scale=800,
            pmi_weight=0,                # Disabled (redundant with recall)
            field_weight=1,
            knowledge_scale=0,           # Disabled (hurts PPL)
            spin3_scale=0,
            category_scale=0,
            logic_rule_scale=0,
            logic_hard_scale=0,

            # Sampling — β will be recall-calibrated
            beta_type=0.001,
            beta_word=0.001,
            copy_enabled=True,
            copy_min_context=2,
            copy_min_confidence=0.25,
            same_word_penalty=500,
            max_closed_class_run=2,
            ising_enabled=False,
            skip_pmi_max_dist=5,

            # No MCMC (hurts PPL)
            mcmc_refine_steps=0,

            # No ConceptNet (hurts PPL)
            use_conceptnet=False,

            # No Walsh (legacy)
            walsh_enabled=False,
            walsh_subspace_rank=64,
            walsh_max_order=2,
            walsh_weight=1,
            walsh_min_coeff=3,

            # No graded couplings (redundant with recall)
            graded_couplings_enabled=False,
            coupling_scale=1000,
            trigram_scale=2000,

            # v8.0: Recall-primary mode
            recall_primary_mode=True,
            auto_calibrate_beta=True,
        )

        model.train(n_samples=n_samples)
        t_train = time.time() - t0
        print(f"\nTraining time: {t_train:.1f}s")

        # Evaluate PPL
        ppl = model.compute_perplexity(n_samples=100)
        print(f"  PPL ({label}): {ppl:.2f}")

        # Quick generation test
        prompts = ["the", "science", "research", "students", "education",
                    "he", "they", "water", "school", "city"]
        print(f"\n  Generation samples ({label}):")
        for p in prompts:
            r = model.generator.generate(prompt=p, length=12)
            print(f"    {p:>12} -> {r['text']}")

        # Memory usage
        gen = model.generator
        ngram_sizes = sum(
            len(gen.ngram_index.index[k])
            for k in gen.ngram_index.index
        )
        print(f"\n  N-gram index: {ngram_sizes:,} contexts")
        print(f"  Vocab size: {model.vocab_size:,}")

        results.append({
            'label': label,
            'n_samples': n_samples,
            'vocab_size': model.vocab_size,
            'ppl': ppl,
            'train_time': t_train,
            'ngram_contexts': ngram_sizes,
        })

        # Free memory for next config
        del model

    # ======================================================================
    # Summary
    # ======================================================================
    print("\n" + "=" * 70)
    print("TRAINING SCALE COMPARISON")
    print("=" * 70)
    print(f"\n  {'Config':<35} {'PPL':>8} {'Vocab':>7} {'N-gram ctx':>12} {'Time':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*7} {'-'*12} {'-'*8}")
    for r in results:
        print(f"  {r['label']:<35} {r['ppl']:>8.1f} {r['vocab_size']:>7,} {r['ngram_contexts']:>12,} {r['train_time']:>7.0f}s")

    best = min(results, key=lambda r: r['ppl'])
    print(f"\n  Best: {best['label']} with PPL = {best['ppl']:.1f}")


if __name__ == "__main__":
    main()
