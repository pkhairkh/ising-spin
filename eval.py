#!/usr/bin/env python3
"""
Ising Spin Model — Standalone PPL Evaluation
=============================================

Evaluates perplexity using integer-only log-probability computation.
Supports β sweep for calibration.

Usage:
  python eval.py                    # Evaluate with default model config
  python eval.py --n-samples 50     # Evaluate on 50 sequences
  python eval.py --beta-factor 0.9  # Use specific β factor
"""

import sys
import os
import time
import json
import argparse
import math

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from ising_spin.model import (
    IsingLMModel, IntegerBoltzmannSampler, LN2_NUM, LN2_DEN, LOG2_SCALE
)


def compute_ppl(model, beta_val, n_seqs=50):
    """Compute PPL with a specific β value."""
    sampler = IntegerBoltzmannSampler(beta=beta_val, max_delta=25000)
    gen = model.generator
    total_log2_prob = 0
    total_tokens = 0
    n_missing = 0

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
                n_missing += 1
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
        return float('inf'), 0, 0

    avg_log2 = total_log2_prob / (total_tokens * LOG2_SCALE)
    ppl_val = 2.0 ** (-avg_log2)
    return ppl_val, total_tokens, n_missing


def beta_sweep(model, n_seqs=20):
    """Sweep β factor to find optimal PPL."""
    factors = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.5]
    recall_scale = model.recall_scale
    best_ppl = float('inf')
    best_factor = 0.9
    results = {}

    for factor in factors:
        beta_val = factor * LN2_NUM / (recall_scale * LN2_DEN)
        ppl_val, n_tok, n_miss = compute_ppl(model, beta_val, n_seqs)
        results[factor] = ppl_val
        marker = " <-- BEST" if ppl_val < best_ppl else ""
        print(f"  f={factor:.2f}: PPL={ppl_val:.1f} ({n_tok} tokens, {n_miss} OOV){marker}")
        if ppl_val < best_ppl:
            best_ppl = ppl_val
            best_factor = factor

    return best_ppl, best_factor, results


def main():
    parser = argparse.ArgumentParser(description="Ising Spin Model PPL Evaluation")
    parser.add_argument("--n-samples", type=int, default=200000, help="Training samples")
    parser.add_argument("--n-seqs", type=int, default=50, help="Eval sequences")
    parser.add_argument("--beta-factor", type=float, default=None, help="Fixed β factor")
    parser.add_argument("--sweep", action="store_true", help="Run β sweep")
    parser.add_argument("--cache", type=str,
                        default=os.path.join(os.path.dirname(__file__), "cached_fineweb_200k.json"),
                        help="Path to cached training data")
    args = parser.parse_args()

    t0 = time.time()

    # Load data and train
    with open(args.cache) as f:
        texts = json.load(f)
    print(f"Loaded {len(texts)} texts")

    model = IsingLMModel(
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
    model.train(n_samples=args.n_samples, texts=texts)
    print(f"\nTraining: {time.time()-t0:.1f}s")

    # Evaluate
    if args.beta_factor is not None:
        recall_scale = model.recall_scale
        beta_val = args.beta_factor * LN2_NUM / (recall_scale * LN2_DEN)
        ppl, n_tok, n_miss = compute_ppl(model, beta_val, args.n_seqs)
        print(f"\nPPL = {ppl:.2f} (β factor={args.beta_factor}, {n_tok} tokens, {n_miss} OOV)")

    if args.sweep or args.beta_factor is None:
        print("\nβ sweep:")
        best_ppl, best_factor, results = beta_sweep(model, args.n_seqs)
        print(f"\nBEST: f={best_factor:.2f}, PPL={best_ppl:.1f}")

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
