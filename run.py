#!/usr/bin/env python3
"""
Ising Spin Glass Language Model — Main Runner
==============================================

Best known configuration (v11.7): PPL=51.54

Architecture:
  - 200K FineWeb-Edu training samples
  - 2K vocabulary (min_freq=25)
  - 5-gram recall index (min_count=2)
  - PMI backoff (weight=5)
  - Recall-primary energy: E = log2(1/P) * recall_scale
  - scale=1600, same_word_penalty=200
  - β calibrated via sweep (typically ~0.9 * ln2/scale)
  - Integer-only Boltzmann sampling (ZERO float ops in inference)

Usage:
  python run.py                    # Full train + eval + generate
  python run.py --eval-only        # Skip training (requires cache)
  python run.py --generate-only    # Only generate text
"""

import sys
import os
import time
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from ising_spin.model import (
    IsingLMModel, IntegerBoltzmannSampler, LN2_NUM, LN2_DEN, LOG2_SCALE
)


# ============================================================================
# Best Configuration (v11.7 — PPL=51.54)
# ============================================================================

BEST_CONFIG = dict(
    # Vocabulary
    vocab_min_freq=25,
    vocab_max_size=2000,

    # N-gram index
    ngram_max_n=5,
    ngram_min_count=2,

    # Energy scales
    recall_scale=1600,
    pmi_weight=5,
    field_weight=1,
    knowledge_scale=0,
    spin3_scale=0,
    category_scale=0,
    logic_rule_scale=0,
    logic_hard_scale=0,

    # Sampling
    beta_type=0.001,
    beta_word=0.001,
    copy_enabled=True,
    copy_min_context=2,
    copy_min_confidence=0.25,
    same_word_penalty=200,
    max_closed_class_run=2,

    # Ising
    ising_enabled=True,
    skip_pmi_max_dist=5,
    mcmc_refine_steps=0,

    # Disabled features
    use_conceptnet=False,
    walsh_enabled=False,
    graded_couplings_enabled=False,
    auto_calibrate_beta=False,
    recall_primary_mode=True,
    interpolated=False,
    kn_backoff=False,
    topic_spin_enabled=False,
)

BEST_N_SAMPLES = 200000
CACHE_PATH = os.path.join(os.path.dirname(__file__), "cached_fineweb_200k.json")


# ============================================================================
# Utilities
# ============================================================================

def load_training_data(path=CACHE_PATH):
    """Load cached FineWeb-Edu training data."""
    if not os.path.exists(path):
        print(f"Cache not found at {path}")
        print("Run: python cache_200k.py   to download and cache data")
        sys.exit(1)
    with open(path) as f:
        texts = json.load(f)
    print(f"Loaded {len(texts)} texts from {path}")
    return texts


def beta_sweep(model, n_seqs=20, factors=None):
    """Sweep β factor to find optimal PPL."""
    if factors is None:
        factors = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.5]

    gen = model.generator
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
        if total_tokens > 0:
            avg_log2 = total_log2_prob / (total_tokens * LOG2_SCALE)
            ppl_val = 2.0 ** (-avg_log2)
            results[factor] = ppl_val
            marker = " <-- BEST" if ppl_val < best_ppl else ""
            print(f"  f={factor:.2f}: PPL={ppl_val:.1f}{marker}")
            if ppl_val < best_ppl:
                best_ppl = ppl_val
                best_factor = factor

    return best_ppl, best_factor, results


def apply_best_beta(model, factor):
    """Apply the best β factor found by sweep."""
    recall_scale = model.recall_scale
    beta_val = factor * LN2_NUM / (recall_scale * LN2_DEN)
    model.generator.word_sampler = IntegerBoltzmannSampler(beta=beta_val, max_delta=25000)
    model.generator.beta_word = beta_val
    model.beta_word = beta_val


def generate_and_save(model, ppl_val, best_factor, results):
    """Generate 400-token texts and save to download directory."""
    print("\n" + "=" * 70)
    print(f"GENERATING 400 TOKENS (PPL={ppl_val:.2f})")
    print("=" * 70)

    prompts = ["the history of", "science and technology", "research shows that"]
    gen_results = {}

    for prompt in prompts:
        result = model.generator.generate(prompt=prompt, length=400)
        text = result['text']
        words = text.split()
        n_recalls = sum(1 for d in result['diagnostics'] if d['recall_hit'])
        n_copies = sum(1 for d in result['diagnostics'] if d['copy'])
        print(f"\n--- '{prompt}' ({len(words)} words) ---")
        print(text)
        print(f"  recalls={n_recalls} copies={n_copies}")
        gen_results[prompt] = result

    # Save primary output
    result = gen_results[prompts[0]]
    text = result['text']
    words = text.split()
    output_path = "/home/z/my-project/download/ising_v11_400tokens.txt"
    with open(output_path, 'w') as f:
        f.write(f"Ising Spin Glass Language Model v11.7 — 400-Token Generation\n")
        f.write(f"Config: 200K data + 2K vocab + PMI=5 + scale=1600 + max_len=30\n")
        f.write(f"PPL: {ppl_val:.2f}\n")
        f.write(f"Prompt: {prompts[0]}\n")
        f.write(f"Training: 200K FineWeb-Edu samples\n")
        f.write(f"Integer-only: YES (ZERO float operations in inference)\n")
        f.write(f"beta factor: {best_factor:.2f}\n")
        f.write("=" * 70 + "\n\n")
        f.write(text)
        f.write(f"\n\n--- beta Sweep ---\n")
        for factor in sorted(results.keys()):
            f.write(f"  f={factor:.2f}: PPL={results[factor]:.1f}\n")
    print(f"\nSaved to: {output_path}")
    return gen_results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ising Spin Language Model")
    parser.add_argument("--eval-only", action="store_true", help="Skip training")
    parser.add_argument("--generate-only", action="store_true", help="Only generate text")
    parser.add_argument("--n-samples", type=int, default=BEST_N_SAMPLES, help="Training samples")
    parser.add_argument("--n-eval", type=int, default=100, help="Eval sequences")
    parser.add_argument("--n-sweep", type=int, default=20, help="Sweep eval sequences")
    args = parser.parse_args()

    t0 = time.time()

    # Train
    model = IsingLMModel(**BEST_CONFIG)

    if not args.generate_only:
        texts = load_training_data()
        print(f"\nTraining on {args.n_samples} samples...")
        model.train(n_samples=args.n_samples, texts=texts)
        print(f"\nTraining: {time.time()-t0:.1f}s")

    # β sweep
    print("\nβ sweep:")
    best_ppl, best_factor, results = beta_sweep(model, n_seqs=args.n_sweep)
    print(f"\nBEST: f={best_factor:.2f}, PPL={best_ppl:.1f}")

    # Apply best β
    apply_best_beta(model, best_factor)

    # Full PPL
    ppl_full = model.compute_perplexity(n_samples=args.n_eval)
    print(f"\nPPL (full, {args.n_eval} seqs): {ppl_full:.2f}")

    # Generate
    generate_and_save(model, ppl_full, best_factor, results)

    total_time = time.time() - t0
    print(f"\nTotal: {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    main()
