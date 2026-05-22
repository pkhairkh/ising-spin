#!/usr/bin/env python3
"""
Ising Spin Model — Text Generation
===================================

Generate text from the Ising Spin Glass Language Model.

Usage:
  python generate.py                              # Default prompts, 400 tokens
  python generate.py --prompt "the history of"    # Custom prompt
  python generate.py --length 200                 # Shorter output
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


def main():
    parser = argparse.ArgumentParser(description="Ising Spin Model Text Generation")
    parser.add_argument("--prompt", type=str, default=None, help="Generation prompt")
    parser.add_argument("--length", type=int, default=400, help="Number of tokens")
    parser.add_argument("--n-samples", type=int, default=200000, help="Training samples")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
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

    # Quick β calibration
    print("Calibrating β...")
    recall_scale = model.recall_scale
    best_ppl = float('inf')
    best_factor = 0.9
    for factor in [0.7, 0.8, 0.85, 0.9, 0.95, 1.0]:
        beta_val = factor * LN2_NUM / (recall_scale * LN2_DEN)
        sampler = IntegerBoltzmannSampler(beta=beta_val, max_delta=25000)
        total_log2_prob = 0
        total_tokens = 0
        for seq in model.test_sequences[:10]:
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
            if ppl_val < best_ppl:
                best_ppl = ppl_val
                best_factor = factor

    beta_val = best_factor * LN2_NUM / (recall_scale * LN2_DEN)
    model.generator.word_sampler = IntegerBoltzmannSampler(beta=beta_val, max_delta=25000)
    model.generator.beta_word = beta_val
    print(f"Best β factor: {best_factor:.2f}")

    # Generate
    prompts = [args.prompt] if args.prompt else [
        "the history of", "science and technology", "research shows that"
    ]

    for prompt in prompts:
        result = model.generator.generate(prompt=prompt, length=args.length)
        text = result['text']
        words = text.split()
        n_recalls = sum(1 for d in result['diagnostics'] if d['recall_hit'])
        n_copies = sum(1 for d in result['diagnostics'] if d['copy'])
        print(f"\n--- '{prompt}' ({len(words)} words) ---")
        print(text)
        print(f"  recalls={n_recalls} copies={n_copies}")

        # Save if output specified or first prompt
        if args.output or (prompt == prompts[0] and not args.output):
            output_path = args.output or "/home/z/my-project/download/ising_generated.txt"
            with open(output_path, 'w') as f:
                f.write(f"Ising Spin Glass Language Model v11.7\n")
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Tokens: {len(words)}\n")
                f.write(f"Recall hits: {n_recalls}, Copy hits: {n_copies}\n")
                f.write("=" * 70 + "\n\n")
                f.write(text)
            print(f"  Saved to: {output_path}")

    print(f"\nTotal: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    main()
