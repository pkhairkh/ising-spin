#!/usr/bin/env python3
"""
Generate 400 tokens of text using Ising Spin Model v8.1.
Recall-primary architecture, FineWeb-Edu training data.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ising_spin.model import IsingLMModel


def main():
    print("=" * 70)
    print("ISING SPIN GLASS LANGUAGE MODEL — v8.1 400-Token Generation")
    print("=" * 70)
    print()

    t0 = time.time()

    model = IsingLMModel(
        # Vocabulary — v8.1: Smaller, cleaner vocab for better n-gram density
        vocab_min_freq=15,
        vocab_max_size=5000,

        # N-gram settings
        ngram_max_n=5,
        ngram_min_count=2,
        pmi_window=5,
        pmi_min_count=2,
        pmi_cap=10,

        # Energy scales — v8.1: Recall is PRIMARY, all others DISABLED
        recall_scale=800,
        pmi_weight=0,
        field_weight=1,
        knowledge_scale=0,
        spin3_scale=0,
        category_scale=0,
        logic_rule_scale=0,
        logic_hard_scale=0,

        # Sampling parameters
        beta_type=0.001,
        beta_word=0.001,
        copy_enabled=True,
        copy_min_context=2,
        copy_min_confidence=0.25,
        same_word_penalty=500,
        max_closed_class_run=2,
        ising_enabled=False,
        skip_pmi_max_dist=5,

        # MCMC disabled
        mcmc_refine_steps=0,

        # No ConceptNet
        use_conceptnet=False,

        # Walsh disabled
        walsh_enabled=False,
        walsh_subspace_rank=64,
        walsh_max_order=2,
        walsh_weight=1,
        walsh_min_coeff=3,

        # Graded couplings disabled
        graded_couplings_enabled=False,
        coupling_scale=1000,
        trigram_scale=2000,
        auto_calibrate_beta=True,

        # Recall-primary mode
        recall_primary_mode=True,
    )

    # Train on 50K samples from FineWeb-Edu
    print("Training on 50K FineWeb-Edu samples...")
    model.train(n_samples=50000)

    t_train = time.time() - t0
    print(f"\nTraining time: {t_train:.1f}s")

    # Quick PPL check
    ppl = model.compute_perplexity(n_samples=50)
    print(f"  PPL: {ppl:.2f}")

    # ======================================================================
    # GENERATE 400 TOKENS
    # ======================================================================
    print("\n" + "=" * 70)
    print("GENERATING 400 TOKENS")
    print("=" * 70)

    # Generate from a natural prompt
    prompts_to_try = [
        "the history of",
        "in the world",
        "science and technology",
        "education is the",
        "research shows that",
    ]

    for prompt in prompts_to_try:
        print(f"\n--- Prompt: '{prompt}' ---")
        result = model.generator.generate(prompt=prompt, length=400)
        text = result['text']

        # Count words
        words = text.split()
        print(f"Generated {len(words)} words (tokens)")
        print(f"\n{text}")

        # Diagnostics
        n_recalls = sum(1 for d in result['diagnostics'] if d['recall_hit'])
        n_pmi = sum(1 for d in result['diagnostics'] if not d['recall_hit'])
        n_copies = sum(1 for d in result['diagnostics'] if d['copy'])
        print(f"\n  Diagnostics: recalls={n_recalls} pmi_only={n_pmi} copies={n_copies}")

        # Save the first (best) one to file
        if prompt == prompts_to_try[0]:
            output_path = "/home/z/my-project/download/ising_v81_400tokens.txt"
            with open(output_path, 'w') as f:
                f.write(f"Ising Spin Glass Language Model v8.1 — 400-Token Generation\n")
                f.write(f"Prompt: '{prompt}'\n")
                f.write(f"PPL: {ppl:.2f}\n")
                f.write(f"Training: 50K FineWeb-Edu samples\n")
                f.write(f"Architecture: Recall-primary (all other layers disabled)\n")
                f.write(f"recall_scale=800, β = 0.85*ln(2)/scale\n")
                f.write(f"=" * 70 + "\n\n")
                f.write(text)
                f.write(f"\n\n--- Diagnostics ---\n")
                f.write(f"Words: {len(words)}\n")
                f.write(f"Recall hits: {n_recalls}\n")
                f.write(f"PMI-only: {n_pmi}\n")
                f.write(f"Copy hits: {n_copies}\n")
            print(f"\n  Saved to: {output_path}")

    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.1f}s")


if __name__ == "__main__":
    main()
