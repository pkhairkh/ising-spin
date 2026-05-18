#!/usr/bin/env python3
"""
PoC Runner for v5 Ising Spin Language Model (P0+P1+P2).

Trains and generates text with all mitigations:
  P0: Locally-balanced proposals, Hard transition constraints
  P1: CALDERA NMF, Strengthened emission, Implicational couplings
  P2: Parallel tempering, History-driven target, Non-reversible MCMC
  Fix: min_count=20 for transition constraints
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ising_spin.enhanced_v5_model import EnhancedV5Model
from ising_spin.type_system import IDX2POS, POS2IDX, N_POS


def run_poc():
    print("=" * 70)
    print("ISING SPIN LANGUAGE MODEL v5 — PoC RUN")
    print("(P0+P1+P2: Parallel Tempering, History Target, Momentum, Strict Transitions)")
    print("=" * 70)

    # Train with moderate data for PoC
    model = EnhancedV5Model(
        vocab_min_freq=5,
        vocab_max_size=5000,
        seq_len=25,
        window=6,
        pmi_cap=12,
        grammar_penalty=60,
        emission_bonus=100,
        emission_penalty=500,
        phase1_beta=200,
        phase2_beta=500,
        phase3_beta=1000,
        total_sweeps=200,
        use_caldera=True,
        nmf_factors=128,
        nmf_iterations=50,
        nmf_n_top=15,
        use_spacy=True,
        transition_min_count=20,  # STRICTER: only genuinely common transitions
        n_replicas=4,
        pt_swap_interval=5,
        history_enabled=True,
        momentum_enabled=True,
        momentum_strength=3,
    )

    model.train(n_samples=50000)
    model.save("data/v5_model")

    # Generate samples
    print("\n" + "=" * 70)
    print("GENERATION SAMPLES (v5: P0+P1+P2)")
    print("=" * 70)

    prompts = ["the", "in", "a", "science", "research", "education", "students",
               "the study", "in the", "we"]
    
    all_results = []
    for prompt in prompts:
        print(f"\n--- Prompt: '{prompt}' ---")
        for trial in range(3):
            try:
                result = model.generate_with_trace(prompt=prompt, length=20)
                text = result['text']
                types = ' '.join(result['types'][:20])
                print(f"  Trial {trial+1}: {text}")
                print(f"           [{types}]")
                all_results.append({
                    "prompt": prompt,
                    "trial": trial + 1,
                    "text": text,
                    "types": result['types'],
                    "energy": result['energy'],
                })
            except Exception as e:
                print(f"  Trial {trial+1}: ERROR: {e}")

    # Grammar evaluation
    print("\n" + "=" * 70)
    print("GRAMMAR EVALUATION (30 samples)")
    print("=" * 70)

    all_metrics = {}
    n_eval = 30
    all_text = []
    for i in range(n_eval):
        try:
            words, types = model.generate_raw(length=20)
            metrics = model.evaluate_grammar(words, types)
            text = model.vocab.decode(words)
            all_text.append(text)
            for k, v in metrics.items():
                all_metrics[k] = all_metrics.get(k, 0) + v
        except Exception as e:
            print(f"  Sample {i+1}: ERROR: {e}")

    print(f"\nGrammar patterns across {n_eval} samples:")
    for k, v in sorted(all_metrics.items()):
        per_sample = v / n_eval
        print(f"  {k}: {v} total ({per_sample:.1f}/sample)")

    # Type distribution
    print("\n" + "=" * 70)
    print("POS TYPE DISTRIBUTION")
    print("=" * 70)

    all_types = []
    for i in range(20):
        try:
            words, types = model.generate_raw(length=20)
            all_types.extend(types)
        except:
            pass

    type_counts = {}
    for t in all_types:
        name = IDX2POS.get(t, "UNK")
        type_counts[name] = type_counts.get(name, 0) + 1

    total = sum(type_counts.values())
    if total > 0:
        for name, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            pct = count * 100 / total
            bar = "█" * int(pct / 2)
            print(f"  {name:8s}: {count:4d} ({pct:5.1f}%) {bar}")

    # PT swap statistics
    print("\n" + "=" * 70)
    print("PARALLEL TEMPERING STATISTICS")
    print("=" * 70)
    if hasattr(model.sampler, 'pt_swap_counts'):
        for i in range(len(model.sampler.pt_swap_counts)):
            swaps = model.sampler.pt_swap_counts[i]
            attempts = model.sampler.pt_attempt_counts[i]
            rate = swaps / max(1, attempts)
            beta_lo = model.sampler.pt_betas[i]
            beta_hi = model.sampler.pt_betas[i + 1] if i + 1 < len(model.sampler.pt_betas) else "?"
            print(f"  Swap {i}↔{i+1} (β={beta_lo}↔{beta_hi}): "
                  f"{swaps}/{attempts} ({rate:.3f})")

    # Repetition analysis
    print("\n" + "=" * 70)
    print("REPETITION ANALYSIS")
    print("=" * 70)
    rep_count = all_metrics.get("repeated_words", 0)
    total_words = n_eval * 20
    rep_rate = rep_count / max(1, total_words - n_eval)  # exclude first words
    print(f"  Repeated adjacent words: {rep_count}/{total_words - n_eval} "
          f"({rep_rate:.3f})")

    # Sample output
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUT (10 examples)")
    print("=" * 70)
    for i, text in enumerate(all_text[:10]):
        print(f"  {i+1}. {text}")

    # Save results
    results = {
        "grammar_metrics": all_metrics,
        "n_samples": n_eval,
        "per_sample": {k: v/n_eval for k, v in all_metrics.items()},
        "repetition_rate": rep_rate,
        "sample_texts": all_text[:10],
        "type_distribution": type_counts,
        "transition_min_count": 20,
    }
    with open("data/v5_model/poc_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to data/v5_model/poc_results.json")

    return model


if __name__ == "__main__":
    run_poc()
