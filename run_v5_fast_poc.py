#!/usr/bin/env python3
"""
Fast PoC Runner for v5 Ising Spin Language Model.
Uses smaller data and fewer sweeps for quick iteration.
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ising_spin.enhanced_v5_model import EnhancedV5Model
from ising_spin.type_system import IDX2POS, POS2IDX, N_POS


def run_fast_poc():
    print("=" * 70)
    print("ISING SPIN LANGUAGE MODEL v5 — FAST PoC RUN")
    print("=" * 70)

    model = EnhancedV5Model(
        vocab_min_freq=5,
        vocab_max_size=3000,       # smaller vocab for speed
        seq_len=20,                # shorter sequences
        window=5,
        pmi_cap=10,
        grammar_penalty=60,
        emission_bonus=100,
        emission_penalty=500,
        phase1_beta=200,
        phase2_beta=500,
        phase3_beta=1000,
        total_sweeps=120,          # fewer sweeps
        use_caldera=True,
        nmf_factors=64,            # smaller NMF
        nmf_iterations=30,
        nmf_n_top=10,
        use_spacy=True,
        spacy_max_texts=5000,      # limit spaCy processing for speed
        transition_min_count=20,   # STRICTER transitions
        n_replicas=3,              # fewer replicas
        pt_swap_interval=5,
        history_enabled=True,
        momentum_enabled=True,
        momentum_strength=3,
    )

    model.train(n_samples=20000)  # smaller corpus
    model.save("data/v5_fast_model")

    # Generate samples
    print("\n" + "=" * 70)
    print("GENERATION SAMPLES (v5: P0+P1+P2)")
    print("=" * 70)

    prompts = ["the", "in", "a", "science", "research", "education", "students"]
    
    all_results = []
    for prompt in prompts:
        print(f"\n--- Prompt: '{prompt}' ---")
        for trial in range(3):
            try:
                result = model.generate_with_trace(prompt=prompt, length=15)
                text = result['text']
                types = ' '.join(result['types'][:15])
                print(f"  Trial {trial+1}: {text}")
                print(f"           [{types}]")
                all_results.append({
                    "prompt": prompt,
                    "trial": trial + 1,
                    "text": text,
                    "types": result['types'],
                    "energy": result['energy'],
                    "pt_stats": result.get('pt_stats', {}),
                })
            except Exception as e:
                print(f"  Trial {trial+1}: ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Grammar evaluation
    print("\n" + "=" * 70)
    print("GRAMMAR EVALUATION (20 samples)")
    print("=" * 70)

    all_metrics = {}
    n_eval = 20
    all_text = []
    for i in range(n_eval):
        try:
            words, types = model.generate_raw(length=15)
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
    for i in range(15):
        try:
            words, types = model.generate_raw(length=15)
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
            bar = "#" * int(pct / 2)
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
            print(f"  Swap {i}<->{i+1} (B={beta_lo}<->{beta_hi}): "
                  f"{swaps}/{attempts} ({rate:.3f})")

    # Repetition analysis
    print("\n" + "=" * 70)
    print("REPETITION ANALYSIS")
    print("=" * 70)
    rep_count = all_metrics.get("repeated_words", 0)
    total_words = n_eval * 15
    rep_rate = rep_count / max(1, total_words - n_eval)
    print(f"  Repeated adjacent words: {rep_count}/{total_words - n_eval} "
          f"({rep_rate:.3f})")

    # DET→NOUN accuracy
    det_noun = all_metrics.get("det_noun", 0)
    det_non = all_metrics.get("det_non_noun", 0)
    det_total = det_noun + det_non
    if det_total > 0:
        print(f"  DET->NOUN accuracy: {det_noun}/{det_total} ({100*det_noun/det_total:.1f}%)")

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
        "det_noun_accuracy": det_noun / max(1, det_total),
        "sample_texts": all_text[:10],
        "type_distribution": type_counts,
        "transition_min_count": 20,
        "model_version": "v5",
    }
    with open("data/v5_fast_model/poc_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to data/v5_fast_model/poc_results.json")

    return model


if __name__ == "__main__":
    run_fast_poc()
