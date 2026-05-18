#!/usr/bin/env python3
"""
Generate text from pre-trained v5 model.
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ising_spin.enhanced_v5_model import EnhancedV5Model
from ising_spin.type_system import IDX2POS, POS2IDX, N_POS


def main():
    print("Loading pre-trained v5 model...")
    t0 = time.time()
    model = EnhancedV5Model.load("data/v5_fast_model")
    print(f"Model loaded in {time.time()-t0:.1f}s")
    print(f"Vocab size: {len(model.vocab)}")
    
    # Generate samples
    print("\n" + "=" * 70)
    print("GENERATION SAMPLES (v5: P0+P1+P2)")
    print("=" * 70)

    prompts = ["the", "in", "a", "science", "research", "education", "students",
               "the study", "in the", "we"]
    
    for prompt in prompts:
        print(f"\n--- Prompt: '{prompt}' ---")
        for trial in range(3):
            try:
                result = model.generate_with_trace(prompt=prompt, length=15)
                text = result['text']
                types = ' '.join(result['types'][:15])
                pt = result.get('pt_stats', {})
                print(f"  Trial {trial+1}: {text}")
                print(f"           [{types}]")
                if pt:
                    print(f"           PT swaps: {pt.get('total_swaps', 0)}/{pt.get('total_attempts', 0)} "
                          f"(rate={pt.get('swap_rate', 0):.3f})")
            except Exception as e:
                print(f"  Trial {trial+1}: ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Grammar evaluation
    print("\n" + "=" * 70)
    print("GRAMMAR EVALUATION (15 samples)")
    print("=" * 70)

    all_metrics = {}
    n_eval = 15
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

    # DET→NOUN accuracy
    det_noun = all_metrics.get("det_noun", 0)
    det_non = all_metrics.get("det_non_noun", 0)
    det_total = det_noun + det_non
    if det_total > 0:
        print(f"\n  DET->NOUN accuracy: {det_noun}/{det_total} ({100*det_noun/det_total:.1f}%)")

    # Repetition analysis
    rep_count = all_metrics.get("repeated_words", 0)
    total_words = n_eval * 15
    rep_rate = rep_count / max(1, total_words - n_eval)
    print(f"  Repetition rate: {rep_count}/{total_words - n_eval} ({rep_rate:.3f})")

    # PT swap stats
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

    # Sample output
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUT (all examples)")
    print("=" * 70)
    for i, text in enumerate(all_text):
        print(f"  {i+1}. {text}")


if __name__ == "__main__":
    main()
