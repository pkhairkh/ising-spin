#!/usr/bin/env python3
"""
Test v8.2: Integer-only + Topic Spin Potts coherence.
Generate 400 tokens WITH topic coherence.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ising_spin.model import IsingLMModel


def main():
    print("=" * 70)
    print("ISING SPIN v8.2 — Integer-Only + Topic Spin Potts Coherence")
    print("=" * 70)
    print()

    t0 = time.time()

    # ======================================================================
    # Test 1: WITHOUT topic spin (baseline)
    # ======================================================================
    print("TRAINING v8.2 WITHOUT Topic Spin (baseline)...")
    model_no_topic = IsingLMModel(
        vocab_min_freq=15,
        vocab_max_size=5000,
        ngram_max_n=5,
        ngram_min_count=2,
        pmi_window=5,
        pmi_min_count=2,
        pmi_cap=10,
        recall_scale=800,
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
        auto_calibrate_beta=True,
        recall_primary_mode=True,
        topic_spin_enabled=False,  # NO topic spin
    )
    model_no_topic.train(n_samples=50000)
    ppl_no_topic = model_no_topic.compute_perplexity(n_samples=50)
    print(f"  PPL (no topic): {ppl_no_topic:.2f}")

    # Generate 400 tokens without topic
    result_no_topic = model_no_topic.generator.generate(prompt="the history of", length=400)
    text_no_topic = result_no_topic['text']

    # ======================================================================
    # Test 2: WITH Topic Spin (coherence)
    # ======================================================================
    print("\n" + "=" * 70)
    print("TRAINING v8.2 WITH Topic Spin Potts Coherence...")
    print("=" * 70)

    model_topic = IsingLMModel(
        vocab_min_freq=15,
        vocab_max_size=5000,
        ngram_max_n=5,
        ngram_min_count=2,
        pmi_window=5,
        pmi_min_count=2,
        pmi_cap=10,
        recall_scale=800,
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
        auto_calibrate_beta=True,
        recall_primary_mode=True,
        # v8.2: Topic Spin ENABLED
        topic_spin_enabled=True,
        topic_n_topics=16,
        topic_coherence_penalty=400,
        topic_spin_flip_interval=20,
        topic_context_window=30,
        topic_coupling_scale=100,
    )
    model_topic.train(n_samples=50000)
    ppl_topic = model_topic.compute_perplexity(n_samples=50)
    print(f"  PPL (with topic): {ppl_topic:.2f}")

    # Generate 400 tokens with topic
    result_topic = model_topic.generator.generate(prompt="the history of", length=400)
    text_topic = result_topic['text']

    # ======================================================================
    # Compare results
    # ======================================================================
    print("\n" + "=" * 70)
    print("COMPARISON: No Topic Spin vs With Topic Spin")
    print("=" * 70)

    print(f"\nPPL without topic: {ppl_no_topic:.2f}")
    print(f"PPL with topic:    {ppl_topic:.2f}")

    print(f"\n--- WITHOUT Topic Spin ({len(text_no_topic.split())} words) ---")
    print(text_no_topic[:500])
    print("...")
    print(text_no_topic[-200:])

    print(f"\n--- WITH Topic Spin ({len(text_topic.split())} words) ---")
    print(text_topic[:500])
    print("...")
    print(text_topic[-200:])

    # Topic spin diagnostics
    if model_topic.topic_spin_layer is not None:
        td = model_topic.topic_spin_layer.get_diagnostics()
        print(f"\n  Topic Spin Diagnostics:")
        print(f"    Spin flips: {td['spin_flips']}")
        print(f"    Coherence penalties: {td['coherence_penalties']}")
        print(f"    Penalty rate: {td['penalty_rate']:.1%}")

    # Save results
    output_path = "/home/z/my-project/download/ising_v82_comparison.txt"
    with open(output_path, 'w') as f:
        f.write("Ising Spin Glass v8.2 — Integer-Only + Topic Spin Comparison\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"PPL without topic: {ppl_no_topic:.2f}\n")
        f.write(f"PPL with topic:    {ppl_topic:.2f}\n\n")
        f.write("--- WITHOUT Topic Spin ---\n")
        f.write(text_no_topic + "\n\n")
        f.write("--- WITH Topic Spin ---\n")
        f.write(text_topic + "\n\n")
        if model_topic.topic_spin_layer is not None:
            td = model_topic.topic_spin_layer.get_diagnostics()
            f.write(f"Topic Spin Diagnostics:\n")
            f.write(f"  Spin flips: {td['spin_flips']}\n")
            f.write(f"  Coherence penalties: {td['coherence_penalties']}\n")
            f.write(f"  Penalty rate: {td['penalty_rate']:.1%}\n")

    print(f"\nResults saved to: {output_path}")

    t_total = time.time() - t0
    print(f"Total time: {t_total:.1f}s")


if __name__ == "__main__":
    main()
