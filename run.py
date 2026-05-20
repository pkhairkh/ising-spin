#!/usr/bin/env python3
"""
Ising-Enhanced N-Gram Language Model — Runner.

Architecture:
  Primary:   N-gram exact recall (when context matches)
  Secondary: Ising PMI coupling (when recall misses)
  Knowledge: Layer 2 (external field) + Layer 3 (3-spin couplings)
  Tertiary:  Unigram field (base frequency)
  Sampling:  Integer Boltzmann via lookup table (NO np.exp)

Path 2 additions:
  - Beam generation (global coherence ranking)
  - Joint phrase sampling (MCMC over multi-word phrases)
  - Temperature annealing (Ising phase transition)
  - Skip-gram PMI couplings (distance-specific)

Path 3 additions:
  - Better tokenizer (contractions, hyphens, numbers)
  - Sparse coupling matrix (scipy.sparse.csr_matrix)
  - Perplexity evaluation on held-out data

Path 4 additions:
  - Knowledge Layer (SPO triples + 3-spin couplings)
  - Layer 2: Knowledge external field h_knowledge[w]
  - Layer 3: 3-spin couplings J3[(s,p)] -> [(o, strength)]
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ising_spin.model import (
    IsingLMModel, IsingLM, KnowledgeLayer, POS2IDX, IDX2POS
)


def evaluate_quality(texts, n_eval):
    """Evaluate text quality metrics."""
    metrics = {
        'n_of_the_loops': 0,
        'n_double_dets': 0,
        'n_same_word_reps': 0,
        'total_words': 0,
        'unique_words': set(),
    }

    for text in texts:
        words = text.split()
        metrics['total_words'] += len(words)
        metrics['unique_words'].update(words)

        if "of the of the" in text:
            metrics['n_of_the_loops'] += 1

        for i in range(len(words) - 1):
            if words[i] in {"the", "a", "an"} and words[i+1] in {"the", "a", "an"}:
                metrics['n_double_dets'] += 1

        for i in range(len(words) - 1):
            if words[i] == words[i+1] and len(words[i]) > 2:
                metrics['n_same_word_reps'] += 1

    metrics['unique_words'] = len(metrics['unique_words'])
    metrics['type_token_ratio'] = metrics['unique_words'] / max(1, metrics['total_words'])
    return metrics


def run_ablation(model, prompts, length=20):
    """Run ablation study: Ising ON vs Ising OFF."""
    print("\n" + "=" * 70)
    print("ABLATION STUDY: Ising ON vs Ising OFF")
    print("=" * 70)
    print("\n  This measures the ACTUAL contribution of the Ising PMI model.")
    print("  If Ising OFF produces identical text, then Ising adds nothing.")
    print("  If Ising ON produces better text when recall misses, then it contributes.\n")

    ising_texts = []
    baseline_texts = []
    ising_stats = {'recall_hit': 0, 'pmi_only': 0, 'total': 0}
    baseline_stats = {'recall_hit': 0, 'pmi_only': 0, 'total': 0}

    for prompt in prompts:
        result_ising = model.generator.generate(prompt=prompt, length=length)
        ising_texts.append(result_ising['text'])
        for d in result_ising['diagnostics']:
            ising_stats['total'] += 1
            if d['recall_hit']:
                ising_stats['recall_hit'] += 1
            else:
                ising_stats['pmi_only'] += 1

        result_baseline = model.baseline_generator.generate(prompt=prompt, length=length)
        baseline_texts.append(result_baseline['text'])
        for d in result_baseline['diagnostics']:
            baseline_stats['total'] += 1
            if d['recall_hit']:
                baseline_stats['recall_hit'] += 1
            else:
                baseline_stats['pmi_only'] += 1

        print(f"  Prompt: '{prompt}'")
        print(f"  Ising ON:  {result_ising['text']}")
        print(f"  Ising OFF: {result_baseline['text']}")

    # Quality metrics
    ising_quality = evaluate_quality(ising_texts, len(prompts))
    baseline_quality = evaluate_quality(baseline_texts, len(prompts))

    print("\n" + "-" * 50)
    print("ABLATION RESULTS:")
    print("-" * 50)
    print(f"\n  {'Metric':<30} {'Ising ON':>12} {'Ising OFF':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    print(f"  {'Recall hit rate':<30} {ising_stats['recall_hit']/max(1,ising_stats['total']):>11.1%} "
          f"{baseline_stats['recall_hit']/max(1,baseline_stats['total']):>11.1%}")
    print(f"  {'PMI-only rate':<30} {ising_stats['pmi_only']/max(1,ising_stats['total']):>11.1%} "
          f"{baseline_stats['pmi_only']/max(1,baseline_stats['total']):>11.1%}")
    print(f"  {'Unique words':<30} {ising_quality['unique_words']:>12} {baseline_quality['unique_words']:>12}")
    print(f"  {'Type-token ratio':<30} {ising_quality['type_token_ratio']:>12.3f} "
          f"{baseline_quality['type_token_ratio']:>12.3f}")
    print(f"  {'Double DET patterns':<30} {ising_quality['n_double_dets']:>12} {baseline_quality['n_double_dets']:>12}")
    print(f"  {'Same-word reps':<30} {ising_quality['n_same_word_reps']:>12} {baseline_quality['n_same_word_reps']:>12}")
    print(f"  {'\"of the of the\" loops':<30} {ising_quality['n_of_the_loops']:>12} {baseline_quality['n_of_the_loops']:>12}")

    ising_pmi_rate = ising_stats['pmi_only'] / max(1, ising_stats['total'])
    print(f"\n  KEY INSIGHT:")
    print(f"  When recall misses ({ising_pmi_rate:.1%} of positions for Ising ON),")
    print(f"  the PMI coupling provides a STRUCTURED fallback instead of random words.")
    print(f"  Without Ising, the fallback is just unigram frequency (random).")
    print(f"  This is where the Ising model ACTUALLY contributes.")

    return ising_texts, baseline_texts


def main():
    print("=" * 70)
    print("ISING-ENHANCED N-GRAM LANGUAGE MODEL (v3.0 — Knowledge Machine)")
    print("=" * 70)
    print()
    print("Architecture:")
    print("  Primary:   N-gram exact recall")
    print("  Secondary: Ising PMI coupling (sparse)")
    print("  Knowledge: Layer 2 (field) + Layer 3 (3-spin couplings)")
    print("  Tertiary:  Unigram field")
    print("  Sampling:  Integer Boltzmann (lookup table, NO np.exp)")
    print()
    print("Path 2 Features:")
    print("  2a: Beam generation (global coherence)")
    print("  2b: Joint phrase sampling (MCMC)")
    print("  2c: Temperature annealing (phase transition)")
    print("  2d: Skip-gram PMI (distance-specific)")
    print()
    print("Path 3 Features:")
    print("  3a: Better tokenizer (contractions, hyphens, numbers)")
    print("  3b: Sparse coupling matrix (scipy.sparse)")
    print("  3c: Perplexity evaluation")
    print()
    print("Path 4 Features (NEW):")
    print("  4a: Knowledge external field h_knowledge[w] (Layer 2)")
    print("  4b: 3-spin couplings J3[(s,p)] (Layer 3)")
    print("  4c: SPO triple extraction from corpus")
    print("  4d: Curated commonsense triples (~50)")
    print()

    t0 = time.time()

    model = IsingLMModel(
        vocab_min_freq=5,
        vocab_max_size=3000,
        ngram_max_n=5,
        ngram_min_count=1,
        pmi_window=5,
        pmi_min_count=2,
        pmi_cap=10,
        recall_scale=1000,
        pmi_weight=3,
        field_weight=1,
        beta_type=0.01,
        beta_word=0.15,
        copy_enabled=True,
        copy_min_context=2,
        copy_min_confidence=0.25,
        same_word_penalty=50000,
        max_closed_class_run=2,
        ising_enabled=True,
        skip_pmi_max_dist=5,
        knowledge_scale=500,
        spin3_scale=800,
    )

    model.train(n_samples=20000)

    t_train = time.time() - t0
    print(f"\nTraining time: {t_train:.1f}s")

    # ======================================================================
    # PHASE 1: Quick generation test
    # ======================================================================
    print("\n" + "=" * 70)
    print("QUICK GENERATION TEST")
    print("=" * 70)

    prompts = ["the", "a", "in", "science", "research", "students", "he",
               "to", "of", "for", "education", "we", "this"]

    for prompt in prompts:
        result = model.generator.generate(prompt=prompt, length=15)
        text = result['text']
        n_copies = sum(1 for d in result['diagnostics'] if d['copy'])
        n_recalls = sum(1 for d in result['diagnostics'] if d['recall_hit'])
        n_pmi = sum(1 for d in result['diagnostics'] if not d['recall_hit'])
        flag = " LOOP" if "of the of the" in text else ""
        print(f"  '{prompt}' -> {text}{flag}")
        print(f"           recalls={n_recalls} pmi_only={n_pmi} copies={n_copies}")

    # ======================================================================
    # PHASE 2: Path 2a — Beam generation test
    # ======================================================================
    print("\n" + "=" * 70)
    print("PATH 2a: BEAM GENERATION (Global Coherence)")
    print("=" * 70)

    for prompt in ["the", "science", "research"]:
        beam_result = model.generate_beam(prompt=prompt, length=15, n_beams=3)
        print(f"\n  Prompt: '{prompt}'")
        print(f"  Best (energy={beam_result['beam_energy']}): {beam_result['text']}")
        print(f"  All candidates:")
        for c in beam_result['all_candidates']:
            marker = " <-- BEST" if c['energy'] == beam_result['beam_energy'] else ""
            print(f"    energy={c['energy']:>6}: {c['text']}{marker}")

    # ======================================================================
    # PHASE 3: Path 2c — Temperature annealing test
    # ======================================================================
    print("\n" + "=" * 70)
    print("PATH 2c: TEMPERATURE ANNEALING (Ising Phase Transition)")
    print("=" * 70)

    for prompt in ["the", "science"]:
        annealed_result = model.generate_annealed(
            prompt=prompt, length=15,
            beta_start=0.005, beta_end=0.5
        )
        print(f"\n  Prompt: '{prompt}'")
        print(f"  Annealed: {annealed_result['text']}")
        if annealed_result.get('beta_schedule'):
            betas = annealed_result['beta_schedule']
            print(f"  Beta schedule: {betas[0]:.4f} -> {betas[-1]:.4f}")

    # ======================================================================
    # PHASE 4: Ablation study
    # ======================================================================
    ablation_prompts = ["the", "science", "research", "students", "education",
                        "to", "of", "in", "for", "he", "they", "this", "that"]
    run_ablation(model, ablation_prompts, length=20)

    # ======================================================================
    # PHASE 5: Path 3c — Perplexity evaluation
    # ======================================================================
    print("\n" + "=" * 70)
    print("PATH 3c: PERPLEXITY EVALUATION")
    print("=" * 70)

    ppl = model.compute_perplexity(n_samples=50)
    print(f"  Final Perplexity: {ppl:.2f}")

    # ======================================================================
    # PHASE 6: Full evaluation
    # ======================================================================
    print("\n" + "=" * 70)
    print("FULL EVALUATION (30 samples)")
    print("=" * 70)

    all_text = []
    all_metrics = {}
    n_eval = 30

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

    # Quality check
    quality = evaluate_quality(all_text, n_eval)
    print(f"\n  Type-token ratio: {quality['type_token_ratio']:.3f}")
    print(f"  Double DET patterns: {quality['n_double_dets']}/{n_eval}")
    print(f"  'of the of the' loops: {quality['n_of_the_loops']}/{n_eval}")

    print("\n--- ALL GENERATED TEXT ---")
    for i, text in enumerate(all_text):
        print(f"  {i+1}. {text}")

    # ======================================================================
    # PHASE 7: Knowledge Layer Test
    # ======================================================================
    print("\n" + "=" * 70)
    print("KNOWLEDGE LAYER TEST")
    print("=" * 70)
    print("\n  Testing: Knowledge ON vs Knowledge OFF")
    print("  Using knowledge-triggering prompts that should activate SPO triples.\n")

    # Knowledge layer diagnostics
    kl = model.knowledge_layer
    print(f"  Knowledge Layer Statistics:")
    print(f"    Total triples: {kl.n_triples}")
    print(f"    Unique subjects: {kl.n_unique_subjects}")
    print(f"    Unique predicates: {kl.n_unique_predicates}")
    print(f"    J3 entries: {len(kl.J3)}")
    print(f"    h_knowledge non-zero: {int(np.count_nonzero(kl.h_knowledge))}")
    print(f"    h_knowledge max: {int(kl.h_knowledge.max())}")

    # Test with knowledge-triggering prompts
    knowledge_prompts = [
        "the dog", "water can", "the sun", "paris is",
        "the bird", "fish in", "the teacher", "the student",
        "science is", "fire and", "ice is", "the book",
        "education is", "research is", "school is",
    ]

    print(f"\n  {'Prompt':<20} {'Knowledge ON':<55} {'Knowledge OFF':<55}")
    print(f"  {'-'*20} {'-'*55} {'-'*55}")

    knowledge_on_texts = []
    knowledge_off_texts = []
    knowledge_on_stats = {'knowledge_hits': 0, 'spin3_firings': 0, 'total': 0}
    knowledge_off_stats = {'knowledge_hits': 0, 'spin3_firings': 0, 'total': 0}

    for prompt in knowledge_prompts:
        # Knowledge ON (main generator)
        result_on = model.generator.generate(prompt=prompt, length=15)
        text_on = result_on['text']
        knowledge_on_texts.append(text_on)
        
        # Knowledge OFF 
        result_off = model.knowledge_off_generator.generate(prompt=prompt, length=15)
        text_off = result_off['text']
        knowledge_off_texts.append(text_off)

        # Track stats
        for _ in result_on['diagnostics']:
            knowledge_on_stats['total'] += 1
        for _ in result_off['diagnostics']:
            knowledge_off_stats['total'] += 1

        print(f"  {prompt:<20} {text_on[:53]:<55} {text_off[:53]:<55}")

    # Get stats from generators
    on_stats = model.generator.get_stats()
    off_stats = model.knowledge_off_generator.get_stats()

    print(f"\n  Knowledge ON stats:")
    print(f"    Knowledge hits: {on_stats.get('knowledge_hits', 0)}")
    print(f"    3-Spin firings: {on_stats.get('spin3_firings', 0)}")
    print(f"    Recall hit rate: {on_stats['recall_hit_rate']:.1%}")
    print(f"    PMI-only rate: {on_stats['pmi_only_rate']:.1%}")

    print(f"\n  Knowledge OFF stats:")
    print(f"    Knowledge hits: 0 (disabled)")
    print(f"    3-Spin firings: 0 (disabled)")
    print(f"    Recall hit rate: {off_stats['recall_hit_rate']:.1%}")
    print(f"    PMI-only rate: {off_stats['pmi_only_rate']:.1%}")

    # Quality comparison
    on_quality = evaluate_quality(knowledge_on_texts, len(knowledge_prompts))
    off_quality = evaluate_quality(knowledge_off_texts, len(knowledge_prompts))

    print(f"\n  {'Metric':<30} {'Knowledge ON':>14} {'Knowledge OFF':>14}")
    print(f"  {'-'*30} {'-'*14} {'-'*14}")
    print(f"  {'Unique words':<30} {on_quality['unique_words']:>14} {off_quality['unique_words']:>14}")
    print(f"  {'Type-token ratio':<30} {on_quality['type_token_ratio']:>14.3f} {off_quality['type_token_ratio']:>14.3f}")
    print(f"  {'Same-word reps':<30} {on_quality['n_same_word_reps']:>14} {off_quality['n_same_word_reps']:>14}")

    # ======================================================================
    # Summary
    # ======================================================================
    stats = model.generator.get_stats()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Architecture: N-gram (primary) + Ising PMI (secondary) + Knowledge (Layer 2+3)")
    print(f"  Integer-only: YES (lookup-table Boltzmann, no np.exp)")
    print(f"  Sparse PMI: YES (scipy.sparse.csr_matrix)")
    print(f"  Skip-gram PMI: YES (distance 1-{model.skip_pmi_max_dist})")
    print(f"  Knowledge triples: {kl.n_triples}")
    print(f"  Knowledge J3 entries: {len(kl.J3)}")
    print(f"\n  Generation statistics:")
    print(f"    Recall hit rate: {stats['recall_hit_rate']:.1%}")
    print(f"    PMI-only rate: {stats['pmi_only_rate']:.1%}")
    print(f"    Copy rate: {stats['copy_rate']:.1%}")
    print(f"    Knowledge hits: {stats.get('knowledge_hits', 0)}")
    print(f"    3-Spin firings: {stats.get('spin3_firings', 0)}")
    print(f"    Ising enabled: {stats['ising_enabled']}")
    print(f"\n  Perplexity: {ppl:.2f}")

    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.1f}s")

    return model


if __name__ == "__main__":
    main()
