#!/usr/bin/env python3
"""
Ising Knowledge Machine — Runner v3.0.

5-Layer Architecture:
  Layer 1: PMI Couplings (word affinities) + Local Field (unigram)
  Layer 2: Knowledge External Field h_knowledge[w]
  Layer 3: 3-Spin Couplings J3[(s,p)] for SPO triples
  Layer 4: Category Couplings J_category (hypernym-based semantic smoothing)
  Layer 5: Markov Logic Penalty (factual consistency rules)

Primary:   N-gram exact recall (when context matches)
Secondary: Ising PMI coupling (when recall misses)
Knowledge: Layers 2-5 (knowledge graph + category + logic)
Sampling:  Integer Boltzmann via lookup table (NO np.exp)

Improvements over v2.0:
  - Vocabulary scaled to V=8000 (was 3000)
  - Triple coverage expanded to 500+ hardcoded + ConceptNet
  - knowledge_scale=2000, spin3_scale=3000 (much stronger)
  - Layer 4: Category couplings (30+ categories)
  - Layer 5: Markov logic rules (30+ rules)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ising_spin.model import (
    IsingLMModel, IsingLM, KnowledgeLayer, CategoryLayer, MarkovLogicLayer,
    POS2IDX, IDX2POS
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

    return ising_texts, baseline_texts


def main():
    print("=" * 70)
    print("ISING KNOWLEDGE MACHINE (v3.0 — 5-Layer Architecture)")
    print("=" * 70)
    print()
    print("5-Layer Architecture:")
    print("  Layer 1: PMI Couplings (word affinities) + Local Field")
    print("  Layer 2: Knowledge External Field h_knowledge[w]")
    print("  Layer 3: 3-Spin Couplings J3[(s,p)] for SPO triples")
    print("  Layer 4: Category Couplings J_category (hypernym-based)")
    print("  Layer 5: Markov Logic Penalty (factual consistency)")
    print()
    print("Scaling Improvements:")
    print("  Vocabulary: V=8000 (was 3000)")
    print("  Triple coverage: 500+ hardcoded + ConceptNet")
    print("  knowledge_scale=2000, spin3_scale=3000 (much stronger)")
    print("  30+ category groups, 30+ logic rules")
    print()

    t0 = time.time()

    model = IsingLMModel(
        # Vocabulary — SCALED UP from 3000 to 8000
        vocab_min_freq=3,            # Was 5, lowered to include more words
        vocab_max_size=8000,         # Was 3000, now 8000

        # N-gram and PMI settings
        ngram_max_n=5,
        ngram_min_count=1,
        pmi_window=5,
        pmi_min_count=2,
        pmi_cap=10,

        # Energy scales — STRENGTHENED knowledge influence
        recall_scale=1000,           # Primary signal (unchanged)
        pmi_weight=3,
        field_weight=1,
        knowledge_scale=2000,        # Was 500, now 2000 (4x stronger)
        spin3_scale=3000,            # Was 800, now 3000 (3.75x stronger)
        category_scale=400,          # NEW: Layer 4
        logic_rule_scale=600,        # NEW: Layer 5 soft
        logic_hard_scale=50000,      # NEW: Layer 5 hard

        # Sampling parameters
        beta_type=0.01,
        beta_word=0.15,
        copy_enabled=True,
        copy_min_context=2,
        copy_min_confidence=0.25,
        same_word_penalty=50000,
        max_closed_class_run=2,
        ising_enabled=True,
        skip_pmi_max_dist=5,

        # ConceptNet
        use_conceptnet=True,
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
               "to", "of", "for", "education", "we", "this", "dog",
               "water", "fire", "school", "city", "animal", "food"]

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
    # PHASE 2: 5-Layer Knowledge Test
    # ======================================================================
    print("\n" + "=" * 70)
    print("5-LAYER KNOWLEDGE TEST")
    print("=" * 70)
    print("\n  Testing: All 5 Layers ON vs All Knowledge Layers OFF")
    print("  Using knowledge-triggering prompts.\n")

    # Knowledge layer diagnostics
    kl = model.knowledge_layer
    cl = model.category_layer
    ml = model.markov_logic_layer

    print(f"  Layer 2+3 (Knowledge):")
    print(f"    Total triples: {kl.n_triples}")
    print(f"    Unique subjects: {kl.n_unique_subjects}")
    print(f"    Unique predicates: {kl.n_unique_predicates}")
    print(f"    J3 entries: {len(kl.J3)}")
    print(f"    h_knowledge non-zero: {int(np.count_nonzero(kl.h_knowledge))}")
    print(f"    h_knowledge max: {int(kl.h_knowledge.max())}")
    
    print(f"\n  Layer 4 (Category):")
    print(f"    Categories: {cl.n_categories}")
    print(f"    Categorized words: {cl.n_categorized_words}")
    print(f"    Peer pairs: {sum(len(p) for p in cl.word_peers.values()) // 2}")
    
    print(f"\n  Layer 5 (Markov Logic):")
    print(f"    Total rules: {ml.n_rules}")
    print(f"    Soft rules: {ml.n_soft_rules}")
    print(f"    Hard rules: {ml.n_hard_rules}")

    # Test with knowledge-triggering prompts
    knowledge_prompts = [
        "the dog", "water can", "the sun", "paris is",
        "the bird", "fish in", "the teacher", "the student",
        "science is", "fire and", "ice is", "the book",
        "education is", "research is", "school is",
        "the doctor", "the cat", "the horse", "mountain is",
        "ocean is", "the forest", "the library", "the kitchen",
    ]

    print(f"\n  {'Prompt':<20} {'5-Layers ON':<55} {'Knowledge OFF':<55}")
    print(f"  {'-'*20} {'-'*55} {'-'*55}")

    knowledge_on_texts = []
    knowledge_off_texts = []

    for prompt in knowledge_prompts:
        # All 5 Layers ON (main generator)
        result_on = model.generator.generate(prompt=prompt, length=15)
        text_on = result_on['text']
        knowledge_on_texts.append(text_on)

        # Knowledge Layers OFF
        result_off = model.knowledge_off_generator.generate(prompt=prompt, length=15)
        text_off = result_off['text']
        knowledge_off_texts.append(text_off)

        print(f"  {prompt:<20} {text_on[:53]:<55} {text_off[:53]:<55}")

    # Get stats from generators
    on_stats = model.generator.get_stats()
    off_stats = model.knowledge_off_generator.get_stats()

    print(f"\n  5-Layers ON stats:")
    print(f"    Recall hit rate: {on_stats['recall_hit_rate']:.1%}")
    print(f"    PMI-only rate: {on_stats['pmi_only_rate']:.1%}")
    print(f"    Knowledge hits: {on_stats.get('knowledge_hits', 0)}")
    print(f"    3-Spin firings: {on_stats.get('spin3_firings', 0)}")
    print(f"    Category hits: {on_stats.get('category_hits', 0)}")
    print(f"    Logic hits: {on_stats.get('logic_hits', 0)}")

    print(f"\n  Knowledge OFF stats:")
    print(f"    Recall hit rate: {off_stats['recall_hit_rate']:.1%}")
    print(f"    PMI-only rate: {off_stats['pmi_only_rate']:.1%}")

    # Quality comparison
    on_quality = evaluate_quality(knowledge_on_texts, len(knowledge_prompts))
    off_quality = evaluate_quality(knowledge_off_texts, len(knowledge_prompts))

    print(f"\n  {'Metric':<30} {'5-Layers ON':>14} {'Knowledge OFF':>14}")
    print(f"  {'-'*30} {'-'*14} {'-'*14}")
    print(f"  {'Unique words':<30} {on_quality['unique_words']:>14} {off_quality['unique_words']:>14}")
    print(f"  {'Type-token ratio':<30} {on_quality['type_token_ratio']:>14.3f} {off_quality['type_token_ratio']:>14.3f}")
    print(f"  {'Same-word reps':<30} {on_quality['n_same_word_reps']:>14} {off_quality['n_same_word_reps']:>14}")

    # ======================================================================
    # PHASE 3: Beam generation test
    # ======================================================================
    print("\n" + "=" * 70)
    print("BEAM GENERATION (Global Coherence)")
    print("=" * 70)

    for prompt in ["the", "science", "the dog"]:
        beam_result = model.generate_beam(prompt=prompt, length=15, n_beams=3)
        print(f"\n  Prompt: '{prompt}'")
        print(f"  Best (energy={beam_result['beam_energy']}): {beam_result['text']}")
        print(f"  All candidates:")
        for c in beam_result['all_candidates']:
            marker = " <-- BEST" if c['energy'] == beam_result['beam_energy'] else ""
            print(f"    energy={c['energy']:>6}: {c['text']}{marker}")

    # ======================================================================
    # PHASE 4: Ablation study
    # ======================================================================
    ablation_prompts = ["the", "science", "research", "students", "education",
                        "to", "of", "in", "for", "he", "they", "this", "that",
                        "dog", "water", "fire", "school", "city"]
    run_ablation(model, ablation_prompts, length=20)

    # ======================================================================
    # PHASE 5: Perplexity evaluation
    # ======================================================================
    print("\n" + "=" * 70)
    print("PERPLEXITY EVALUATION")
    print("=" * 70)

    ppl = model.compute_perplexity(n_samples=50)
    print(f"  Final Perplexity: {ppl:.2f}")

    # ======================================================================
    # Summary
    # ======================================================================
    stats = model.generator.get_stats()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Architecture: 5-Layer Ising Knowledge Machine")
    print(f"  Layer 1: PMI couplings + local field")
    print(f"  Layer 2: Knowledge external field (h_knowledge)")
    print(f"  Layer 3: 3-Spin couplings (J3 SPO triples)")
    print(f"  Layer 4: Category couplings (hypernym-based)")
    print(f"  Layer 5: Markov logic penalties (factual consistency)")
    print(f"\n  Integer-only: YES (lookup-table Boltzmann, no np.exp)")
    print(f"  Sparse PMI: YES (scipy.sparse.csr_matrix)")
    print(f"  Skip-gram PMI: YES (distance 1-{model.skip_pmi_max_dist})")
    print(f"\n  Scale comparison:")
    print(f"    recall_scale=     {model.recall_scale:>6}")
    print(f"    knowledge_scale=  {model.knowledge_scale:>6}")
    print(f"    spin3_scale=      {model.spin3_scale:>6}")
    print(f"    category_scale=   {model.category_scale:>6}")
    print(f"    logic_rule_scale= {model.logic_rule_scale:>6}")
    print(f"\n  Knowledge Layer: {kl.n_triples} triples, {len(kl.J3)} J3 entries")
    print(f"  Category Layer: {cl.n_categories} categories, {cl.n_categorized_words} words")
    print(f"  Logic Layer: {ml.n_rules} rules ({ml.n_soft_rules} soft, {ml.n_hard_rules} hard)")
    print(f"\n  Generation statistics:")
    print(f"    Recall hit rate: {stats['recall_hit_rate']:.1%}")
    print(f"    PMI-only rate: {stats['pmi_only_rate']:.1%}")
    print(f"    Copy rate: {stats['copy_rate']:.1%}")
    print(f"    Knowledge hits: {stats.get('knowledge_hits', 0)}")
    print(f"    3-Spin firings: {stats.get('spin3_firings', 0)}")
    print(f"    Category hits: {stats.get('category_hits', 0)}")
    print(f"    Logic hits: {stats.get('logic_hits', 0)}")
    print(f"    Ising enabled: {stats['ising_enabled']}")
    print(f"\n  Perplexity: {ppl:.2f}")

    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.1f}s")

    return model


if __name__ == "__main__":
    main()
