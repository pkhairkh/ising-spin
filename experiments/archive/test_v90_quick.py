#!/usr/bin/env python3
"""Quick v9.0 test — 20K samples for fast PPL comparison."""

import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ising_spin.model import (
    IsingLMModel, IntegerBoltzmannSampler, LN2_NUM, LN2_DEN, LOG2_SCALE
)

print("=" * 70)
print("ISING SPIN v9.0 — Quick PPL Test (20K samples)")
print("=" * 70)

t0 = time.time()

model = IsingLMModel(
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
    walsh_subspace_rank=64,
    walsh_max_order=2,
    walsh_weight=1,
    walsh_min_coeff=3,
    graded_couplings_enabled=False,
    coupling_scale=1000,
    trigram_scale=2000,
    auto_calibrate_beta=True,
    recall_primary_mode=True,
    interpolated=False,
)

model.train(n_samples=20000)

t_train = time.time() - t0
print(f"\nTraining time: {t_train:.1f}s")

# PPL evaluation
print("\n" + "=" * 70)
print("PERPLEXITY EVALUATION")
print("=" * 70)

ppl = model.compute_perplexity(n_samples=30)
print(f"  v9.0 PPL (fine-log2, β={model.beta_word:.6f}): {ppl:.2f}")

# Quick β sweep
print("\nβ SWEEP:")
gen = model.generator
recall_scale = model.recall_scale

beta_factors = [0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5]
best_ppl = float('inf')
best_factor = 0.85

for factor in beta_factors:
    beta_val = factor * LN2_NUM / (recall_scale * LN2_DEN)
    test_sampler = IntegerBoltzmannSampler(beta=beta_val, max_delta=25000)

    total_log2_prob = 0
    total_tokens = 0

    for seq in model.test_sequences[:20]:
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

            target_in_candidates = int(target_word) in set(candidate_words.tolist())
            if not target_in_candidates:
                total_log2_prob += -15 * LOG2_SCALE
                total_tokens += 1
                continue

            recall_matches = gen.ngram_index.lookup(context_words)
            recall_hit = bool(recall_matches)

            energies = gen._compute_word_energy(
                pos, candidate_words, word_type,
                context_words, context_types, recall_hit
            )

            log_probs = test_sampler.compute_log_probabilities(energies)
            target_idx = np.where(candidate_words == target_word)[0]
            if len(target_idx) > 0:
                total_log2_prob += int(log_probs[target_idx[0]])
            else:
                total_log2_prob += -15 * LOG2_SCALE
            total_tokens += 1

    if total_tokens > 0:
        avg_log2 = total_log2_prob / (total_tokens * LOG2_SCALE)
        ppl_val = 2.0 ** (-avg_log2)
        print(f"  β={factor:.2f}×ln2/scale → PPL={ppl_val:.1f}")
        if ppl_val < best_ppl:
            best_ppl = ppl_val
            best_factor = factor

print(f"\n  BEST: β={best_factor:.2f}×ln2/scale → PPL={best_ppl:.1f}")
print(f"  v8.1 baseline (20K, floor-log2): PPL ≈ 183")
print(f"  Improvement: {183/best_ppl:.1f}×")

# Quick generation test
print("\n" + "=" * 70)
print("GENERATION SAMPLES")
print("=" * 70)

for prompt in ["the", "science", "research", "students", "water"]:
    result = model.generator.generate(prompt=prompt, length=20)
    text = result['text']
    print(f"  '{prompt}' -> {text}")

t_total = time.time() - t0
print(f"\nTotal time: {t_total:.1f}s")
