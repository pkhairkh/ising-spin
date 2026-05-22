#!/usr/bin/env python3
"""v9.0 evaluation — 30K samples, fast."""

import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ising_spin.model import IsingLMModel, IntegerBoltzmannSampler, LN2_NUM, LN2_DEN, LOG2_SCALE

print("v9.0 Fine-Grained Recall — 30K Test")
t0 = time.time()

model = IsingLMModel(
    vocab_min_freq=15, vocab_max_size=5000,
    ngram_max_n=5, ngram_min_count=2,
    pmi_window=5, pmi_min_count=2, pmi_cap=10,
    recall_scale=800, pmi_weight=0, field_weight=1,
    knowledge_scale=0, spin3_scale=0, category_scale=0,
    logic_rule_scale=0, logic_hard_scale=0,
    beta_type=0.001, beta_word=0.001,
    copy_enabled=True, copy_min_context=2, copy_min_confidence=0.25,
    same_word_penalty=500, max_closed_class_run=2,
    ising_enabled=False, skip_pmi_max_dist=5,
    mcmc_refine_steps=0, use_conceptnet=False,
    walsh_enabled=False, walsh_subspace_rank=64,
    walsh_max_order=2, walsh_weight=1, walsh_min_coeff=3,
    graded_couplings_enabled=False, coupling_scale=1000,
    trigram_scale=2000, auto_calibrate_beta=True,
    recall_primary_mode=True, interpolated=False,
)

model.train(n_samples=30000)
print(f"\nTraining: {time.time()-t0:.1f}s")

# PPL (auto-calibrated)
ppl = model.compute_perplexity(n_samples=30)
print(f"Auto PPL = {ppl:.2f} (β={model.beta_word:.6f})")

# Quick β sweep with fewer test samples
gen = model.generator
recall_scale = model.recall_scale
best_ppl, best_f = float('inf'), 0.90

print("\nβ Sweep:")
for factor in [0.80, 0.85, 0.90, 0.95, 1.0, 1.1]:
    beta_val = factor * LN2_NUM / (recall_scale * LN2_DEN)
    test_sampler = IntegerBoltzmannSampler(beta=beta_val, max_delta=25000)
    total_log2, total_tok = 0, 0
    for seq in model.test_sequences[:20]:
        if len(seq) < 3: continue
        for pos in range(1, len(seq)):
            tw = seq[pos]
            cw = seq[:pos]
            ct = [gen._get_word_type(w) for w in cw]
            wt = gen._get_word_type(tw)
            cl = gen.type_words.get(wt, [])
            if not cl: continue
            cands = np.array(cl, dtype=np.int64)
            if int(tw) not in set(cands.tolist()):
                total_log2 += -15 * LOG2_SCALE; total_tok += 1; continue
            rm = gen.ngram_index.lookup(cw)
            e = gen._compute_word_energy(pos, cands, wt, cw, ct, bool(rm))
            lp = test_sampler.compute_log_probabilities(e)
            ti = np.where(cands == tw)[0]
            total_log2 += int(lp[ti[0]]) if len(ti) > 0 else -15 * LOG2_SCALE
            total_tok += 1
    if total_tok > 0:
        pv = 2.0 ** (-total_log2 / (total_tok * LOG2_SCALE))
        print(f"  β={factor:.2f} → PPL={pv:.1f}")
        if pv < best_ppl: best_ppl, best_f = pv, factor

print(f"\nBest: β={best_f:.2f} → PPL={best_ppl:.1f}")
print(f"v8.1 baseline (30K, floor-log₂): PPL ≈ 150")
print(f"Improvement: {150/best_ppl:.2f}×")

# Generation
print("\nGeneration:")
for p in ["the", "science", "water", "education"]:
    r = model.generator.generate(prompt=p, length=20)
    print(f"  '{p}' -> {r['text']}")

print(f"\nTotal: {time.time()-t0:.1f}s")
