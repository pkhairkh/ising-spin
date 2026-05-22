#!/usr/bin/env python3
"""Minimal v9.0 test — just train + PPL, no β sweep."""

import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ising_spin.model import IsingLMModel

print("ISING SPIN v9.0 — Quick 50K Test")
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

model.train(n_samples=50000)
t_train = time.time() - t0
print(f"\nTraining: {t_train:.1f}s")

# PPL
ppl = model.compute_perplexity(n_samples=50)
print(f"PPL = {ppl:.2f} (β={model.beta_word:.6f})")

# Quick generation
for p in ["the", "science", "research"]:
    r = model.generator.generate(prompt=p, length=20)
    print(f"  '{p}' -> {r['text']}")

print(f"\nTotal: {time.time()-t0:.1f}s")
