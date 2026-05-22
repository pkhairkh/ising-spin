#!/usr/bin/env python3
"""Quick timing test of v10.0 — find the bottleneck."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from ising_spin.model import IsingLMModel

# Test 1: WITHOUT interpolated (baseline speed)
print("=" * 50)
print("TEST 1: 10K, NO interpolated, NO KN")
print("=" * 50)
t0 = time.time()
m1 = IsingLMModel(
    vocab_min_freq=15, vocab_max_size=3000,
    ngram_max_n=4, ngram_min_count=2,
    recall_scale=800, pmi_weight=0, field_weight=1,
    knowledge_scale=0, spin3_scale=0, category_scale=0,
    logic_rule_scale=0, logic_hard_scale=0,
    beta_type=0.001, beta_word=0.001,
    copy_enabled=True, copy_min_context=2, copy_min_confidence=0.25,
    same_word_penalty=500, max_closed_class_run=2,
    ising_enabled=False, skip_pmi_max_dist=5, mcmc_refine_steps=0,
    use_conceptnet=False, walsh_enabled=False,
    graded_couplings_enabled=False, auto_calibrate_beta=True,
    recall_primary_mode=True,
    interpolated=False, kn_backoff=False,
)
m1.train(n_samples=10000)
t1 = time.time() - t0
ppl1 = m1.compute_perplexity(n_samples=20)
print(f"Time: {t1:.1f}s, PPL: {ppl1:.1f}")

# Test 2: WITH interpolated (check speed)
print("\n" + "=" * 50)
print("TEST 2: 10K, WITH interpolated, WITH KN")
print("=" * 50)
t0 = time.time()
m2 = IsingLMModel(
    vocab_min_freq=15, vocab_max_size=3000,
    ngram_max_n=4, ngram_min_count=2,
    recall_scale=800, pmi_weight=0, field_weight=1,
    knowledge_scale=0, spin3_scale=0, category_scale=0,
    logic_rule_scale=0, logic_hard_scale=0,
    beta_type=0.001, beta_word=0.001,
    copy_enabled=True, copy_min_context=2, copy_min_confidence=0.25,
    same_word_penalty=500, max_closed_class_run=2,
    ising_enabled=False, skip_pmi_max_dist=5, mcmc_refine_steps=0,
    use_conceptnet=False, walsh_enabled=False,
    graded_couplings_enabled=False, auto_calibrate_beta=True,
    recall_primary_mode=True,
    interpolated=True, kn_backoff=True,
)
m2.train(n_samples=10000)
t2 = time.time() - t0
ppl2 = m2.compute_perplexity(n_samples=20)
print(f"Time: {t2:.1f}s, PPL: {ppl2:.1f}")

print(f"\nSpeed ratio: {t2/t1:.1f}×")
print(f"PPL improvement: {ppl1/ppl2:.2f}×")
