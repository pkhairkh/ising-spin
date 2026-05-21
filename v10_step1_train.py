#!/usr/bin/env python3
"""v10.0 Step 1: Train 50K + quick PPL"""
import sys, os, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from ising_spin.model import IsingLMModel, IntegerBoltzmannSampler, LN2_NUM, LN2_DEN
import numpy as np

t0 = time.time()
model = IsingLMModel(
    vocab_min_freq=15, vocab_max_size=5000,
    ngram_max_n=5, ngram_min_count=2,
    recall_scale=800, pmi_weight=0, field_weight=1,
    knowledge_scale=0, spin3_scale=0, category_scale=0,
    logic_rule_scale=0, logic_hard_scale=0,
    beta_type=0.001, beta_word=0.001,
    copy_enabled=True, copy_min_context=2, copy_min_confidence=0.25,
    same_word_penalty=500, max_closed_class_run=2,
    ising_enabled=False, skip_pmi_max_dist=5, mcmc_refine_steps=0,
    use_conceptnet=False, walsh_enabled=False,
    graded_couplings_enabled=False, auto_calibrate_beta=False,
    recall_primary_mode=True,
    interpolated=False, kn_backoff=False,
)
model.train(n_samples=50000)

# Set β = 0.9*ln(2)/recall_scale
BETA_FACTOR = 0.9
recall_scale = model.recall_scale
beta_val = BETA_FACTOR * LN2_NUM / (recall_scale * LN2_DEN)
model.generator.word_sampler = IntegerBoltzmannSampler(beta=beta_val, max_delta=25000)
model.generator.beta_word = beta_val
model.beta_word = beta_val

# Quick PPL with just 30 samples
ppl = model.compute_perplexity(n_samples=30)
print(f"\nInitial PPL (f={BETA_FACTOR}): {ppl:.2f}")

# Save model to disk
with open("/home/z/my-project/download/v10_model.pkl", "wb") as f:
    pickle.dump(model, f)
print(f"Model saved. Total: {time.time()-t0:.1f}s")
