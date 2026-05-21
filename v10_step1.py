#!/usr/bin/env python3
"""v10.0 Step 1: Train + initial PPL"""
import sys, os, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from ising_spin.model import IsingLMModel

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
    graded_couplings_enabled=False, auto_calibrate_beta=True,
    recall_primary_mode=True,
    interpolated=False, kn_backoff=True,
)

print("Training on 20K...")
model.train(n_samples=20000)
print(f"Training: {time.time()-t0:.1f}s")

ppl = model.compute_perplexity(n_samples=50)
print(f"PPL: {ppl:.2f}")

# Save model for later steps
import json
info = {
    "ppl": ppl,
    "beta": model.beta_word,
    "train_time": time.time()-t0,
    "n_samples": 20000,
}
with open("/home/z/my-project/download/v10_step1_info.json", "w") as f:
    json.dump(info, f, indent=2)

# Save essential model components for reload
print(f"Total: {time.time()-t0:.1f}s")
