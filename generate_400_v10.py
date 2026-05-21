#!/usr/bin/env python3
"""Generate 400 tokens with best PPL config: precise ratio, f=0.9, 50K samples."""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from ising_spin.model import (
    IsingLMModel, IntegerBoltzmannSampler, LN2_NUM, LN2_DEN, LOG2_SCALE
)

t0 = time.time()

# Use 3K vocab (better PPL due to smaller candidate set per type)
model = IsingLMModel(
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
    graded_couplings_enabled=False, auto_calibrate_beta=False,
    recall_primary_mode=True,
    interpolated=False, kn_backoff=False,
)

print("Training 50K, 3K vocab...")
model.train(n_samples=50000)

# Set β = 0.9*ln(2)/recall_scale (optimal from β sweep)
BETA_FACTOR = 0.9
recall_scale = model.recall_scale
beta_val = BETA_FACTOR * LN2_NUM / (recall_scale * LN2_DEN)
model.generator.word_sampler = IntegerBoltzmannSampler(beta=beta_val, max_delta=25000)
model.generator.beta_word = beta_val
model.beta_word = beta_val
print(f"β = {BETA_FACTOR}×ln(2)/scale = {beta_val:.6f}")

# PPL
ppl = model.compute_perplexity(n_samples=30)
print(f"PPL: {ppl:.1f}")

# Generate 400 tokens
print("\n" + "=" * 70)
print("GENERATING 400 TOKENS")
print("=" * 70)

for prompt in ["the history of", "science and technology", "research shows that"]:
    result = model.generator.generate(prompt=prompt, length=400)
    text = result['text']
    words = text.split()
    n_recalls = sum(1 for d in result['diagnostics'] if d['recall_hit'])
    n_copies = sum(1 for d in result['diagnostics'] if d['copy'])
    print(f"\n--- '{prompt}' ({len(words)} words) ---")
    print(text)
    print(f"  recalls={n_recalls} copies={n_copies}")

    if prompt == "the history of":
        output_path = "/home/z/my-project/download/ising_v10_400tokens.txt"
        with open(output_path, 'w') as f:
            f.write(f"Ising Spin Glass Language Model v10.0 — 400-Token Generation\n")
            f.write(f"Prompt: '{prompt}'\n")
            f.write(f"PPL: {ppl:.1f}\n")
            f.write(f"Training: 50K FineWeb-Edu samples, 3K vocab\n")
            f.write(f"Architecture: Recall-primary + Fine-grained log2\n")
            f.write(f"recall_scale=800, β = {BETA_FACTOR}×ln(2)/scale\n")
            f.write("=" * 70 + "\n\n")
            f.write(text)
            f.write(f"\n\n--- Diagnostics ---\n")
            f.write(f"Words: {len(words)}\n")
            f.write(f"Recall hits: {n_recalls}\n")
            f.write(f"Copy hits: {n_copies}\n")
        print(f"  Saved to: {output_path}")

print(f"\nTotal: {time.time()-t0:.1f}s")
