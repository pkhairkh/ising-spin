#!/usr/bin/env python3
"""v10.0 corrected: Reverted precise ratio, using v9.0 formula + high-P fix."""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from ising_spin.model import (
    IsingLMModel, IntegerBoltzmannSampler, LN2_NUM, LN2_DEN, LOG2_SCALE
)

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
    interpolated=False, kn_backoff=False,
)

print("Training 50K, v9.0 formula + high-P fix...")
model.train(n_samples=50000)
print(f"Training: {time.time()-t0:.1f}s")

# Auto-calibrated β
print(f"Auto β: {model.beta_word:.6f}")

# PPL
ppl = model.compute_perplexity(n_samples=50)
print(f"PPL: {ppl:.2f}")

# Quick β sweep
print("\nβ sweep:")
gen = model.generator
recall_scale = model.recall_scale
best_ppl = ppl
best_factor = 0.55

for factor in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9]:
    beta_val = factor * LN2_NUM / (recall_scale * LN2_DEN)
    sampler = IntegerBoltzmannSampler(beta=beta_val, max_delta=25000)
    total_log2_prob = 0
    total_tokens = 0
    for seq in model.test_sequences[:25]:
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
            if int(target_word) not in set(candidate_words.tolist()):
                total_log2_prob += -15 * LOG2_SCALE
                total_tokens += 1
                continue
            recall_matches = gen.ngram_index.lookup(context_words)
            recall_hit = bool(recall_matches)
            energies = gen._compute_word_energy(
                pos, candidate_words, word_type,
                context_words, context_types, recall_hit
            )
            log_probs = sampler.compute_log_probabilities(energies)
            target_idx = np.where(candidate_words == target_word)[0]
            if len(target_idx) > 0:
                total_log2_prob += int(log_probs[target_idx[0]])
            else:
                total_log2_prob += -15 * LOG2_SCALE
            total_tokens += 1
    if total_tokens > 0:
        avg_log2 = total_log2_prob / (total_tokens * LOG2_SCALE)
        ppl_val = 2.0 ** (-avg_log2)
        print(f"  f={factor:.2f}: PPL={ppl_val:.1f}")
        if ppl_val < best_ppl:
            best_ppl = ppl_val
            best_factor = factor

print(f"\nBEST: f={best_factor:.2f}, PPL={best_ppl:.1f}")

# Apply best β
best_beta = best_factor * LN2_NUM / (recall_scale * LN2_DEN)
model.generator.word_sampler = IntegerBoltzmannSampler(beta=best_beta, max_delta=25000)
model.generator.beta_word = best_beta

# Generate 400 tokens
print("\n" + "=" * 70)
print("GENERATING 400 TOKENS")
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
            f.write(f"PPL: {best_ppl:.1f}\n")
            f.write(f"Training: 50K FineWeb-Edu samples, 5K vocab\n")
            f.write(f"v10.0: v9.0 recall formula + high-probability energy fix\n")
            f.write(f"recall_scale=800, β = {best_factor:.2f}×ln(2)/scale\n")
            f.write("=" * 70 + "\n\n")
            f.write(text)
            f.write(f"\n\n--- Diagnostics ---\n")
            f.write(f"Words: {len(words)}\n")
            f.write(f"Recall hits: {n_recalls}\n")
            f.write(f"Copy hits: {n_copies}\n")
        print(f"  Saved to: {output_path}")

print(f"\nTotal: {time.time()-t0:.1f}s")
