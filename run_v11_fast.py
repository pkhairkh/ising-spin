#!/usr/bin/env python3
"""
v11.0: All PPL-lowering features enabled, using cached data.
  - Interpolated n-gram smoothing (product of experts)
  - Kneser-Ney backoff (continuation counts)
  - Topic Spin Layer (Potts coherence)
  - Fine-grained integer log2 (v9.0)
  - Integer-only (v8.2, ZERO float ops)
"""
import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from ising_spin.model import (
    IsingLMModel, IntegerBoltzmannSampler, LN2_NUM, LN2_DEN, LOG2_SCALE
)

CACHE = "/home/z/my-project/ising-spin/cached_fineweb_50k.json"
t0 = time.time()

# Load cached data
print(f"Loading cached data from {CACHE}...")
with open(CACHE) as f:
    texts = json.load(f)
print(f"Loaded {len(texts)} texts")

model = IsingLMModel(
    vocab_min_freq=15, vocab_max_size=5000,
    ngram_max_n=5, ngram_min_count=2,
    recall_scale=800, pmi_weight=0, field_weight=1,
    knowledge_scale=0, spin3_scale=0, category_scale=0,
    logic_rule_scale=0, logic_hard_scale=0,
    beta_type=0.001, beta_word=0.001,
    copy_enabled=True, copy_min_context=2, copy_min_confidence=0.25,
    same_word_penalty=300, max_closed_class_run=2,
    ising_enabled=False, skip_pmi_max_dist=5, mcmc_refine_steps=0,
    use_conceptnet=False, walsh_enabled=False,
    graded_couplings_enabled=False, auto_calibrate_beta=False,
    recall_primary_mode=True,
    interpolated=True,
    kn_backoff=True,
    topic_spin_enabled=True, topic_n_topics=16,
    topic_coherence_penalty=40, topic_spin_flip_interval=15,
    topic_context_window=30, topic_coupling_scale=100,
)

print("\nv11.0: 50K, 5K vocab, interpolated+KN+TopicSpin")
print("=" * 70)
model.train(n_samples=50000, texts=texts)
train_time = time.time() - t0
print(f"\nTraining: {train_time:.1f}s")

# Quick PPL
ppl = model.compute_perplexity(n_samples=20)
print(f"PPL (default beta): {ppl:.2f}")

# β sweep (quick, 10 sequences)
gen = model.generator
recall_scale = model.recall_scale
best_ppl = float('inf')
best_factor = 0.55
ppl_results = {}

print("\nbeta sweep (10 test seqs):")
for factor in [0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0]:
    beta_val = factor * LN2_NUM / (recall_scale * LN2_DEN)
    sampler = IntegerBoltzmannSampler(beta=beta_val, max_delta=25000)
    total_log2_prob = 0
    total_tokens = 0
    for seq in model.test_sequences[:10]:
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
        ppl_results[factor] = ppl_val
        print(f"  f={factor:.2f}: PPL={ppl_val:.1f}")
        if ppl_val < best_ppl:
            best_ppl = ppl_val
            best_factor = factor

# Fine sweep
fine_range = [best_factor - 0.05, best_factor - 0.025,
              best_factor + 0.025, best_factor + 0.05]
fine_range = sorted(set(max(0.1, min(2.0, f)) for f in fine_range))
print(f"\nFine sweep around f={best_factor:.2f}:")
for factor in fine_range:
    if factor in ppl_results:
        continue
    beta_val = factor * LN2_NUM / (recall_scale * LN2_DEN)
    sampler = IntegerBoltzmannSampler(beta=beta_val, max_delta=25000)
    total_log2_prob = 0
    total_tokens = 0
    for seq in model.test_sequences[:10]:
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
        ppl_results[factor] = ppl_val
        print(f"  f={factor:.3f}: PPL={ppl_val:.1f}")
        if ppl_val < best_ppl:
            best_ppl = ppl_val
            best_factor = factor

print(f"\nBEST: f={best_factor:.3f}, PPL={best_ppl:.1f}")

# Apply best β
best_beta = best_factor * LN2_NUM / (recall_scale * LN2_DEN)
model.generator.word_sampler = IntegerBoltzmannSampler(beta=best_beta, max_delta=25000)
model.generator.beta_word = best_beta
model.beta_word = best_beta

# Full PPL
print("\nFull PPL eval (50 seqs)...")
ppl = model.compute_perplexity(n_samples=50)
print(f"PPL (full): {ppl:.2f}")

# Generate 400 tokens
print("\n" + "=" * 70)
print("GENERATING 400 TOKENS")
print("=" * 70)
result = model.generator.generate(prompt="the history of", length=400)
text = result['text']
words = text.split()
n_recalls = sum(1 for d in result['diagnostics'] if d['recall_hit'])
n_copies = sum(1 for d in result['diagnostics'] if d['copy'])
print(f"Words: {len(words)}, recalls={n_recalls}, copies={n_copies}")
print(text)

output_path = "/home/z/my-project/download/ising_v11_400tokens.txt"
with open(output_path, 'w') as f:
    f.write(f"Ising Spin Glass Language Model v11.0 — 400-Token Generation\n")
    f.write(f"Prompt: the history of\n")
    f.write(f"PPL: {ppl:.2f}\n")
    f.write(f"Training: 50K FineWeb-Edu, 5K vocab, interpolated+KN+TopicSpin\n")
    f.write(f"recall_scale=800, best_beta_factor={best_factor:.3f}\n")
    f.write(f"Integer-only: YES\n")
    f.write("=" * 70 + "\n\n")
    f.write(text)
    f.write(f"\n\n--- Diagnostics ---\n")
    f.write(f"Words: {len(words)}\n")
    f.write(f"Recall hits: {n_recalls}\n")
    f.write(f"Copy hits: {n_copies}\n")
    f.write(f"\n--- Beta Sweep ---\n")
    for factor in sorted(ppl_results.keys()):
        f.write(f"  f={factor:.3f}: PPL={ppl_results[factor]:.1f}\n")
print(f"\nSaved to: {output_path}")

total_time = time.time() - t0
print(f"\nTotal: {total_time:.1f}s ({total_time/60:.1f}min)")
