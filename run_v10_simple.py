#!/usr/bin/env python3
"""v10.0 final: Train + PPL + generate, no β sweep."""
import sys, os, time
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
    interpolated=False, kn_backoff=False,
)

print("Training 50K...")
model.train(n_samples=50000)
print(f"Training: {time.time()-t0:.1f}s")

ppl = model.compute_perplexity(n_samples=50)
print(f"\nPPL (auto β={model.beta_word:.6f}): {ppl:.2f}")

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
            f.write(f"PPL: {ppl:.1f}\n")
            f.write(f"Training: 50K FineWeb-Edu samples, 5K vocab\n")
            f.write(f"v10.0: v9.0 recall formula + high-probability energy fix\n")
            f.write(f"recall_scale=800, β auto-calibrated\n")
            f.write("=" * 70 + "\n\n")
            f.write(text)
            f.write(f"\n\n--- Diagnostics ---\n")
            f.write(f"Words: {len(words)}\n")
            f.write(f"Recall hits: {n_recalls}\n")
            f.write(f"Copy hits: {n_copies}\n")
        print(f"  Saved to: {output_path}")

print(f"\nTotal: {time.time()-t0:.1f}s")
