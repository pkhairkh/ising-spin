#!/usr/bin/env python3
"""Quick test: v8.2 with Topic Spin, single training run."""

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
    ising_enabled=False, skip_pmi_max_dist=5,
    mcmc_refine_steps=0, use_conceptnet=False,
    walsh_enabled=False, graded_couplings_enabled=False,
    auto_calibrate_beta=True, recall_primary_mode=True,
    # v8.2: Topic Spin ENABLED
    topic_spin_enabled=True, topic_n_topics=16,
    topic_coherence_penalty=40, topic_spin_flip_interval=20,
    topic_context_window=30, topic_coupling_scale=100,
)
model.train(n_samples=50000)

# PPL
ppl = model.compute_perplexity(n_samples=50)
print(f"\n  PPL (with Topic Spin): {ppl:.2f}")

# Generate 400 tokens with topic spin
prompts = ["the history of", "science and technology", "education is the"]
for prompt in prompts:
    result = model.generator.generate(prompt=prompt, length=400)
    words = result['text'].split()
    print(f"\n--- Prompt: '{prompt}' ({len(words)} words) ---")
    print(result['text'][:600])
    print("...")

# Topic diagnostics
if model.topic_spin_layer is not None:
    td = model.topic_spin_layer.get_diagnostics()
    print(f"\n  Topic Spin: flips={td['spin_flips']}, "
          f"penalties={td['coherence_penalties']}, "
          f"rate={td['penalty_rate']:.1%}")

# Save
with open("/home/z/my-project/download/ising_v82_topic_spin.txt", 'w') as f:
    f.write(f"v8.2 Topic Spin — PPL={ppl:.2f}\n\n")
    for prompt in prompts:
        r = model.generator.generate(prompt=prompt, length=400)
        f.write(f"--- {prompt} ---\n{r['text']}\n\n")

print(f"\nTotal: {time.time()-t0:.1f}s")
