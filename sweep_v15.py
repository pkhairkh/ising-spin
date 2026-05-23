#!/usr/bin/env python3
"""Sweep accumulator weights to find optimal configuration."""
import os
os.environ["PYTHONUNBUFFERED"] = "1"
import json
import sys
sys.path.insert(0, 'src')

from ising_spin.model import IsingLMModel

with open("cached_fineweb_50k.json") as f:
    texts = json.load(f)[:50000]

for acc_weight in [0, 200, 400]:
    print(f"\n{'='*60}", flush=True)
    print(f"ACC_WEIGHT = {acc_weight}", flush=True)
    print(f"{'='*60}", flush=True)
    
    model = IsingLMModel(
        vocab_min_freq=25, vocab_max_size=4000,
        ngram_max_n=5, ngram_min_count=2, ngram_max_sequences=1000000,
        pmi_window=5, pmi_min_count=2, pmi_cap=10, pmi_weight=5,
        recall_scale=1600, field_weight=1, same_word_penalty=200,
        beta_type=0.001, beta_word=0.001,
        copy_enabled=True, copy_min_context=2, copy_min_confidence=0.25,
        ising_enabled=True, skip_pmi_max_dist=5,
        recall_primary_mode=True,
        topic_spin_enabled=True, topic_n_topics=16,
        topic_coherence_penalty=400, topic_spin_flip_interval=20,
        topic_context_window=30, topic_coupling_scale=100,
        interpolated=True, kn_backoff=True,
        knowledge_scale=0, spin3_scale=0, category_scale=0, logic_rule_scale=0,
        use_conceptnet=False, auto_calibrate_beta=True, max_closed_class_run=2,
        grassmann_flag_enabled=False,
        grassmann_n_clusters=64, grassmann_n_topics=16,
        grassmann_wedge_weight=80, grassmann_max_wedge_distance=3,
        grassmann_max_cluster_ngram=6, grassmann_cluster_recall_scale=200,
        context_accumulator_enabled=(acc_weight > 0),
        accumulator_weight=acc_weight,
        accumulator_context_window=50,
        accumulator_decay_interval=10, accumulator_histogram_increment=16,
        caf_spin3_weight=200,
        caf_spin3_window=20, caf_spin3_min_count=3,
        caf_confidence_min_count=10, caf_min_confidence_q8=128,
    )
    model.train(n_samples=len(texts), texts=texts)
    ppl = model.compute_perplexity(n_samples=10)
    print(f"RESULT: acc_weight={acc_weight} PPL={ppl:.1f}", flush=True)
