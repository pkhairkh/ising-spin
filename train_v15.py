#!/usr/bin/env python3
"""
v15.0 Training Script — Context Accumulator Architecture (DUAL-PATH)

THE FUNDAMENTAL PROBLEM WITH v14 AND EARLIER
---------------------------------------------
All non-recall energy layers are capped at 5-10% of recall_scale.
They can NEVER override recall's word ranking. The model is effectively
just a smoothed n-gram model with tiny perturbations.

v15 ARCHITECTURE: CONTEXT ACCUMULATOR FIELD (DUAL-PATH)
-------------------------------------------------------
Three NEW mechanisms that can actually COMPETE with recall:

1. CLUSTER HISTOGRAM ACCUMULATOR + H2W FIELD (50% of recall_scale)
   - Running histogram H[64] over last 50 tokens (exponential decay)
   - H2W coupling matrix: J_h2w[c, w] maps cluster activation → word energy
   - field[w] = Σ_c H[c] × J_h2w[c, w] → can reach ±800
   - THIS COMPETES WITH RECALL for the first time

2. CLUSTER 3-SPIN COUPLINGS (25% of recall_scale)
   - When clusters A,B both active in context, cluster C's words get bonus
   - 64³ = 262K triples, estimated from training data
   - Compositional: captures "A + B → C" patterns n-grams miss

3. RECALL CONFIDENCE SCALING
   - When n-gram match is weak (1-gram backoff, low count): recall_scale drops
   - When n-gram match is strong (5-gram, high count): recall_scale stays at 100%
   - This lets the accumulator MATTER when recall is uncertain

ENERGY BALANCE (v15):
  recall_scale       = 1600   [PRIMARY — but confidence-scaled]
  accumulator_weight = 800    [50% of recall — CAN override weak recall]
  spin3_weight       = 400    [25% of recall — compositional]
  cluster_ngram      = 200    [12.5% of recall — long-range sequential]
  wedge              = 80     [5% of recall — directional]
  topic              = 400    [25% of recall — coherence penalty]

Target: PPL ~30 on Pi (16GB RAM, 4 cores)

Usage:
  python -u train_v15.py                          # Default: 500K samples
  python -u train_v15.py --samples 1000000         # 1M samples
  python -u train_v15.py --no-accumulator           # Ablation: without H2W
  python -u train_v15.py --no-3spin                 # Ablation: without 3-spin

NOTE: Always use `python -u` or `PYTHONUNBUFFERED=1` when running with nohup!
"""

# ─── UNBUFFERED OUTPUT ────────────────────────────────────────────
import os
os.environ["PYTHONUNBUFFERED"] = "1"

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────

DEFAULT_SAMPLES = 500000
DEFAULT_VOCAB = 4000
DEFAULT_RECALL_SCALE = 1600
DEFAULT_PMI_WEIGHT = 5
DEFAULT_SAME_WORD_PENALTY = 200

# v15 Context Accumulator defaults
DEFAULT_ACCUMULATOR_WEIGHT = 800     # 50% of recall_scale
DEFAULT_CONTEXT_WINDOW = 50          # Look-back window for H2W
DEFAULT_DECAY_INTERVAL = 10          # Histogram decay interval
DEFAULT_HISTOGRAM_INCREMENT = 16     # How much to add per cluster
DEFAULT_SPIN3_WEIGHT = 400           # 25% of recall_scale
DEFAULT_SPIN3_WINDOW = 20            # 3-spin look-back window
DEFAULT_SPIN3_MIN_COUNT = 3          # Minimum triple count
DEFAULT_CONFIDENCE_MIN_COUNT = 10    # Min count for full recall confidence
DEFAULT_MIN_CONFIDENCE_Q8 = 128      # Minimum 50% confidence

# Grassmann/cluster parameters (inherited from v14.1)
DEFAULT_N_CLUSTERS = 64
DEFAULT_N_TOPICS = 16
DEFAULT_WEDGE_WEIGHT = 80
DEFAULT_MAX_WEDGE_DIST = 3
DEFAULT_CLUSTER_RECALL_SCALE = 200
DEFAULT_MAX_CLUSTER_NGRAM = 6

# Cache directory
CACHE_DIR = Path(__file__).parent
OUTPUT_DIR = CACHE_DIR / "output"


def get_rss_mb() -> int:
    """Get current process RSS in MB."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        except Exception:
            return 0
    return 0


def find_cache_file(n_samples: int) -> str:
    """Find the best cache file for the requested number of samples."""
    cache_files = {}
    for f in CACHE_DIR.glob("cached_fineweb_*.json"):
        name = f.stem
        size_str = name.split("_")[-1]
        try:
            if size_str.endswith("k"):
                size = int(size_str[:-1]) * 1000
            elif size_str.endswith("m"):
                size = int(size_str[:-1]) * 1000000
            else:
                size = int(size_str)
            cache_files[size] = str(f)
        except (ValueError, IndexError):
            continue
    
    if n_samples in cache_files:
        return cache_files[n_samples]
    
    sufficient = {s: p for s, p in cache_files.items() if s >= n_samples}
    if sufficient:
        best_size = min(sufficient.keys())
        return sufficient[best_size]
    
    return None


def load_data(n_samples: int) -> list:
    """Load or download FineWeb-Edu data with proper cache handling."""
    cache_path = find_cache_file(n_samples)
    
    if cache_path:
        print(f"Loading cache: {cache_path}")
        t0 = time.time()
        with open(cache_path, "r") as f:
            texts = json.load(f)
        t_load = time.time() - t0
        print(f"  {len(texts):,} texts loaded in {t_load:.1f}s")
        
        if len(texts) >= n_samples:
            return texts[:n_samples]
        else:
            print(f"  Cache has {len(texts):,} texts but need {n_samples:,}")
    
    print("No cached data found. Downloading from HuggingFace...")
    from ising_spin.model import load_fineweb_edu
    t0 = time.time()
    texts = load_fineweb_edu(n_samples=n_samples)
    print(f"  Downloaded {len(texts):,} texts in {time.time()-t0:.1f}s")
    
    if n_samples >= 1000000 and n_samples % 1000000 == 0:
        cache_name = f"cached_fineweb_{n_samples // 1000000}m.json"
    elif n_samples >= 1000 and n_samples % 1000 == 0:
        cache_name = f"cached_fineweb_{n_samples // 1000}k.json"
    else:
        cache_name = f"cached_fineweb_{n_samples}.json"
    
    cache_file = CACHE_DIR / cache_name
    print(f"  Saving cache to: {cache_file}")
    with open(cache_file, "w") as f:
        json.dump(texts, f)
    
    return texts


def beta_sweep_ppl(model, beta_factors=None, n_seqs=10):
    """Two-phase beta sweep to find optimal PPL."""
    import numpy as np
    from ising_spin.model import LN2_NUM, LN2_DEN, IntegerBoltzmannSampler
    
    recall_scale = model.recall_scale
    base_beta = 0.5 * float(LN2_NUM) / float(LN2_DEN) / recall_scale
    
    if beta_factors is None:
        beta_factors = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00, 5.00]
    
    best_ppl = float('inf')
    best_f = 1.0
    
    print("\nBeta sweep (Phase 1 — coarse):")
    for f in beta_factors:
        new_beta = base_beta * f
        model.generator.word_sampler = IntegerBoltzmannSampler(
            beta=new_beta, max_delta=50000
        )
        try:
            ppl = model.compute_perplexity(n_samples=n_seqs)
            marker = " <-- BEST" if ppl < best_ppl else ""
            print(f"    f={f:.2f} (β={new_beta:.6f}): PPL={ppl:.1f}{marker}")
            if ppl < best_ppl:
                best_ppl = ppl
                best_f = f
        except Exception as e:
            print(f"    f={f:.2f}: Error: {e}")
    
    # Phase 2: Fine sweep
    fine_lo = max(0.5, best_f - 0.5)
    fine_hi = best_f + 0.5
    fine_factors = np.arange(fine_lo, fine_hi + 0.1, 0.1).tolist()
    fine_factors = [f for f in fine_factors if abs(f - best_f) > 0.05 or f == best_f]
    
    if len(fine_factors) > 1:
        print(f"\nBeta sweep (Phase 2 — fine around f={best_f:.2f}):")
        for f in fine_factors:
            new_beta = base_beta * f
            model.generator.word_sampler = IntegerBoltzmannSampler(
                beta=new_beta, max_delta=50000
            )
            try:
                ppl = model.compute_perplexity(n_samples=n_seqs)
                marker = " <-- BEST" if ppl < best_ppl else ""
                print(f"    f={f:.2f} (β={new_beta:.6f}): PPL={ppl:.1f}{marker}")
                if ppl < best_ppl:
                    best_ppl = ppl
                    best_f = f
            except Exception as e:
                print(f"    f={f:.2f}: Error: {e}")
    
    model.generator.word_sampler = IntegerBoltzmannSampler(
        beta=base_beta * best_f, max_delta=50000
    )
    print(f"\nBest: f={best_f:.2f} (β={base_beta * best_f:.6f}), PPL={best_ppl:.1f}")
    
    return best_f, best_ppl


def main():
    parser = argparse.ArgumentParser(description="v15.0 Training — Context Accumulator Architecture")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
    parser.add_argument("--recall-scale", type=int, default=DEFAULT_RECALL_SCALE)
    parser.add_argument("--pmi-weight", type=int, default=DEFAULT_PMI_WEIGHT)
    parser.add_argument("--same-word-penalty", type=int, default=DEFAULT_SAME_WORD_PENALTY)
    
    # v15 Context Accumulator parameters
    parser.add_argument("--no-accumulator", action="store_true",
                        help="Ablation: disable H2W accumulator field")
    parser.add_argument("--no-3spin", action="store_true",
                        help="Ablation: disable cluster 3-spin couplings")
    parser.add_argument("--accumulator-weight", type=int, default=DEFAULT_ACCUMULATOR_WEIGHT)
    parser.add_argument("--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW)
    parser.add_argument("--decay-interval", type=int, default=DEFAULT_DECAY_INTERVAL)
    parser.add_argument("--histogram-increment", type=int, default=DEFAULT_HISTOGRAM_INCREMENT)
    parser.add_argument("--spin3-weight", type=int, default=DEFAULT_SPIN3_WEIGHT)
    parser.add_argument("--spin3-window", type=int, default=DEFAULT_SPIN3_WINDOW)
    parser.add_argument("--spin3-min-count", type=int, default=DEFAULT_SPIN3_MIN_COUNT)
    parser.add_argument("--confidence-min-count", type=int, default=DEFAULT_CONFIDENCE_MIN_COUNT)
    parser.add_argument("--min-confidence-q8", type=int, default=DEFAULT_MIN_CONFIDENCE_Q8)
    
    # Grassmann/cluster parameters (inherited)
    parser.add_argument("--n-clusters", type=int, default=DEFAULT_N_CLUSTERS)
    parser.add_argument("--n-topics", type=int, default=DEFAULT_N_TOPICS)
    parser.add_argument("--wedge-weight", type=int, default=DEFAULT_WEDGE_WEIGHT)
    parser.add_argument("--max-wedge-dist", type=int, default=DEFAULT_MAX_WEDGE_DIST)
    parser.add_argument("--cluster-recall-scale", type=int, default=DEFAULT_CLUSTER_RECALL_SCALE)
    parser.add_argument("--max-cluster-ngram", type=int, default=DEFAULT_MAX_CLUSTER_NGRAM)
    
    # Standard parameters
    parser.add_argument("--no-topic-spin", action="store_true")
    parser.add_argument("--no-kn-backoff", action="store_true")
    parser.add_argument("--no-interpolated", action="store_true")
    parser.add_argument("--ngram-min-count", type=int, default=2)
    parser.add_argument("--ngram-max-seqs", type=int, default=1000000)
    parser.add_argument("--max-seq-len", type=int, default=30)
    args = parser.parse_args()
    
    # ─── Header ─────────────────────────────────────────────
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / f"v15_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70, flush=True)
    print("ISING SPIN GLASS LANGUAGE MODEL — v15.0 CONTEXT ACCUMULATOR", flush=True)
    print(f"Started: {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Workers: {os.cpu_count()}", flush=True)
    rss = get_rss_mb()
    if rss > 0:
        print(f"Memory (RSS): {rss:,} MB", flush=True)
    print("=" * 70, flush=True)
    
    # ─── Load Data ──────────────────────────────────────────
    texts = load_data(args.samples)
    n_texts = len(texts)
    print(f"Using {n_texts:,} texts for training")
    
    # ─── Config ─────────────────────────────────────────────
    topic_spin = not args.no_topic_spin
    kn_backoff = not args.no_kn_backoff
    interpolated = not args.no_interpolated
    accumulator_enabled = not args.no_accumulator
    spin3_enabled = not args.no_3spin
    
    print(f"\n{'=' * 70}")
    print(f"CONFIG: v15.0 — Context Accumulator Architecture (DUAL-PATH)")
    print(f"  CONTEXT ACCUMULATOR LAYER:")
    print(f"    enabled={accumulator_enabled}")
    print(f"    accumulator_weight={args.accumulator_weight}")
    print(f"    context_window={args.context_window}")
    print(f"    decay_interval={args.decay_interval}")
    print(f"    histogram_increment={args.histogram_increment}")
    print(f"    spin3_enabled={spin3_enabled}")
    print(f"    spin3_weight={args.spin3_weight}")
    print(f"    spin3_window={args.spin3_window}")
    print(f"    spin3_min_count={args.spin3_min_count}")
    print(f"    confidence_min_count={args.confidence_min_count}")
    print(f"    min_confidence_q8={args.min_confidence_q8}")
    print(f"  CLUSTER PARAMETERS (inherited from v14.1):")
    print(f"    n_clusters={args.n_clusters}")
    print(f"    n_topics={args.n_topics}")
    print(f"    wedge_weight={args.wedge_weight}")
    print(f"    max_wedge_dist={args.max_wedge_dist}")
    print(f"    cluster_recall_scale={args.cluster_recall_scale}")
    print(f"    max_cluster_ngram={args.max_cluster_ngram}")
    print(f"  STANDARD:")
    print(f"    vocab_max_size={args.vocab}")
    print(f"    kn_backoff={kn_backoff}")
    print(f"    interpolated={interpolated}")
    print(f"    topic_spin={topic_spin}")
    print(f"    recall_scale={args.recall_scale}")
    print(f"    pmi_weight={args.pmi_weight}")
    print(f"    n_samples={n_texts:,}")
    print(f"{'=' * 70}")
    
    # ─── Train ──────────────────────────────────────────────
    from ising_spin.model import IsingLMModel
    
    model = IsingLMModel(
        # Vocabulary
        vocab_min_freq=25,
        vocab_max_size=args.vocab,
        # N-gram
        ngram_max_n=5,
        ngram_min_count=args.ngram_min_count,
        ngram_max_sequences=args.ngram_max_seqs,
        # PMI — keep LOW since Context Accumulator provides the coupling
        pmi_window=5,
        pmi_min_count=2,
        pmi_cap=10,
        pmi_weight=args.pmi_weight,
        # Energy scales
        recall_scale=args.recall_scale,
        field_weight=1,
        same_word_penalty=args.same_word_penalty,
        # Beta
        beta_type=0.001,
        beta_word=0.001,
        # Copy mechanism
        copy_enabled=True,
        copy_min_context=2,
        copy_min_confidence=0.25,
        # Ising
        ising_enabled=True,
        skip_pmi_max_dist=5,
        # v8.0: Recall-primary mode
        recall_primary_mode=True,
        # v8.2: Topic Spin (Potts coherence layer)
        topic_spin_enabled=topic_spin,
        topic_n_topics=16,
        topic_coherence_penalty=400,
        topic_spin_flip_interval=20,
        topic_context_window=30,
        topic_coupling_scale=100,
        # v9.0: Interpolated n-gram smoothing
        interpolated=interpolated,
        # v10.0: Kneser-Ney backoff
        kn_backoff=kn_backoff,
        # Disabled layers (recall-primary mode + Context Accumulator replaces these)
        knowledge_scale=0,
        spin3_scale=0,
        category_scale=0,
        logic_rule_scale=0,
        use_conceptnet=False,
        # Beta
        auto_calibrate_beta=True,
        max_closed_class_run=2,
        # Grassmann Flag parameters (used by Context Accumulator internally)
        grassmann_flag_enabled=False,  # Context Accumulator supersedes it
        grassmann_n_clusters=args.n_clusters,
        grassmann_n_topics=args.n_topics,
        grassmann_wedge_weight=args.wedge_weight,
        grassmann_max_wedge_distance=args.max_wedge_dist,
        grassmann_max_cluster_ngram=args.max_cluster_ngram,
        grassmann_cluster_recall_scale=args.cluster_recall_scale,
        # ─── v15.0: CONTEXT ACCUMULATOR LAYER ───────────────
        context_accumulator_enabled=accumulator_enabled,
        accumulator_weight=args.accumulator_weight,
        accumulator_context_window=args.context_window,
        accumulator_decay_interval=args.decay_interval,
        accumulator_histogram_increment=args.histogram_increment,
        caf_spin3_weight=args.spin3_weight if spin3_enabled else 0,
        caf_spin3_window=args.spin3_window,
        caf_spin3_min_count=args.spin3_min_count,
        caf_confidence_min_count=args.confidence_min_count,
        caf_min_confidence_q8=args.min_confidence_q8,
    )
    
    # Monkey-patch truncate_sequences
    import ising_spin.model as model_module
    _original_truncate = model_module.truncate_sequences
    
    def _custom_truncate(sequences, max_len=args.max_seq_len):
        return _original_truncate(sequences, max_len=max_len)
    
    model_module.truncate_sequences = _custom_truncate
    
    t_start = time.time()
    
    try:
        model.train(n_samples=n_texts, texts=texts)
    except MemoryError:
        rss = get_rss_mb()
        print(f"\n!!! OUT OF MEMORY during training !!! (RSS: {rss:,} MB)")
        sys.exit(1)
    except Exception as e:
        rss = get_rss_mb()
        print(f"\n!!! TRAINING ERROR: {e} !!! (RSS: {rss:,} MB)")
        traceback.print_exc()
        sys.exit(1)
    
    t_train = time.time() - t_start
    rss = get_rss_mb()
    print(f"\nTraining complete: {t_train:.1f}s ({t_train/60:.1f}min)")
    print(f"  Vocab size: {len(model.vocab)}")
    if rss > 0:
        print(f"  Peak memory (RSS): {rss:,} MB")
    
    model_module.truncate_sequences = _original_truncate
    
    # ─── Evaluation ─────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("EVALUATION")
    print(f"{'=' * 70}")
    
    # Quick PPL at default beta
    print("\nQuick PPL (default beta, 10 seqs):", end=" ")
    try:
        quick_ppl = model.compute_perplexity(n_samples=10)
        print(f"{quick_ppl:.1f}")
    except Exception as e:
        print(f"Error: {e}")
        quick_ppl = 999
    
    # Beta sweep
    best_f, best_sweep_ppl = beta_sweep_ppl(model, n_seqs=10)
    
    # Full PPL evaluation at best beta
    print(f"\nFull PPL evaluation...")
    try:
        full_ppl = model.compute_perplexity(n_samples=100)
        print(f"  Perplexity: {full_ppl:.2f}")
    except Exception as e:
        print(f"  Error: {e}")
        full_ppl = best_sweep_ppl
    
    print(f"\nPPL (full, 100 seqs): {full_ppl:.2f}")
    
    # ─── Context Accumulator diagnostics ─────────────────────
    if model.context_accumulator_layer is not None:
        ca_diag = model.context_accumulator_layer.get_diagnostics()
        print(f"\nContext Accumulator Diagnostics:")
        print(f"  H2W hits: {ca_diag.get('h2w_hits', 0)}")
        print(f"  3-Spin hits: {ca_diag.get('spin3_hits', 0)}")
        print(f"  Cluster n-gram hits: {ca_diag.get('cluster_ngram_hits', 0)}")
        print(f"  Wedge hits: {ca_diag.get('wedge_hits', 0)}")
        print(f"  H2W energy sum: {ca_diag.get('h2w_energy_sum', 0)}")
        print(f"  3-Spin energy sum: {ca_diag.get('spin3_energy_sum', 0)}")
        avg_conf = ca_diag.get('avg_confidence', 0)
        print(f"  Avg recall confidence: {avg_conf:.1f}/256 ({avg_conf*100/256:.0f}%)")
    
    # ─── Generation ─────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"GENERATION (PPL={full_ppl:.2f})")
    print(f"{'=' * 70}")
    
    prompts = ["the history of", "science and technology", "research shows that"]
    generated_texts = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n  --- '{prompt}' (400 words) ---")
        try:
            result = model.generator.generate(prompt=prompt, length=400)
            text = result.get("text", str(result))
            if isinstance(text, list):
                text = " ".join(text)
            print(f"  {text[:300]}...")
            
            stats = model.generator.get_stats()
            recalls = stats.get("recall_hit", 0)
            copies = stats.get("copy_used", 0)
            h2w = stats.get("caf_h2w_hits", 0)
            s3 = stats.get("caf_spin3_hits", 0)
            print(f"  recalls={recalls} copies={copies} "
                  f"CAF: h2w={h2w} 3spin={s3}")
            
            generated_texts.append(text)
            
            gen_file = output_dir / f"generated_{i}.txt"
            with open(gen_file, "w") as f:
                f.write(text)
        except Exception as e:
            print(f"  Generation error: {e}")
            traceback.print_exc()
            generated_texts.append("")
    
    # ─── Save Results ───────────────────────────────────────
    results = {
        "version": "v15.0",
        "architecture": "Context Accumulator (H2W field + 3-spin + confidence + cluster n-gram + wedge)",
        "timestamp": timestamp,
        "config": {
            "context_accumulator_enabled": accumulator_enabled,
            "accumulator_weight": args.accumulator_weight,
            "context_window": args.context_window,
            "decay_interval": args.decay_interval,
            "histogram_increment": args.histogram_increment,
            "spin3_enabled": spin3_enabled,
            "spin3_weight": args.spin3_weight if spin3_enabled else 0,
            "spin3_window": args.spin3_window,
            "spin3_min_count": args.spin3_min_count,
            "confidence_min_count": args.confidence_min_count,
            "min_confidence_q8": args.min_confidence_q8,
            "n_clusters": args.n_clusters,
            "n_topics": args.n_topics,
            "wedge_weight": args.wedge_weight,
            "max_wedge_dist": args.max_wedge_dist,
            "cluster_recall_scale": args.cluster_recall_scale,
            "max_cluster_ngram": args.max_cluster_ngram,
            "vocab_max_size": args.vocab,
            "recall_scale": args.recall_scale,
            "pmi_weight": args.pmi_weight,
            "kn_backoff": kn_backoff,
            "interpolated": interpolated,
            "n_samples": n_texts,
        },
        "results": {
            "training_time_sec": t_train,
            "quick_ppl": quick_ppl,
            "best_beta_factor": best_f,
            "best_sweep_ppl": best_sweep_ppl,
            "full_ppl": full_ppl,
            "vocab_size": len(model.vocab),
            "peak_rss_mb": get_rss_mb(),
        },
    }
    
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {results_file}")
    
    root_results = CACHE_DIR / "training_results.json"
    with open(root_results, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {root_results}")
    
    t_total = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"DONE — v15.0 Context Accumulator Architecture")
    print(f"Total time: {t_total:.1f}s ({t_total/60:.1f}min)")
    print(f"Results: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
