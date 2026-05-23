#!/usr/bin/env python3
"""
v14.1 Training Script — Grassmann Flag Architecture (FIXED)

v14.0 FAILED: PPL=50.67, same as v12.1. All three Grassmann sub-layers
(flag, wedge, block memory) contributed ~10K undifferentiated energy.

v14.1 FIX: Replace redundant layers with CLUSTER N-GRAM RECALL.

The GENUINELY NOVEL contribution: word n-grams can only go to 5-gram
(4000^5 is impossibly sparse), but cluster n-grams with alphabet 64 can
go to 8-gram. This provides LONG-RANGE context that word n-grams CANNOT.

Usage:
  python -u train_v14_1.py                          # Default: 500K samples
  python -u train_v14_1.py --samples 1000000         # 1M samples
  python -u train_v14_1.py --no-grassmann            # Ablation: without Grassmann
  python -u train_v14_1.py --cluster-scale 400       # Higher cluster recall weight
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

# Grassmann Flag Layer v14.1 defaults
DEFAULT_N_CLUSTERS = 64
DEFAULT_N_TOPICS = 16
DEFAULT_WEDGE_WEIGHT = 80        # ~5% of recall_scale
DEFAULT_MAX_WEDGE_DIST = 3       # Only nearby context
DEFAULT_MAX_CLUSTER_NGRAM = 6    # Cluster 2-6 grams
DEFAULT_CLUSTER_RECALL_SCALE = 200  # 10% of recall_scale

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
    parser = argparse.ArgumentParser(description="v14.1 Training — Grassmann Flag Architecture (FIXED)")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
    parser.add_argument("--recall-scale", type=int, default=DEFAULT_RECALL_SCALE)
    parser.add_argument("--pmi-weight", type=int, default=DEFAULT_PMI_WEIGHT)
    parser.add_argument("--same-word-penalty", type=int, default=DEFAULT_SAME_WORD_PENALTY)
    
    # Grassmann Flag Layer v14.1 parameters
    parser.add_argument("--no-grassmann", action="store_true",
                        help="Disable Grassmann Flag Layer (ablation)")
    parser.add_argument("--n-clusters", type=int, default=DEFAULT_N_CLUSTERS)
    parser.add_argument("--n-topics", type=int, default=DEFAULT_N_TOPICS)
    parser.add_argument("--wedge-weight", type=int, default=DEFAULT_WEDGE_WEIGHT)
    parser.add_argument("--max-wedge-dist", type=int, default=DEFAULT_MAX_WEDGE_DIST)
    parser.add_argument("--max-cluster-ngram", type=int, default=DEFAULT_MAX_CLUSTER_NGRAM)
    parser.add_argument("--cluster-scale", type=int, default=DEFAULT_CLUSTER_RECALL_SCALE)
    
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
    output_dir = OUTPUT_DIR / f"v14_1_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70, flush=True)
    print("ISING SPIN GLASS LANGUAGE MODEL — v14.1 GRASSMANN FLAG (FIXED)", flush=True)
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
    grassmann_enabled = not args.no_grassmann
    topic_spin = not args.no_topic_spin
    kn_backoff = not args.no_kn_backoff
    interpolated = not args.no_interpolated
    
    print(f"\n{'=' * 70}")
    print(f"CONFIG: v14.1 — Grassmann Flag Architecture (FIXED)")
    print(f"  CHANGES FROM v14.0:")
    print(f"    REMOVED: Flag cluster energy (redundant with word n-gram recall)")
    print(f"    REMOVED: Flag topic energy (redundant with Potts topic spin)")
    print(f"    REMOVED: Block memory (replaced by cluster n-gram recall)")
    print(f"    ADDED: Cluster n-gram recall (2-{args.max_cluster_ngram} gram, KN-smoothed)")
    print(f"    FIXED: Wedge coupling (zero-meaned, capped, distance 1-3)")
    print(f"  GRASSMANN FLAG LAYER:")
    print(f"    enabled={grassmann_enabled}")
    print(f"    n_clusters={args.n_clusters}")
    print(f"    n_topics={args.n_topics}")
    print(f"    cluster_recall_scale={args.cluster_scale}")
    print(f"    max_cluster_ngram={args.max_cluster_ngram}")
    print(f"    wedge_weight={args.wedge_weight}")
    print(f"    max_wedge_distance={args.max_wedge_dist}")
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
        # PMI — keep LOW since Grassmann provides additional coupling
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
        # Disabled layers (recall-primary mode)
        knowledge_scale=0,
        spin3_scale=0,
        category_scale=0,
        logic_rule_scale=0,
        use_conceptnet=False,
        # Beta
        auto_calibrate_beta=True,
        max_closed_class_run=2,
        # ─── v14.1: GRASSMANN FLAG LAYER ───────────────────
        grassmann_flag_enabled=grassmann_enabled,
        grassmann_n_clusters=args.n_clusters,
        grassmann_n_topics=args.n_topics,
        grassmann_cluster_weight=0,         # DEPRECATED
        grassmann_topic_weight=0,           # DEPRECATED
        grassmann_wedge_weight=args.wedge_weight,
        grassmann_max_wedge_distance=args.max_wedge_dist,
        grassmann_block_size=32,            # DEPRECATED
        grassmann_max_blocks=0,             # DEPRECATED
        grassmann_memory_weight=0,          # DEPRECATED
        grassmann_max_cluster_ngram=args.max_cluster_ngram,
        grassmann_cluster_recall_scale=args.cluster_scale,
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
            print(f"  {text[:200]}...")
            
            stats = model.generator.get_stats()
            recalls = stats.get("recall_hit", 0)
            copies = stats.get("copy_used", 0)
            gf_diag = model.grassmann_flag_layer.get_diagnostics() if model.grassmann_flag_layer else {}
            print(f"  recalls={recalls} copies={copies} "
                  f"grassmann: cngram={gf_diag.get('cluster_ngram_hits', 0)} "
                  f"wedge={gf_diag.get('wedge_coupling_hits', 0)}")
            
            generated_texts.append(text)
            
            gen_file = output_dir / f"generated_{i}.txt"
            with open(gen_file, "w") as f:
                f.write(text)
        except Exception as e:
            print(f"  Generation error: {e}")
            generated_texts.append("")
    
    # ─── Save Results ───────────────────────────────────────
    results = {
        "version": "v14.1",
        "architecture": "Grassmann Flag v14.1 (cluster n-gram recall + wedge coupling)",
        "timestamp": timestamp,
        "config": {
            "grassmann_flag_enabled": grassmann_enabled,
            "n_clusters": args.n_clusters,
            "n_topics": args.n_topics,
            "cluster_recall_scale": args.cluster_scale,
            "max_cluster_ngram": args.max_cluster_ngram,
            "wedge_weight": args.wedge_weight,
            "max_wedge_distance": args.max_wedge_dist,
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
    print(f"DONE — v14.1 Grassmann Flag Architecture (FIXED)")
    print(f"Total time: {t_total:.1f}s ({t_total/60:.1f}min)")
    print(f"Results: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
