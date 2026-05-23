#!/usr/bin/env python3
"""
v16.1 Training Script — Sparse Long-Range Word-Word Coupling (Memory-Fixed)

WHY v15 FAILED
--------------
Context Accumulator routed through 64 clusters → diluted signal:
  H2W per-word energy: ±63 (0.3% of recall's ±32000)
  3-Spin per-word energy: ±828 (4.3% of recall)
  PPL = 50.35 (no improvement over v12.1's 50.53)

v16.0 OOM BUG (FIXED in v16.1)
-------------------------------
Phase 2 collected ALL (target, context) pairs into giant arrays:
  27M tokens × 30 offsets ≈ 810M pairs × 16 bytes = 13GB intermediates
  With 6GB already used by n-gram index → OOM even on 16GB Pi!
v16.1 FIX: Incremental CSR construction — each batch converted to CSR
  immediately and summed into accumulator. Peak memory: ~1 batch only.
Also fixed boundary detection bug (boundary_within_d computed but never used).

v16 ARCHITECTURE: DIRECT WORD-WORD COUPLING (NO CLUSTERS)
----------------------------------------------------------
For each target word, store top-200 context words with their PMI.
At inference: sum PMI contributions from last 30 tokens → per-word energy.

KEY DIFFERENCE from v15:
  v15: context → cluster histogram → H2W cluster→word → per-word (DILUTED)
  v16: context → direct word→word PMI → per-word (NOT diluted)

Expected energy range:
  5 active context words × PMI 5 × decay 0.5 × weight 800 = 10,000
  This COMPETES with recall (±32,000) — can actually change rankings!

Usage:
  python -u train_v16.py                          # Default: 500K samples
  python -u train_v16.py --samples 1000000         # 1M samples
  python -u train_v16.py --no-longrange             # Ablation: without long-range

NOTE: Always use `python -u` or `PYTHONUNBUFFERED=1` when running with nohup!
"""

# ─── UNBUFFERED OUTPUT ────────────────────────────────────────────
import os
import sys
os.environ["PYTHONUNBUFFERED"] = "1"

# Ensure src/ is on the Python path
from pathlib import Path
_src = str(Path(__file__).parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

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

# v16 Long-Range Coupling defaults
DEFAULT_LONGRANGE_WEIGHT = 800      # 50% of recall_scale
DEFAULT_LONGRANGE_WINDOW = 30       # 30-token look-back
DEFAULT_LONGRANGE_TOP_K = 200       # Top-200 context words per target
DEFAULT_LONGRANGE_PMI_CAP = 64      # PMI cap in Q3 (8× real PMI)
DEFAULT_LONGRANGE_MIN_COUNT = 5     # Min co-occurrence count
DEFAULT_LR_CONFIDENCE_MIN_COUNT = 10
DEFAULT_LR_MIN_CONFIDENCE_Q8 = 128  # Minimum 50% confidence

# Inherited parameters
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
    parser = argparse.ArgumentParser(description="v16.0 Training — Sparse Long-Range Word-Word Coupling")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
    parser.add_argument("--recall-scale", type=int, default=DEFAULT_RECALL_SCALE)
    parser.add_argument("--pmi-weight", type=int, default=DEFAULT_PMI_WEIGHT)
    parser.add_argument("--same-word-penalty", type=int, default=DEFAULT_SAME_WORD_PENALTY)
    
    # v16 Long-Range Coupling parameters
    parser.add_argument("--no-longrange", action="store_true",
                        help="Ablation: disable long-range coupling")
    parser.add_argument("--longrange-weight", type=int, default=DEFAULT_LONGRANGE_WEIGHT)
    parser.add_argument("--longrange-window", type=int, default=DEFAULT_LONGRANGE_WINDOW)
    parser.add_argument("--longrange-top-k", type=int, default=DEFAULT_LONGRANGE_TOP_K)
    parser.add_argument("--longrange-pmi-cap", type=int, default=DEFAULT_LONGRANGE_PMI_CAP)
    parser.add_argument("--longrange-min-count", type=int, default=DEFAULT_LONGRANGE_MIN_COUNT)
    parser.add_argument("--lr-confidence-min-count", type=int, default=DEFAULT_LR_CONFIDENCE_MIN_COUNT)
    parser.add_argument("--lr-min-confidence-q8", type=int, default=DEFAULT_LR_MIN_CONFIDENCE_Q8)
    
    # Inherited cluster/topic parameters
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
    output_dir = OUTPUT_DIR / f"v16_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70, flush=True)
    print("ISING SPIN GLASS LANGUAGE MODEL — v16.1 LONG-RANGE COUPLING (MEMORY-FIXED)", flush=True)
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
    longrange_enabled = not args.no_longrange
    
    print(f"\n{'=' * 70}")
    print(f"CONFIG: v16.0 — Sparse Long-Range Word-Word Coupling")
    print(f"  LONG-RANGE COUPLING LAYER:")
    print(f"    enabled={longrange_enabled}")
    print(f"    longrange_weight={args.longrange_weight}")
    print(f"    longrange_window={args.longrange_window}")
    print(f"    longrange_top_k={args.longrange_top_k}")
    print(f"    longrange_pmi_cap={args.longrange_pmi_cap}")
    print(f"    longrange_min_count={args.longrange_min_count}")
    print(f"    lr_confidence_min_count={args.lr_confidence_min_count}")
    print(f"    lr_min_confidence_q8={args.lr_min_confidence_q8}")
    print(f"  INHERITED PARAMETERS:")
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
        # PMI — keep LOW since Long-Range Coupling provides the coupling
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
        # Disabled layers (recall-primary mode + Long-Range replaces these)
        knowledge_scale=0,
        spin3_scale=0,
        category_scale=0,
        logic_rule_scale=0,
        use_conceptnet=False,
        # Beta
        auto_calibrate_beta=True,
        max_closed_class_run=2,
        # Grassmann Flag parameters (used by Context Accumulator, disabled by longrange)
        grassmann_flag_enabled=False,
        grassmann_n_clusters=args.n_clusters,
        grassmann_n_topics=args.n_topics,
        grassmann_wedge_weight=args.wedge_weight,
        grassmann_max_wedge_distance=args.max_wedge_dist,
        grassmann_max_cluster_ngram=args.max_cluster_ngram,
        grassmann_cluster_recall_scale=args.cluster_recall_scale,
        # v15: Context Accumulator (disabled — longrange supersedes it)
        context_accumulator_enabled=False,
        accumulator_weight=0,
        # ─── v16.0: LONG-RANGE COUPLING LAYER ──────────────
        longrange_enabled=longrange_enabled,
        longrange_weight=args.longrange_weight,
        longrange_window=args.longrange_window,
        longrange_top_k=args.longrange_top_k,
        longrange_pmi_cap=args.longrange_pmi_cap,
        longrange_min_count=args.longrange_min_count,
        longrange_confidence_min_count=args.lr_confidence_min_count,
        longrange_min_confidence_q8=args.lr_min_confidence_q8,
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
    
    # ─── Long-Range Coupling diagnostics ─────────────────────
    if model.longrange_layer is not None:
        lr_diag = model.longrange_layer.get_diagnostics()
        print(f"\nLong-Range Coupling Diagnostics:")
        print(f"  LR hits: {lr_diag.get('lr_hits', 0)}")
        print(f"  LR energy sum: {lr_diag.get('lr_energy_sum', 0)}")
        print(f"  LR zero count: {lr_diag.get('lr_zero_count', 0)}")
        print(f"  LR total candidates: {lr_diag.get('lr_total_candidates', 0)}")
        avg_conf = lr_diag.get('confidence_sum', 0) / max(1, lr_diag.get('confidence_count', 1))
        print(f"  Avg recall confidence: {avg_conf:.1f}/256 ({avg_conf*100/256:.0f}%)")
        
        # Energy per hit
        hits = lr_diag.get('lr_hits', 0)
        if hits > 0:
            avg_energy = lr_diag.get('lr_energy_sum', 0) / hits
            print(f"  Avg energy per hit: {avg_energy:.0f}")
    
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
            lr = stats.get("lr_hits", 0)
            print(f"  recalls={recalls} copies={copies} lr_hits={lr}")
            
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
        "version": "v16.0",
        "architecture": "Sparse Long-Range Word-Word Coupling (direct PMI, no clusters)",
        "timestamp": timestamp,
        "config": {
            "longrange_enabled": longrange_enabled,
            "longrange_weight": args.longrange_weight,
            "longrange_window": args.longrange_window,
            "longrange_top_k": args.longrange_top_k,
            "longrange_pmi_cap": args.longrange_pmi_cap,
            "longrange_min_count": args.longrange_min_count,
            "lr_confidence_min_count": args.lr_confidence_min_count,
            "lr_min_confidence_q8": args.lr_min_confidence_q8,
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
    print(f"DONE — v16.0 Sparse Long-Range Word-Word Coupling")
    print(f"Total time: {t_total:.1f}s ({t_total/60:.1f}min)")
    print(f"Results: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
