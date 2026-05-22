#!/usr/bin/env python3
"""
v12.2 Training Script — Extended Beta Sweep + Empirical Calibration

Key improvements over v12.1:
  1. Extended beta sweep range (f=0.5 to 5.0) — previous sweep only went to 1.5,
     but PPL was still dropping at that point. Two-phase sweep finds the true optimum.
  2. Empirical β calibration from median ΔE — theoretical β was too low for
     v12 configs (recall_scale=1600 + KN backoff). Now uses max(theoretical, empirical).
  3. Boltzmann table max_delta increased to 50K (was capped at 25K, losing discrimination)
  4. All previous fixes: OOM cap, unbuffered output, memory monitoring, etc.

Previous fixes (v12.1):
  - CRITICAL FIX: Multi-word prompt tokenization (was causing recalls=0)
  - CRITICAL FIX: Beta sweep creates NEW sampler per beta (table was cached)
  - CRITICAL FIX: KN backoff energy scaled 2x for energy discrimination
  - CRITICAL FIX: Stats key names (recall_hit, copy_used)
  - N-gram index capped at 1M sequences (avoids OOM on 3M+ texts)
  - Unbuffered output (PYTHONUNBUFFERED=1) — no more silent nohup hangs

Target: PPL ~20 on Pi (16GB RAM, 4 cores)

Usage:
  python -u train_v12.py                    # Default: 500K samples (ALWAYS use -u!)
  python -u train_v12.py --samples 1000000  # 1M samples
  python -u train_v12.py --samples 3000000  # 3M samples (n-gram capped at 1M)
  python -u train_v12.py --samples 5000000  # 5M samples (n-gram capped at 1M)

NOTE: Always use `python -u` or `PYTHONUNBUFFERED=1` when running with nohup!
"""

# ─── UNBUFFERED OUTPUT — prevents silent nohup hangs ─────────────────
import os
os.environ["PYTHONUNBUFFERED"] = "1"

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────

DEFAULT_SAMPLES = 1000000  # v12.1: 1M (same as successful baseline run)
DEFAULT_VOCAB = 4000
DEFAULT_RECALL_SCALE = 1600
DEFAULT_PMI_WEIGHT = 5
DEFAULT_SAME_WORD_PENALTY = 200  # v12.1: Match train_long.py baseline (5000 was too aggressive)
# v12.1: N-gram index uses at most this many sequences.
# This is the KEY fix for OOM on 3M+ texts. N-gram statistics converge
# well before the corpus is exhausted. PMI/skip-gram (which produce
# fixed-size sparse matrices) still use the full corpus.
DEFAULT_NGRAM_MAX_SEQS = 1000000

# Cache directory
CACHE_DIR = Path(__file__).parent
OUTPUT_DIR = CACHE_DIR / "output"


def get_rss_mb() -> int:
    """Get current process RSS in MB."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024  # KB -> MB
    except Exception:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        except Exception:
            return 0
    return 0


def find_cache_file(n_samples: int) -> str:
    """
    Find the best cache file for the requested number of samples.
    
    Rules:
      - Look for exact match first (e.g., cached_fineweb_500k.json for 500K)
      - Then look for a cache with ENOUGH data (>= n_samples)
      - Prefer the smallest sufficient cache to minimize loading time
      - Never use a cache with fewer texts than requested
    """
    cache_files = {}
    for f in CACHE_DIR.glob("cached_fineweb_*.json"):
        # Parse size from filename: cached_fineweb_500k.json -> 500000
        name = f.stem  # cached_fineweb_500k
        size_str = name.split("_")[-1]  # 500k
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
    
    # Exact match first
    if n_samples in cache_files:
        return cache_files[n_samples]
    
    # Find smallest cache with enough data
    sufficient = {s: p for s, p in cache_files.items() if s >= n_samples}
    if sufficient:
        best_size = min(sufficient.keys())
        return sufficient[best_size]
    
    # No cache has enough data
    return None


def load_data(n_samples: int) -> list:
    """
    Load or download FineWeb-Edu data with proper cache handling.
    """
    cache_path = find_cache_file(n_samples)
    
    if cache_path:
        print(f"Loading cache: {cache_path}")
        t0 = time.time()
        with open(cache_path, "r") as f:
            texts = json.load(f)
        t_load = time.time() - t0
        print(f"  {len(texts):,} texts loaded in {t_load:.1f}s")
        
        if len(texts) >= n_samples:
            # Use only the requested number
            return texts[:n_samples]
        else:
            # Cache doesn't have enough — need to download more
            print(f"  Cache has {len(texts):,} texts but need {n_samples:,}")
            print(f"  Downloading {n_samples:,} texts from FineWeb-Edu...")
            # Fall through to download
    
    # Download from HuggingFace
    print("No cached data found. Downloading from HuggingFace...")
    print("(This may take a while on first run)")
    
    from ising_spin.model import load_fineweb_edu
    t0 = time.time()
    texts = load_fineweb_edu(n_samples=n_samples)
    print(f"  Downloaded {len(texts):,} texts in {time.time()-t0:.1f}s")
    
    # Save cache with size-based naming
    if n_samples >= 1000000 and n_samples % 1000000 == 0:
        cache_name = f"cached_fineweb_{n_samples // 1000000}m.json"
    elif n_samples >= 1000 and n_samples % 1000 == 0:
        cache_name = f"cached_fineweb_{n_samples // 1000}k.json"
    else:
        cache_name = f"cached_fineweb_{n_samples}.json"
    
    cache_file = CACHE_DIR / cache_name
    print(f"  Saving cache to: {cache_file}")
    t0 = time.time()
    with open(cache_file, "w") as f:
        json.dump(texts, f)
    print(f"  Cache saved in {time.time()-t0:.1f}s")
    
    return texts


def beta_sweep_ppl(model, beta_factors=None, n_seqs=10):
    """
    Two-phase beta sweep to find optimal PPL.
    
    Phase 1: Coarse sweep over wide range (f=0.5 to 5.0)
    Phase 2: Fine sweep around the best coarse result
    
    v12.2: Extended range because previous sweeps showed PPL still
    dropping at f=1.5 — the optimal beta is much higher than theoretical.
    """
    import numpy as np
    from ising_spin.model import LN2_NUM, LN2_DEN, IntegerBoltzmannSampler
    
    recall_scale = model.recall_scale
    base_beta = 0.5 * float(LN2_NUM) / float(LN2_DEN) / recall_scale
    
    # Phase 1: Coarse sweep over wide range
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
    
    # Phase 2: Fine sweep around best coarse result
    fine_lo = max(0.5, best_f - 0.5)
    fine_hi = best_f + 0.5
    fine_factors = np.arange(fine_lo, fine_hi + 0.1, 0.1).tolist()
    # Remove duplicates with phase 1
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
    
    # Restore best beta with a FRESH sampler (table must be rebuilt)
    model.generator.word_sampler = IntegerBoltzmannSampler(
        beta=base_beta * best_f, max_delta=50000
    )
    print(f"\nBest: f={best_f:.2f} (β={base_beta * best_f:.6f}), PPL={best_ppl:.1f}")
    
    return best_f, best_ppl


def main():
    parser = argparse.ArgumentParser(description="v12.2 Training — Extended Beta Sweep + Empirical Calibration")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help="Number of training samples (default: 500K)")
    parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB,
                        help="Vocabulary size (default: 4000)")
    parser.add_argument("--recall-scale", type=int, default=DEFAULT_RECALL_SCALE,
                        help="Recall energy scale (default: 1600)")
    parser.add_argument("--pmi-weight", type=int, default=DEFAULT_PMI_WEIGHT,
                        help="PMI coupling weight (default: 5)")
    parser.add_argument("--same-word-penalty", type=int, default=DEFAULT_SAME_WORD_PENALTY,
                        help="Same-word repetition penalty (default: 5000)")
    parser.add_argument("--no-topic-spin", action="store_true",
                        help="Disable Topic Spin coherence engine")
    parser.add_argument("--no-kn-backoff", action="store_true",
                        help="Disable Kneser-Ney backoff")
    parser.add_argument("--no-interpolated", action="store_true",
                        help="Disable interpolated n-gram smoothing")
    parser.add_argument("--ngram-min-count", type=int, default=2,
                        help="N-gram minimum count (default: 2)")
    parser.add_argument("--ngram-max-seqs", type=int, default=DEFAULT_NGRAM_MAX_SEQS,
                        help=f"Max sequences for n-gram index (default: {DEFAULT_NGRAM_MAX_SEQS:,}, 0=uncapped)")
    parser.add_argument("--max-seq-len", type=int, default=30,
                        help="Maximum sequence length for training (default: 30)")
    args = parser.parse_args()
    
    # ─── Header ─────────────────────────────────────────────
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / f"v12_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70, flush=True)
    print("ISING SPIN GLASS LANGUAGE MODEL — v12.2 TRAINING", flush=True)
    print(f"Started: {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Workers: {os.cpu_count()} (CPU count: {os.cpu_count()})", flush=True)
    rss = get_rss_mb()
    if rss > 0:
        print(f"Memory (RSS): {rss:,} MB", flush=True)
    print("=" * 70, flush=True)
    
    # ─── Load Data ──────────────────────────────────────────
    texts = load_data(args.samples)
    n_texts = len(texts)
    print(f"Using {n_texts:,} texts for training")
    rss = get_rss_mb()
    if rss > 0:
        print(f"Memory after load (RSS): {rss:,} MB")
    
    # ─── Config ─────────────────────────────────────────────
    topic_spin = not args.no_topic_spin
    kn_backoff = not args.no_kn_backoff
    interpolated = not args.no_interpolated
    
    print(f"\n{'=' * 70}")
    print(f"CONFIG: v12.2")
    print(f"  vocab_max_size={args.vocab}")
    print(f"  kn_backoff={kn_backoff}")
    print(f"  interpolated={interpolated}")
    print(f"  topic_spin={topic_spin}")
    print(f"  ngram_min_count={args.ngram_min_count}")
    print(f"  ngram_max_sequences={args.ngram_max_seqs:,}")
    print(f"  recall_scale={args.recall_scale}")
    print(f"  pmi_weight={args.pmi_weight}")
    print(f"  same_word_penalty={args.same_word_penalty}")
    print(f"  max_seq_len={args.max_seq_len}")
    print(f"  n_samples={n_texts:,}")
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
        # v12.1: N-gram index sequence cap (KEY FIX for OOM)
        ngram_max_sequences=args.ngram_max_seqs,
        # PMI
        pmi_window=5,
        pmi_min_count=2,
        pmi_cap=10,
        # Energy scales
        recall_scale=args.recall_scale,
        pmi_weight=args.pmi_weight,
        field_weight=1,
        same_word_penalty=args.same_word_penalty,
        # Beta — use auto-calibration from recall energies
        beta_type=0.001,
        beta_word=0.001,
        # Copy mechanism
        copy_enabled=True,
        copy_min_context=2,
        copy_min_confidence=0.25,
        # Ising
        ising_enabled=True,
        skip_pmi_max_dist=5,
        # v8.0: Recall-primary mode (keeps other layers as small perturbations)
        recall_primary_mode=True,
        # v8.2: Topic Spin (Potts coherence layer)
        topic_spin_enabled=topic_spin,
        topic_n_topics=16,
        topic_coherence_penalty=400,
        topic_spin_flip_interval=20,
        topic_context_window=30,
        topic_coupling_scale=100,
        # v9.0: Interpolated n-gram smoothing (product of experts)
        interpolated=interpolated,
        # v10.0: Kneser-Ney backoff (continuation counts)
        kn_backoff=kn_backoff,
        # Disabled layers (recall-primary mode)
        knowledge_scale=0,
        spin3_scale=0,
        category_scale=0,
        logic_rule_scale=0,
        # v12: Enable auto-calibrate beta
        auto_calibrate_beta=True,
        use_conceptnet=False,
        max_closed_class_run=2,
    )
    
    # Monkey-patch truncate_sequences to use configurable max_len
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
        print("Suggestions:")
        print(f"  1. Reduce --samples (currently {n_texts:,})")
        print(f"  2. Reduce --ngram-max-seqs (currently {args.ngram_max_seqs:,})")
        print(f"  3. Increase --ngram-min-count (currently {args.ngram_min_count})")
        print(f"  4. Reduce --vocab (currently {args.vocab})")
        sys.exit(1)
    except Exception as e:
        rss = get_rss_mb()
        print(f"\n!!! TRAINING ERROR: {e} !!! (RSS: {rss:,} MB)")
        traceback.print_exc()
        sys.exit(1)
    
    t_train = time.time() - t_start
    rss = get_rss_mb()
    print(f"\nTraining complete: {t_train:.1f}s ({t_train/60:.1f}min)")
    print(f"  Throughput: {n_texts / t_train:.0f} samples/sec")
    print(f"  Vocab size: {len(model.vocab)}")
    if rss > 0:
        print(f"  Peak memory (RSS): {rss:,} MB")
    
    # Restore original truncate
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
            
            # Count recalls and copies from stats
            stats = model.generator.get_stats()
            recalls = stats.get("recall_hit", 0)
            copies = stats.get("copy_used", 0)
            print(f"  recalls={recalls} copies={copies}")
            
            generated_texts.append(text)
            
            # Save generated text
            gen_file = output_dir / f"generated_{i}.txt"
            with open(gen_file, "w") as f:
                f.write(text)
        except Exception as e:
            print(f"  Generation error: {e}")
            generated_texts.append("")
    
    # ─── Save Results ───────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SAVING RESULTS")
    print(f"{'=' * 70}")
    
    results = {
        "version": "v12.2",
        "timestamp": timestamp,
        "config": {
            "vocab_max_size": args.vocab,
            "ngram_min_count": args.ngram_min_count,
            "ngram_max_sequences": args.ngram_max_seqs,
            "recall_scale": args.recall_scale,
            "pmi_weight": args.pmi_weight,
            "same_word_penalty": args.same_word_penalty,
            "kn_backoff": kn_backoff,
            "interpolated": interpolated,
            "topic_spin": topic_spin,
            "n_samples": n_texts,
            "max_seq_len": args.max_seq_len,
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
    
    # Also save to root for easy access
    root_results = CACHE_DIR / "training_results.json"
    with open(root_results, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {root_results}")
    
    t_total = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"DONE")
    print(f"Total time: {t_total:.1f}s ({t_total/60:.1f}min)")
    print(f"Results: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
