#!/usr/bin/env python3
"""
Ising Spin Glass Language Model — unified training script.

No more version-in-filename slugs. All features are controlled
by CLI flags, mirroring the ModelConfig dataclass.

Usage:
  python -u train.py                              # Default: 500K samples
  python -u train.py --samples 1000000              # 1M samples
  python -u train.py --vocab 49000                   # Custom vocab size
  python -u train.py --no-vsa                        # Ablation: disable VSA
  python -u train.py --no-dense-am                   # Ablation: disable Dense AM
  python -u train.py --no-rff                        # Ablation: disable RFF
  python -u train.py --no-reservoir                  # Ablation: disable ESN
  python -u train.py --no-mf                         # Ablation: disable coupling
  python -u train.py --dense-am-degree 1             # Linear (no Dense AM)
"""

# --- UNBUFFERED OUTPUT ---
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
import time
import traceback
from pathlib import Path

from ising_spin.model import IsingLMModel, ModelConfig
from ising_spin.helpers import get_rss_mb
from ising_spin.sampling import IntegerBoltzmannSampler, LN2_NUM, LN2_DEN

# --- Paths ---
CACHE_DIR = Path(__file__).parent
OUTPUT_DIR = CACHE_DIR / "output"


def find_cache_file(n_samples: int) -> str | None:
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
        return sufficient[min(sufficient.keys())]
    return None


def load_data(n_samples: int) -> list:
    """Load or download FineWeb-Edu data with cache."""
    from ising_spin.helpers import load_fineweb_edu

    cache_path = find_cache_file(n_samples)
    if cache_path:
        print(f"Loading cache: {cache_path}")
        t0 = time.time()
        with open(cache_path, "r") as f:
            texts = json.load(f)
        print(f"  {len(texts):,} texts loaded in {time.time()-t0:.1f}s")
        if len(texts) >= n_samples:
            return texts[:n_samples]
        print(f"  Cache has {len(texts):,} texts but need {n_samples:,}")

    print("No cached data found. Downloading from HuggingFace...")
    texts = load_fineweb_edu(n_samples=n_samples)

    # Save cache
    if n_samples >= 1_000_000 and n_samples % 1_000_000 == 0:
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

    recall_scale = model.config.recall_scale
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
            print(f"    f={f:.2f} (beta={new_beta:.6f}): PPL={ppl:.1f}{marker}")
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
                print(f"    f={f:.2f} (beta={new_beta:.6f}): PPL={ppl:.1f}{marker}")
                if ppl < best_ppl:
                    best_ppl = ppl
                    best_f = f
            except Exception as e:
                print(f"    f={f:.2f}: Error: {e}")

    model.generator.word_sampler = IntegerBoltzmannSampler(
        beta=base_beta * best_f, max_delta=50000
    )
    print(f"\nBest: f={best_f:.2f} (beta={base_beta * best_f:.6f}), PPL={best_ppl:.1f}")
    return best_f, best_ppl


def main():
    parser = argparse.ArgumentParser(
        description="Ising Spin Glass LM — unified training script"
    )

    # Core parameters
    parser.add_argument("--samples", type=int, default=500000)
    parser.add_argument("--vocab", type=int, default=49000)
    parser.add_argument("--recall-scale", type=int, default=1600)
    parser.add_argument("--pos-recall-scale", type=int, default=800)
    parser.add_argument("--topic-recall-scale", type=int, default=400)
    parser.add_argument("--state-scale", type=int, default=400)
    parser.add_argument("--vsa-scale", type=int, default=800)
    parser.add_argument("--vsa-dimension", type=int, default=512)

    # Dense AM
    parser.add_argument("--dense-am-scale", type=int, default=1200)
    parser.add_argument("--dense-am-dim", type=int, default=256)
    parser.add_argument("--dense-am-degree", type=int, default=2, choices=[1, 2])
    parser.add_argument("--dense-am-hash-dim", type=int, default=32)

    # Reservoir
    parser.add_argument("--reservoir-scale", type=int, default=800)
    parser.add_argument("--reservoir-dim", type=int, default=512)
    parser.add_argument("--reservoir-alpha", type=int, default=31130)

    # Coupling
    parser.add_argument("--coupling-scale", type=int, default=200)
    parser.add_argument("--mf-iterations", type=int, default=5)
    parser.add_argument("--mf-lambda-q15", type=int, default=16384)

    # RFF
    parser.add_argument("--rff-scale", type=int, default=600)
    parser.add_argument("--rff-dim", type=int, default=256)
    parser.add_argument("--rff-hash-dim", type=int, default=32)

    # Standard n-gram
    parser.add_argument("--pos-ngram-max-n", type=int, default=10)
    parser.add_argument("--topic-ngram-max-n", type=int, default=10)
    parser.add_argument("--ngram-min-count", type=int, default=2)
    parser.add_argument("--ngram-max-seqs", type=int, default=1000000)
    parser.add_argument("--max-seq-len", type=int, default=30)
    parser.add_argument("--same-word-penalty", type=int, default=200)
    parser.add_argument("--n-topics", type=int, default=16)

    # Ablation flags
    parser.add_argument("--no-pos-recall", action="store_true")
    parser.add_argument("--no-topic-recall", action="store_true")
    parser.add_argument("--no-state", action="store_true")
    parser.add_argument("--no-vsa", action="store_true")
    parser.add_argument("--no-dense-am", action="store_true")
    parser.add_argument("--no-reservoir", action="store_true")
    parser.add_argument("--no-mf", action="store_true")
    parser.add_argument("--no-rff", action="store_true")
    parser.add_argument("--no-kn-backoff", action="store_true")
    parser.add_argument("--no-interpolated", action="store_true")

    args = parser.parse_args()

    # --- Build config from args ---
    config = ModelConfig(
        vocab_max_size=args.vocab,
        recall_scale=args.recall_scale,
        pos_recall_scale=0 if args.no_pos_recall else args.pos_recall_scale,
        topic_recall_scale=0 if args.no_topic_recall else args.topic_recall_scale,
        state_scale=0 if args.no_state else args.state_scale,
        vsa_scale=args.vsa_scale,
        vsa_dimension=args.vsa_dimension,
        vsa_enabled=not args.no_vsa,
        dense_am_scale=args.dense_am_scale,
        dense_am_dim=args.dense_am_dim,
        dense_am_degree=args.dense_am_degree,
        dense_am_hash_dim=args.dense_am_hash_dim,
        dense_am_enabled=not args.no_dense_am,
        reservoir_scale=args.reservoir_scale,
        reservoir_dim=args.reservoir_dim,
        reservoir_alpha_q15=args.reservoir_alpha,
        reservoir_enabled=not args.no_reservoir,
        coupling_scale=args.coupling_scale,
        mf_iterations=args.mf_iterations,
        mf_lambda_q15=args.mf_lambda_q15,
        mf_enabled=not args.no_mf,
        rff_scale=args.rff_scale,
        rff_dim=args.rff_dim,
        rff_hash_dim=args.rff_hash_dim,
        rff_enabled=not args.no_rff,
        same_word_penalty=args.same_word_penalty,
        pos_ngram_max_n=args.pos_ngram_max_n,
        topic_ngram_max_n=args.topic_ngram_max_n,
        ngram_min_count=args.ngram_min_count,
        ngram_max_sequences=args.ngram_max_seqs,
        max_seq_len=args.max_seq_len,
        n_topics=args.n_topics,
        interpolated=not args.no_interpolated,
        kn_backoff=not args.no_kn_backoff,
    )

    # --- Header ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("ISING SPIN GLASS LANGUAGE MODEL — unified training", flush=True)
    print(f"Started: {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    rss = get_rss_mb()
    if rss > 0:
        print(f"Memory (RSS): {rss:,} MB", flush=True)
    print("=" * 70, flush=True)

    # --- Load Data ---
    texts = load_data(args.samples)
    n_texts = len(texts)
    print(f"Using {n_texts:,} texts for training")

    # --- Train ---
    model = IsingLMModel(config=config)
    t_start = time.time()

    try:
        model.train(n_samples=n_texts, texts=texts)
    except MemoryError:
        print(f"\n!!! OUT OF MEMORY !!! (RSS: {get_rss_mb():,} MB)")
        sys.exit(1)
    except Exception as e:
        print(f"\n!!! TRAINING ERROR: {e} !!! (RSS: {get_rss_mb():,} MB)")
        traceback.print_exc()
        sys.exit(1)

    t_train = time.time() - t_start
    rss = get_rss_mb()
    print(f"\nTraining complete: {t_train:.1f}s ({t_train/60:.1f}min)")
    print(f"  Vocab size: {len(model.vocab)}")
    if rss > 0:
        print(f"  Peak memory (RSS): {rss:,} MB")

    # --- Evaluation ---
    print(f"\n{'=' * 70}")
    print("EVALUATION")
    print(f"{'=' * 70}")

    # Quick PPL
    print("\nQuick PPL (default beta, 10 seqs):", end=" ")
    try:
        quick_ppl = model.compute_perplexity(n_samples=10)
        print(f"{quick_ppl:.1f}")
    except Exception as e:
        print(f"Error: {e}")
        quick_ppl = 999

    # Beta sweep
    best_f, best_sweep_ppl = beta_sweep_ppl(model, n_seqs=10)

    # Full PPL
    print(f"\nFull PPL evaluation...")
    try:
        full_ppl = model.compute_perplexity(n_samples=100)
        print(f"  Perplexity: {full_ppl:.2f}")
    except Exception as e:
        print(f"  Error: {e}")
        full_ppl = best_sweep_ppl

    # --- Generation ---
    print(f"\n{'=' * 70}")
    print(f"GENERATION (PPL={full_ppl:.2f})")
    print(f"{'=' * 70}")

    prompts = ["the history of", "science and technology", "research shows that"]
    for prompt in prompts:
        print(f"\n  --- '{prompt}' (100 words) ---")
        try:
            result = model.generator.generate(prompt=prompt, length=100)
            text = result.get("text", str(result))
            if isinstance(text, list):
                text = " ".join(text)
            print(f"  {text[:300]}...")
        except Exception as e:
            print(f"  Generation error: {e}")

    # --- Save Results ---
    results = {
        "architecture": "unified Ising Spin Glass LM",
        "timestamp": timestamp,
        "config": {
            k: v for k, v in config.__dict__.items()
            if not k.startswith("_")
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

    t_total = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"DONE — Total time: {t_total:.1f}s ({t_total/60:.1f}min), PPL: {full_ppl:.2f}")
    print(f"Results: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
