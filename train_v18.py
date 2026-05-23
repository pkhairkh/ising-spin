#!/usr/bin/env python3
"""
v18.0 Training Script — VSA Binding + State Scale Rebalance

v18.0 CHANGES (from v17.4 — PPL=19.19 best but incoherent generation):
  - NEW: VSA qFHRR binding module (E_vsa_bind energy term)
    Captures compositional word+POS+topic interactions that additive v17 cannot.
  - CHANGED: state_scale default 50 → 400 (state was <3% of total energy)
  - NEW: vsa_scale parameter (default 800)
  - NEW: --no-vsa ablation flag
  - INCREASED: Vocab from 8000 to 49000

ARCHITECTURE:
  Word n-gram (5) + POS n-gram (10) + Topic n-gram (10)
  + VSA Binding (512-dim qFHRR) + Document State (7 vars, scale=400)

Usage:
  python -u train_v18.py                          # Default: 500K samples
  python -u train_v18.py --samples 1000000         # 1M samples
  python -u train_v18.py --vocab 49000              # Custom vocab size
  python -u train_v18.py --no-vsa                   # Ablation: without VSA binding
  python -u train_v18.py --vsa-scale 400            # Custom VSA energy scale
  python -u train_v18.py --state-scale 400          # Custom state scale
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

# --- Configuration ---

DEFAULT_SAMPLES = 500000
DEFAULT_VOCAB = 49000       # v18: increased from 8000
DEFAULT_MIN_FREQ = 15
DEFAULT_RECALL_SCALE = 1600
DEFAULT_POS_RECALL_SCALE = 800
DEFAULT_TOPIC_RECALL_SCALE = 400
DEFAULT_STATE_SCALE = 400   # v18: increased from 50
DEFAULT_VSA_SCALE = 800     # v18 NEW
DEFAULT_VSA_DIMENSION = 512 # v18 NEW
DEFAULT_POS_NGRAM_MAX_N = 10
DEFAULT_TOPIC_NGRAM_MAX_N = 10

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
    from ising_spin.model_v18 import _load_fineweb_edu
    t0 = time.time()
    texts = _load_fineweb_edu(n_samples=n_samples)
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
    from ising_spin.sampling import IntegerBoltzmannSampler, LN2_NUM, LN2_DEN

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
        description="v18.0 Training — VSA Binding + State Scale Rebalance"
    )

    # Core parameters
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
    parser.add_argument("--recall-scale", type=int, default=DEFAULT_RECALL_SCALE,
                        help="Word n-gram recall energy scale (default: 1600)")
    parser.add_argument("--pos-recall-scale", type=int, default=DEFAULT_POS_RECALL_SCALE,
                        help="POS n-gram recall energy scale (default: 800)")
    parser.add_argument("--topic-recall-scale", type=int, default=DEFAULT_TOPIC_RECALL_SCALE,
                        help="Topic n-gram recall energy scale (default: 400)")
    parser.add_argument("--state-scale", type=int, default=DEFAULT_STATE_SCALE,
                        help="Document state energy scale (default: 400, was 50 in v17)")
    parser.add_argument("--vsa-scale", type=int, default=DEFAULT_VSA_SCALE,
                        help="VSA binding energy scale (default: 800)")
    parser.add_argument("--vsa-dimension", type=int, default=DEFAULT_VSA_DIMENSION,
                        help="VSA vector dimension (default: 512)")

    # POS and Topic n-gram sizes
    parser.add_argument("--pos-ngram-max-n", type=int, default=DEFAULT_POS_NGRAM_MAX_N)
    parser.add_argument("--topic-ngram-max-n", type=int, default=DEFAULT_TOPIC_NGRAM_MAX_N)

    # Ablation flags
    parser.add_argument("--no-pos-recall", action="store_true",
                        help="Ablation: disable POS n-gram recall")
    parser.add_argument("--no-topic-recall", action="store_true",
                        help="Ablation: disable topic n-gram recall")
    parser.add_argument("--no-state", action="store_true",
                        help="Ablation: disable document state")
    parser.add_argument("--no-vsa", action="store_true",
                        help="Ablation: disable VSA binding module")

    # Standard n-gram parameters
    parser.add_argument("--no-kn-backoff", action="store_true")
    parser.add_argument("--no-interpolated", action="store_true")
    parser.add_argument("--ngram-min-count", type=int, default=2)
    parser.add_argument("--ngram-max-seqs", type=int, default=1000000)
    parser.add_argument("--max-seq-len", type=int, default=30)
    parser.add_argument("--same-word-penalty", type=int, default=200)
    parser.add_argument("--n-topics", type=int, default=16)

    args = parser.parse_args()

    # --- Apply ablation flags ---
    pos_recall_scale = 0 if args.no_pos_recall else args.pos_recall_scale
    topic_recall_scale = 0 if args.no_topic_recall else args.topic_recall_scale
    state_scale = 0 if args.no_state else args.state_scale
    vsa_enabled = not args.no_vsa

    kn_backoff = not args.no_kn_backoff
    interpolated = not args.no_interpolated

    # --- Header ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / f"v18_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("ISING SPIN GLASS LANGUAGE MODEL — v18.0 VSA BINDING", flush=True)
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

    # --- Config ---
    print(f"\n{'=' * 70}")
    print(f"CONFIG: v18.0 — VSA Binding + State Scale Rebalance")
    print(f"  WORD RECALL:")
    print(f"    ngram_max_n=5, recall_scale={args.recall_scale}")
    print(f"  POS RECALL:")
    print(f"    pos_ngram_max_n={args.pos_ngram_max_n}, pos_recall_scale={pos_recall_scale}"
          f"{' [DISABLED]' if args.no_pos_recall else ''}")
    print(f"  TOPIC RECALL:")
    print(f"    topic_ngram_max_n={args.topic_ngram_max_n}, topic_recall_scale={topic_recall_scale}"
          f"{' [DISABLED]' if args.no_topic_recall else ''}")
    print(f"  VSA BINDING (v18 NEW):")
    print(f"    enabled={vsa_enabled}, dimension={args.vsa_dimension}, vsa_scale={args.vsa_scale}"
          f"{' [DISABLED --no-vsa]' if args.no_vsa else ''}")
    print(f"  DOCUMENT STATE:")
    print(f"    state_scale={state_scale} (was 50 in v17)"
          f"{' [DISABLED]' if args.no_state else ''}")
    print(f"  STANDARD:")
    print(f"    vocab_max_size={args.vocab}")
    print(f"    kn_backoff={kn_backoff}")
    print(f"    interpolated={interpolated}")
    print(f"    n_topics={args.n_topics}")
    print(f"    n_samples={n_texts:,}")
    print(f"{'=' * 70}")

    # --- Train ---
    from ising_spin.model_v18 import IsingLMModelV18

    model = IsingLMModelV18(
        # Vocabulary
        vocab_min_freq=15,
        vocab_max_size=args.vocab,
        # Word N-gram
        ngram_max_n=5,
        ngram_min_count=args.ngram_min_count,
        ngram_max_sequences=args.ngram_max_seqs,
        # POS N-gram
        pos_ngram_max_n=args.pos_ngram_max_n,
        pos_ngram_min_count=2,
        # Topic
        n_topics=args.n_topics,
        topic_ngram_max_n=args.topic_ngram_max_n,
        topic_ngram_min_count=3,
        # Energy scales
        recall_scale=args.recall_scale,
        pos_recall_scale=pos_recall_scale,
        topic_recall_scale=topic_recall_scale,
        state_scale=state_scale,
        vsa_scale=args.vsa_scale,
        # VSA
        vsa_enabled=vsa_enabled,
        vsa_dimension=args.vsa_dimension,
        vsa_seed=42,
        # Hard constraints
        same_word_penalty=args.same_word_penalty,
        max_closed_class_run=2,
        # Beta
        auto_calibrate_beta=True,
        # Interpolation
        interpolated=interpolated,
        kn_backoff=kn_backoff,
        # Copy mechanism
        copy_enabled=True,
        copy_min_context=3,
        copy_min_confidence=0.4,
        # Misc
        max_seq_len=args.max_seq_len,
    )

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

    print(f"\nPPL (full, 100 seqs): {full_ppl:.2f}")

    # --- Generation ---
    print(f"\n{'=' * 70}")
    print(f"GENERATION (PPL={full_ppl:.2f})")
    print(f"{'=' * 70}")

    prompts = ["the history of", "science and technology", "research shows that"]

    for i, prompt in enumerate(prompts):
        print(f"\n  --- '{prompt}' (100 words) ---")
        try:
            result = model.generator.generate(prompt=prompt, length=100)
            text = result.get("text", str(result))
            if isinstance(text, list):
                text = " ".join(text)
            print(f"  {text[:300]}...")
        except Exception as e:
            print(f"  Generation error: {e}")
            traceback.print_exc()

    # --- Save Results ---
    results = {
        "version": "v18.0",
        "architecture": "Multi-Scale Recall + VSA Binding + Document State",
        "timestamp": timestamp,
        "config": {
            "recall_scale": args.recall_scale,
            "pos_recall_scale": pos_recall_scale,
            "topic_recall_scale": topic_recall_scale,
            "state_scale": state_scale,
            "vsa_scale": args.vsa_scale,
            "vsa_dimension": args.vsa_dimension,
            "vsa_enabled": vsa_enabled,
            "vocab_max_size": args.vocab,
            "kn_backoff": kn_backoff,
            "interpolated": interpolated,
            "n_topics": args.n_topics,
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

    t_total = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"DONE — v18.0 VSA Binding + State Scale Rebalance")
    print(f"Total time: {t_total:.1f}s ({t_total/60:.1f}min)")
    print(f"PPL: {full_ppl:.2f}")
    print(f"Results: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
