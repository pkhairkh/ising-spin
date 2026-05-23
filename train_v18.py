#!/usr/bin/env python3
"""
v18.3 Training Script — Cross-Scale RFF + Integer ESN Reservoir + Factorial State Coupling

v18.3 CHANGES (from v18.2):
  - NEW: Cross-Scale RFF (E_rff energy term)
    Joint word+POS+topic random Fourier features
    Captures cross-scale interactions that independent per-scale terms miss
  - NEW: --rff-dim (default 256), --rff-hash-dim (default 32), --rff-scale (default 600), --no-rff

v18.2 CHANGES (from v18.1):
  - NEW: Integer ESN Reservoir (E_reservoir energy term, ~50 token lookback)
    Fixed random recurrent network with exponential decay (alpha ~0.95)
    Pre-aggregated readout matrix R: (V, D) int16 via training accumulation
  - NEW: Factorial State Coupling with mean-field inference
    5 pairwise compatibility tables (topic×mode, topic×tense, mode×tense, etc.)
    Mean-field iterations refine state variables for consistency
    E_coupling energy term penalizes unlikely state combinations
  - NEW: --reservoir-dim (default 512), --reservoir-alpha (default 31130)
  - NEW: --reservoir-scale (default 800), --no-reservoir
  - NEW: --coupling-scale (default 200), --no-mf

v18.1 CHANGES (from v18.0):
  - NEW: Dense Associative Memory module (E_dense_am energy term)
    Nonlinear F(x)=x^2 energy creates sharper basins, capacity ~N instead of ~0.14N
    Random feature pre-aggregation: O(D) per candidate instead of O(N*D)
  - NEW: --dense-am-dim (default 256), --dense-am-degree (default 2), --no-dense-am
  - CHANGED: Dense AM replaces linear n-gram for word-level matching

v18.0 CHANGES (from v17.4 — PPL=19.19 best but incoherent generation):
  - NEW: VSA qFHRR binding module (E_vsa_bind energy term)
    Captures compositional word+POS+topic interactions that additive v17 cannot.
  - CHANGED: state_scale default 50 → 400 (state was <3% of total energy)
  - NEW: vsa_scale parameter (default 800)
  - NEW: --no-vsa ablation flag
  - INCREASED: Vocab from 8000 to 49000

ARCHITECTURE:
  Word n-gram (5) + POS n-gram (10) + Topic n-gram (10)
  + Dense AM (256-dim random features, degree=2) + VSA Binding (512-dim qFHRR)
  + Cross-Scale RFF (256-dim, joint word+POS+topic features)
  + ESN Reservoir (512-dim, alpha=0.95, ~50 token lookback)
  + Factorial Coupling (5 pairs, mean-field inference)
  + Document State (7 vars, scale=400)

Usage:
  python -u train_v18.py                          # Default: 500K samples
  python -u train_v18.py --samples 1000000         # 1M samples
  python -u train_v18.py --vocab 49000              # Custom vocab size
  python -u train_v18.py --no-vsa                   # Ablation: without VSA binding
  python -u train_v18.py --no-dense-am              # Ablation: without Dense AM
  python -u train_v18.py --no-rff                    # Ablation: without Cross-Scale RFF
  python -u train_v18.py --rff-dim 128                # Smaller RFF
  python -u train_v18.py --rff-scale 400               # Weaker RFF energy
  python -u train_v18.py --no-reservoir             # Ablation: without ESN reservoir
  python -u train_v18.py --no-mf                    # Ablation: without mean-field coupling
  python -u train_v18.py --reservoir-dim 256        # Smaller reservoir
  python -u train_v18.py --reservoir-alpha 31130    # Custom decay (0.95)
  python -u train_v18.py --coupling-scale 400       # Stronger coupling
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
DEFAULT_STATE_SCALE = 400   # v18.0: increased from 50
DEFAULT_VSA_SCALE = 800     # v18.0 NEW
DEFAULT_VSA_DIMENSION = 512 # v18.0 NEW
DEFAULT_DENSE_AM_SCALE = 1200  # v18.1 NEW
DEFAULT_DENSE_AM_DIM = 256     # v18.1 NEW
DEFAULT_DENSE_AM_DEGREE = 2    # v18.1 NEW
DEFAULT_DENSE_AM_HASH_DIM = 32 # v18.1 NEW
DEFAULT_RESERVOIR_SCALE = 800    # v18.2 NEW
DEFAULT_RESERVOIR_DIM = 512      # v18.2 NEW
DEFAULT_RESERVOIR_ALPHA = 31130  # v18.2 NEW (~0.95 in Q15)
DEFAULT_COUPLING_SCALE = 200     # v18.2 NEW
DEFAULT_MF_ITERATIONS = 5        # v18.2 NEW
DEFAULT_MF_LAMBDA_Q15 = 16384    # v18.2 NEW (~0.5 in Q15)
DEFAULT_RFF_SCALE = 600            # v18.3 NEW
DEFAULT_RFF_DIM = 256              # v18.3 NEW
DEFAULT_RFF_HASH_DIM = 32          # v18.3 NEW
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
        description="v18.3 Training — Cross-Scale RFF + ESN Reservoir + Factorial State Coupling"
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

    # Dense AM parameters (v18.1)
    parser.add_argument("--dense-am-scale", type=int, default=DEFAULT_DENSE_AM_SCALE,
                        help="Dense AM energy scale (default: 1200)")
    parser.add_argument("--dense-am-dim", type=int, default=DEFAULT_DENSE_AM_DIM,
                        help="Dense AM random feature dimension (default: 256)")
    parser.add_argument("--dense-am-degree", type=int, default=DEFAULT_DENSE_AM_DEGREE,
                        help="Dense AM polynomial degree: 1=linear, 2=Dense AM (default: 2)")
    parser.add_argument("--dense-am-hash-dim", type=int, default=DEFAULT_DENSE_AM_HASH_DIM,
                        help="Dense AM context hash dimension (default: 32)")

    # Reservoir parameters (v18.2 NEW)
    parser.add_argument("--reservoir-scale", type=int, default=DEFAULT_RESERVOIR_SCALE,
                        help="ESN reservoir energy scale (default: 800)")
    parser.add_argument("--reservoir-dim", type=int, default=DEFAULT_RESERVOIR_DIM,
                        help="ESN reservoir state dimension (default: 512)")
    parser.add_argument("--reservoir-alpha", type=int, default=DEFAULT_RESERVOIR_ALPHA,
                        help="ESN decay factor in Q15 (default: 31130 ≈ 0.95)")

    # Coupling parameters (v18.2 NEW)
    parser.add_argument("--coupling-scale", type=int, default=DEFAULT_COUPLING_SCALE,
                        help="Factorial state coupling energy scale (default: 200)")
    parser.add_argument("--mf-iterations", type=int, default=DEFAULT_MF_ITERATIONS,
                        help="Number of mean-field iterations (default: 5)")
    parser.add_argument("--mf-lambda-q15", type=int, default=DEFAULT_MF_LAMBDA_Q15,
                        help="Mean-field coupling strength in Q15 (default: 16384 ≈ 0.5)")

    # POS and Topic n-gram sizes
    parser.add_argument("--pos-ngram-max-n", type=int, default=DEFAULT_POS_NGRAM_MAX_N)
    parser.add_argument("--topic-ngram-max-n", type=int, default=DEFAULT_TOPIC_NGRAM_MAX_N)

    # RFF parameters (v18.3 NEW)
    parser.add_argument("--rff-scale", type=int, default=DEFAULT_RFF_SCALE,
                        help="Cross-Scale RFF energy scale (default: 600)")
    parser.add_argument("--rff-dim", type=int, default=DEFAULT_RFF_DIM,
                        help="Cross-Scale RFF feature dimension (default: 256)")
    parser.add_argument("--rff-hash-dim", type=int, default=DEFAULT_RFF_HASH_DIM,
                        help="Cross-Scale RFF context hash dimension (default: 32)")

    # Ablation flags
    parser.add_argument("--no-pos-recall", action="store_true",
                        help="Ablation: disable POS n-gram recall")
    parser.add_argument("--no-topic-recall", action="store_true",
                        help="Ablation: disable topic n-gram recall")
    parser.add_argument("--no-state", action="store_true",
                        help="Ablation: disable document state")
    parser.add_argument("--no-vsa", action="store_true",
                        help="Ablation: disable VSA binding module")
    parser.add_argument("--no-dense-am", action="store_true",
                        help="Ablation: disable Dense AM module (v18.1)")
    parser.add_argument("--no-reservoir", action="store_true",
                        help="Ablation: disable ESN reservoir (v18.2)")
    parser.add_argument("--no-mf", action="store_true",
                        help="Ablation: disable mean-field coupling inference (v18.2)")
    parser.add_argument("--no-rff", action="store_true",
                        help="Ablation: disable Cross-Scale RFF (v18.3)")

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
    dense_am_enabled = not args.no_dense_am
    reservoir_enabled = not args.no_reservoir    # v18.2
    mf_enabled = not args.no_mf                  # v18.2
    rff_enabled = not args.no_rff                # v18.3

    kn_backoff = not args.no_kn_backoff
    interpolated = not args.no_interpolated

    # --- Header ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / f"v18_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("ISING SPIN GLASS LANGUAGE MODEL — v18.3 CROSS-SCALE RFF + ESN RESERVOIR + FACTORIAL COUPLING", flush=True)
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
    print(f"CONFIG: v18.3 — Cross-Scale RFF + Integer ESN Reservoir + Factorial State Coupling")
    print(f"  WORD RECALL:")
    print(f"    ngram_max_n=5, recall_scale={args.recall_scale}")
    print(f"  POS RECALL:")
    print(f"    pos_ngram_max_n={args.pos_ngram_max_n}, pos_recall_scale={pos_recall_scale}"
          f"{' [DISABLED]' if args.no_pos_recall else ''}")
    print(f"  TOPIC RECALL:")
    print(f"    topic_ngram_max_n={args.topic_ngram_max_n}, topic_recall_scale={topic_recall_scale}"
          f"{' [DISABLED]' if args.no_topic_recall else ''}")
    print(f"  DENSE AM (v18.1):")
    print(f"    enabled={dense_am_enabled}, D={args.dense_am_dim}, degree={args.dense_am_degree}, "
          f"scale={args.dense_am_scale}, hash_dim={args.dense_am_hash_dim}"
          f"{' [DISABLED --no-dense-am]' if args.no_dense_am else ''}")
    print(f"  VSA BINDING (v18.0):")
    print(f"    enabled={vsa_enabled}, dimension={args.vsa_dimension}, vsa_scale={args.vsa_scale}"
          f"{' [DISABLED --no-vsa]' if args.no_vsa else ''}")
    print(f"  ESN RESERVOIR (v18.2 NEW):")
    print(f"    enabled={reservoir_enabled}, D={args.reservoir_dim}, alpha_q15={args.reservoir_alpha}, "
          f"scale={args.reservoir_scale}"
          f"{' [DISABLED --no-reservoir]' if args.no_reservoir else ''}")
    print(f"  FACTORIAL COUPLING (v18.2 NEW):")
    print(f"    enabled={mf_enabled}, coupling_scale={args.coupling_scale}, "
          f"mf_iterations={args.mf_iterations}, lambda_q15={args.mf_lambda_q15}"
          f"{' [DISABLED --no-mf]' if args.no_mf else ''}")
    print(f"  CROSS-SCALE RFF (v18.3 NEW):")
    print(f"    enabled={rff_enabled}, D={args.rff_dim}, hash_dim={args.rff_hash_dim}, "
          f"scale={args.rff_scale}"
          f"{' [DISABLED --no-rff]' if args.no_rff else ''}")
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
        dense_am_scale=args.dense_am_scale,
        reservoir_scale=args.reservoir_scale,      # v18.2 NEW
        coupling_scale=args.coupling_scale,         # v18.2 NEW
        # VSA
        vsa_enabled=vsa_enabled,
        vsa_dimension=args.vsa_dimension,
        vsa_seed=42,
        # Dense AM
        dense_am_enabled=dense_am_enabled,
        dense_am_dim=args.dense_am_dim,
        dense_am_degree=args.dense_am_degree,
        dense_am_seed=42,
        dense_am_hash_dim=args.dense_am_hash_dim,
        # Reservoir (v18.2 NEW)
        reservoir_enabled=reservoir_enabled,
        reservoir_dim=args.reservoir_dim,
        reservoir_alpha_q15=args.reservoir_alpha,
        reservoir_seed=42,
        # Coupling (v18.2 NEW)
        mf_enabled=mf_enabled,
        mf_iterations=args.mf_iterations,
        mf_lambda_q15=args.mf_lambda_q15,
        # RFF (v18.3 NEW)
        rff_enabled=rff_enabled,
        rff_dim=args.rff_dim,
        rff_hash_dim=args.rff_hash_dim,
        rff_seed=42,
        rff_scale=args.rff_scale,
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
        "version": "v18.3",
        "architecture": "Multi-Scale Recall + Dense AM + VSA + Cross-Scale RFF + ESN Reservoir + Factorial Coupling",
        "timestamp": timestamp,
        "config": {
            "recall_scale": args.recall_scale,
            "pos_recall_scale": pos_recall_scale,
            "topic_recall_scale": topic_recall_scale,
            "state_scale": state_scale,
            "vsa_scale": args.vsa_scale,
            "vsa_dimension": args.vsa_dimension,
            "vsa_enabled": vsa_enabled,
            "dense_am_scale": args.dense_am_scale,
            "dense_am_dim": args.dense_am_dim,
            "dense_am_degree": args.dense_am_degree,
            "dense_am_hash_dim": args.dense_am_hash_dim,
            "dense_am_enabled": dense_am_enabled,
            "reservoir_scale": args.reservoir_scale,
            "reservoir_dim": args.reservoir_dim,
            "reservoir_alpha_q15": args.reservoir_alpha,
            "reservoir_enabled": reservoir_enabled,
            "coupling_scale": args.coupling_scale,
            "mf_iterations": args.mf_iterations,
            "mf_lambda_q15": args.mf_lambda_q15,
            "mf_enabled": mf_enabled,
            "rff_scale": args.rff_scale,
            "rff_dim": args.rff_dim,
            "rff_hash_dim": args.rff_hash_dim,
            "rff_enabled": rff_enabled,
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
    print(f"DONE — v18.3 Cross-Scale RFF + Integer ESN Reservoir + Factorial State Coupling")
    print(f"Total time: {t_total:.1f}s ({t_total/60:.1f}min)")
    print(f"PPL: {full_ppl:.2f}")
    print(f"Results: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
