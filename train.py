#!/usr/bin/env python3
"""
Attractor Language Machine v52 — Training Script

v52: POSITIONAL VSA CONTEXT — fix the order-blind BOW encoding
  - v50 had PPL=16.65 (great!) but generation was word salad.
  - v51 added POS skeleton (13 types) — too coarse, PPL regressed to 27.73.
  - ROOT CAUSE: BOW context encoding destroys word order. The DAM knows
    "what words appear together" but not "in what order". During generation,
    each wrong word pollutes the BOW, causing error cascade.
  - v52 FIX: Replace BOW with positional VSA context encoding.
    Each word's SDR is rotated by its relative position hash before
    superposition. "the cat" and "cat the" now produce DIFFERENT context
    SDRs. The DAM can learn order-dependent patterns natively.
  - Key difference from v45 (which failed, PPL 221→1909):
    v45 used content-dependent hash (bind(word, hash(prev_word))) →
    same BOW context produced different SDRs → DAM couldn't handle it.
    v52 uses position-dependent hash (rotate(word, hash(rel_pos))) →
    same ordered context always produces the same SDR → consistent training.
  - Also: restore bigram_weight=16, skip_weight=5, remove POS skeleton,
    reduce bind_weight to 15 (positional VSA handles order in DAM now),
    restore top-k=10.

Architecture: D=512, 50K samples, Hebbian L0

Usage:
  python -u train.py                                     # Default: 50K samples
  python -u train.py --samples 100000                    # 100K samples
  python -u train.py --memory-budget 14000               # Pi 5 (16GB)
  python -u train.py --same-word-penalty 1200          # Stronger repetition suppression
  python -u train.py --bigram-weight 8                   # Lighter bigram
  python -u train.py --skip-weight 0                     # Disable skip bigram
  python -u train.py --pos-weight 0                      # Disable POS skeleton
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

DEFAULT_SAMPLES = 50000
DEFAULT_VOCAB = 2000
DEFAULT_DATASET = "tinystories"
DEFAULT_MEMORY_BUDGET = 0
DEFAULT_SDR_DIM = 512
DEFAULT_SDR_SPARSITY = 0.02

# Cache directory
CACHE_DIR = Path(__file__).parent
OUTPUT_DIR = CACHE_DIR / "output"


def get_rss_mb() -> int:
    """Get current process RSS in MB."""
    from ising_spin.utils import get_rss_mb as _get_rss_mb
    return _get_rss_mb()


def find_cache_file(n_samples: int, dataset_name: str = "tinystories") -> str:
    """Find the best cache file for the requested number of samples."""
    cache_files = {}
    glob_pattern = f"cached_{dataset_name}_*.json"
    for f in CACHE_DIR.glob(glob_pattern):
        name = f.stem
        parts = name.split("_")
        size_str = parts[-1]
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


def load_data(n_samples: int, dataset_name: str = DEFAULT_DATASET) -> list:
    """Load or download data with proper cache handling."""
    from ising_spin.utils import DATASET_LOADERS, DEFAULT_DATASET as _DEF

    dataset_name = dataset_name or _DEF
    if dataset_name not in DATASET_LOADERS:
        print(f"  Unknown dataset '{dataset_name}', falling back to {_DEF}")
        dataset_name = _DEF

    cache_dataset = dataset_name.replace("-", "_")
    cache_path = find_cache_file(n_samples, cache_dataset)

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

    print(f"No cached data found. Downloading {dataset_name} from HuggingFace...")
    loader = DATASET_LOADERS[dataset_name]
    t0 = time.time()
    texts = loader(n_samples=n_samples)
    print(f"  Downloaded {len(texts):,} texts in {time.time()-t0:.1f}s")

    if n_samples >= 1000000 and n_samples % 1000000 == 0:
        cache_name = f"cached_{cache_dataset}_{n_samples // 1000000}m.json"
    elif n_samples >= 1000 and n_samples % 1000 == 0:
        cache_name = f"cached_{cache_dataset}_{n_samples // 1000}k.json"
    else:
        cache_name = f"cached_{cache_dataset}_{n_samples}.json"

    cache_file = CACHE_DIR / cache_name
    print(f"  Saving cache to: {cache_file}")
    with open(cache_file, "w") as f:
        json.dump(texts, f)

    return texts


def main():
    parser = argparse.ArgumentParser(
        description="Attractor Language Machine v52 — POSITIONAL VSA"
    )

    # Core parameters
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help="Number of training samples (default: 50K)")
    parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB,
                        help="Max vocabulary size (default: 2000)")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        choices=["tinystories", "tiny-textbooks", "writingprompts", "fineweb-edu"],
                        help=f"Dataset (default: {DEFAULT_DATASET})")

    # SDR parameters
    parser.add_argument("--sdr-dim", type=int, default=DEFAULT_SDR_DIM,
                        help="SDR dimension (default: 512)")
    parser.add_argument("--sdr-sparsity", type=float, default=DEFAULT_SDR_SPARSITY,
                        help="SDR sparsity (default: 0.02 = 2%%)")

    # Energy scales
    parser.add_argument("--dam-scale", type=int, default=1600,
                        help="DAM energy scale (default: 1600)")
    parser.add_argument("--episodic-scale", type=int, default=100,
                        help="Episodic memory energy scale (default: 100)")
    parser.add_argument("--same-word-penalty", type=int, default=800,
                        help="Same-word repetition penalty (default: 800)")

    # F function parameters
    parser.add_argument("--f-type", type=str, default="exp_approx",
                        choices=["quadratic", "cubic", "exp_approx"],
                        help="F function type (default: exp_approx)")
    parser.add_argument("--exp-temperature", type=int, default=100,
                        help="Exponential F temperature in Q8 (100=1.0, 50=0.5 sharper, default: 100)")

    # UV-complete parameters
    parser.add_argument("--uv-regularize", action="store_true", default=True,
                        help="Enable UV-complete regularization (default: True)")
    parser.add_argument("--no-uv-regularize", action="store_true",
                        help="Disable UV-complete regularization")
    parser.add_argument("--uv-lambda", type=int, default=5,
                        help="UV regularization strength (default: 5)")
    parser.add_argument("--topdown-scale", type=int, default=200,
                        help="Top-down feedback scale (default: 200)")

    # Coupling parameters
    parser.add_argument("--j-clip", type=int, default=2000,
                        help="Coupling matrix clip value (default: 2000)")

    # Episodic memory
    parser.add_argument("--max-episodes", type=int, default=10000,
                        help="Max episodic memory episodes (default: 10000)")

    # Generation
    parser.add_argument("--max-seq-len", type=int, default=30,
                        help="Max sequence length (default: 30)")
    parser.add_argument("--vocab-min-freq", type=int, default=5,
                        help="Min word frequency for vocab (default: 5)")

    # VSA Binding parameters (v39)
    parser.add_argument("--bind-window", type=int, default=8,
                        help="Binding context window size (default: 8)")
    parser.add_argument("--bind-weight", type=int, default=15,
                        help="Binding energy weight (default: 15, v52 reduced — positional VSA handles order in DAM)")
    parser.add_argument("--n-unbind-words", type=int, default=3,
                        help="Number of recent words for multi-step unbinding (default: 3)")
    parser.add_argument("--bind-density", type=int, default=0,
                        help="M_bind target density in bits (default: 0=auto, i.e. 2*k=20, v44 value)")

    # Bigram DAM (v52: strong bigram + positional VSA context)
    parser.add_argument("--bigram-weight", type=int, default=16,
                        help="Bigram coupling weight (default: 16 for log-normalized, 0=disabled)")
    # Skip bigram (v52: increased weight)
    parser.add_argument("--skip-weight", type=int, default=5,
                        help="Skip bigram weight for J2[words[-2],c] (default: 5, 0=disabled)")
    # POS skeleton (v52: disabled — positional VSA replaces it)
    parser.add_argument("--pos-weight", type=int, default=0,
                        help="POS trigram skeleton weight (default: 0=disabled, positional VSA replaces it)")

    # Memory budget
    parser.add_argument("--memory-budget", type=int, default=DEFAULT_MEMORY_BUDGET,
                        help="Memory budget in MB (0=unlimited; 14000=16GB Pi)")

    args = parser.parse_args()

    # Parse F type
    f_type_map = {"quadratic": 0, "cubic": 1, "exp_approx": 2}
    f_type = f_type_map[args.f_type]

    # --- Header ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / f"attractor_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("ATTRACTOR LANGUAGE MACHINE v52 — POSITIONAL VSA", flush=True)
    print(f"Started: {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    rss = get_rss_mb()
    if rss > 0:
        print(f"Memory (RSS): {rss:,} MB", flush=True)
    print("=" * 70, flush=True)

    # --- Load Data ---
    texts = load_data(args.samples, dataset_name=args.dataset)
    n_texts = len(texts)
    print(f"Using {n_texts:,} texts for training")

    # --- Config ---
    uv_regularize = args.uv_regularize and not args.no_uv_regularize

    print(f"\n{'=' * 70}")
    print(f"CONFIG: Attractor Language Machine v52 (POSITIONAL VSA)")
    print(f"  ARCHITECTURE:")
    print(f"    SDR: D={args.sdr_dim}, sparsity={args.sdr_sparsity} ({int(args.sdr_dim * args.sdr_sparsity)} active bits)")
    print(f"    Hierarchy: L0(512)->L1(256)->L2(128)->L3(64)")
    print(f"    RG flow: J_eff[l] decimated, Kadanoff rescaling (v34 fix preserved)")
    print(f"    F function: INLINE piecewise exp (NO J_MAX clip)")
    print(f"    Energy: NORMALIZED log2-F (LOG2_NORM=512, NO k div, NO h, dE ~ O(200-300))")
    print(f"  BINDING (v52: VSA secondary — positional VSA + J2 are primary):")
    print(f"    Type: VSA permutation — bind(a,hash(b)), unbind=rot(D-hash(b))")
    print(f"    Hash: sum(active_bits) mod D (full [0,D-1] spread)")
    print(f"    Window: {args.bind_window} recent bigram bindings")
    print(f"    Weight: {args.bind_weight}")
    print(f"    N_unbind: {args.n_unbind_words} (multi-step unbinding)")
    print(f"    M_bind density: {args.bind_density if args.bind_density > 0 else 'auto=20'} bits")
    print(f"    M_bind: attractor dynamics ONLY (not DAM energy — v45 reverted)")
    print(f"    Recency: NONE (uniform — recency reverted, hurt PPL)")
    print(f"  BIGRAM DAM (v52: STRONG bigram + positional VSA context):")
    print(f"    J2: V×V int32 matrix of log2(count+1) values")
    print(f"    Weight: {args.bigram_weight}{' (DISABLED)' if args.bigram_weight == 0 else ''}")
    print(f"    Skip bigram: J2[words[-2],c] weight={args.skip_weight}{' (DISABLED)' if args.skip_weight == 0 else ''}")
    print(f"    Energy: E_bigram(c) = -J2[prev_word, c] * weight")
    print(f"    Range: [0, ~16] × weight → max ~{16 * args.bigram_weight}")
    print(f"    Memory: ~{args.vocab * args.vocab * 4 / 1024 / 1024:.0f} MB")
    print(f"  POS SKELETON (v52: DISABLED — positional VSA replaces it):")
    print(f"    J_pos_bi: 13×13 POS bigram transitions, log2(count+1)")
    print(f"    J_pos_tri: 13×13×13 POS trigram transitions, log2(count+1)")
    print(f"    Weight: {args.pos_weight}{' (DISABLED)' if args.pos_weight == 0 else ''}")
    print(f"    Backoff: trigram → bigram (75% weight) when trigram count=0")
    print(f"    Memory: ~10 KB (trivial)")
    print(f"  F FUNCTION:")
    print(f"    Type: {args.f_type}")
    if f_type == 2:
        print(f"    Temperature: {args.exp_temperature/100:.2f} (Q8: {args.exp_temperature})")
    print(f"  ENERGY SCALES:")
    print(f"    DAM scale={args.dam_scale}")
    print(f"    Episodic scale={args.episodic_scale}")
    print(f"    Same-word penalty={args.same_word_penalty} (generation only, not PPL)")
    print(f"    Generation: top-k=10 + Boltzmann (v52)")
    print(f"    Repetition window=15, distance-decay (v40 fix)")
    print(f"    Grammar penalty scaled to ~33% median_dE (v40 fix)")
    print(f"    Special tokens (idx<4) filtered from candidates (v40 fix)")
    print(f"  UV-COMPLETE:")
    print(f"    Regularize={uv_regularize}, lambda={args.uv_lambda}")
    print(f"    Top-down scale={args.topdown_scale}")
    print(f"    Ward identity checks: ENABLED")
    print(f"  COUPLING:")
    print(f"    J_clip={args.j_clip}")
    print(f"    Learning: Hebbian (L0 only, RG flow to higher levels)")
    print(f"  EPISODIC:")
    print(f"    Max episodes={args.max_episodes}")
    print(f"  DATA:")
    print(f"    Dataset={args.dataset}, samples={n_texts:,}")
    print(f"    Vocab max={args.vocab}, min_freq={args.vocab_min_freq}")
    print(f"    Max seq len={args.max_seq_len}")
    if args.memory_budget > 0:
        print(f"    Memory budget={args.memory_budget:,} MB")
    print(f"{'=' * 70}")

    # --- Train ---
    from ising_spin.attractor import AttractorLanguageModel

    model = AttractorLanguageModel(
        vocab_min_freq=args.vocab_min_freq,
        vocab_max_size=args.vocab,
        sdr_dim=args.sdr_dim,
        sdr_sparsity=args.sdr_sparsity,
        dam_scale=args.dam_scale,
        episodic_scale=args.episodic_scale,
        same_word_penalty=args.same_word_penalty,
        uv_regularize=uv_regularize,
        uv_lambda=args.uv_lambda,
        topdown_scale=args.topdown_scale,
        j_clip=args.j_clip,
        max_episodes=args.max_episodes,
        max_seq_len=args.max_seq_len,
        memory_budget_mb=args.memory_budget,
        f_type=f_type,
        exp_temperature=args.exp_temperature,
        bind_window=args.bind_window,
        bind_weight=args.bind_weight,
        n_unbind_words=args.n_unbind_words,
        bind_density=args.bind_density,
        bigram_weight=args.bigram_weight,
        skip_weight=args.skip_weight,
        pos_weight=args.pos_weight,
        seed=42,
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
    print("\nQuick PPL (10 seqs):", end=" ")
    try:
        quick_ppl = model.compute_perplexity(n_samples=10)
        print(f"{quick_ppl:.1f}")
    except Exception as e:
        print(f"Error: {e}")
        quick_ppl = 999

    # Full PPL
    print(f"\nFull PPL evaluation (100 seqs)...")
    try:
        full_ppl = model.compute_perplexity(n_samples=100)
        print(f"  Perplexity: {full_ppl:.2f}")
    except Exception as e:
        print(f"  Error: {e}")
        full_ppl = quick_ppl

    # --- Generation ---
    print(f"\n{'=' * 70}")
    print(f"GENERATION (PPL={full_ppl:.2f})")
    print(f"{'=' * 70}")

    prompts = ["once upon a time", "there was a little", "the little girl"]
    if args.dataset != "tinystories":
        prompts = ["the history of", "science and technology", "research shows that"]

    generated_texts = []
    for i, prompt in enumerate(prompts):
        print(f"\n  --- '{prompt}' (200 words) ---")
        try:
            result = model.generate(prompt=prompt, length=200)
            text = result.get("text", str(result))
            if isinstance(text, list):
                text = " ".join(text)
            print(f"  {text[:300]}...")
            generated_texts.append(text)

            gen_file = output_dir / f"generated_{i}.txt"
            with open(gen_file, "w") as f:
                f.write(text)
        except Exception as e:
            print(f"  Generation error: {e}")
            traceback.print_exc()
            generated_texts.append("")

    # --- Save Results ---
    results = {
        "version": "52.0.0",
        "architecture": "Attractor Language Machine v52 — POSITIONAL VSA (v52: positional VSA context encoding — rotate(word_sdr, pos_hash(rel_pos)) before superposition, preserving word ORDER in context SDR; bigram J2 weight={bw}, skip weight={sw}, POS weight={pw} (disabled), bind weight={bind_w}, top-k=10)".format(bw=args.bigram_weight, sw=args.skip_weight, pw=args.pos_weight, bind_w=args.bind_weight),
        "dataset": args.dataset,
        "timestamp": timestamp,
        "config": {
            "sdr_dim": args.sdr_dim,
            "sdr_sparsity": args.sdr_sparsity,
            "dam_scale": args.dam_scale,
            "episodic_scale": args.episodic_scale,
            "uv_regularize": uv_regularize,
            "uv_lambda": args.uv_lambda,
            "topdown_scale": args.topdown_scale,
            "j_clip": args.j_clip,
            "max_episodes": args.max_episodes,
            "vocab_max_size": args.vocab,
            "same_word_penalty": args.same_word_penalty,
            "bind_window": args.bind_window,
            "bind_weight": args.bind_weight,
            "n_unbind_words": args.n_unbind_words,
            "bind_density": args.bind_density,
            "bigram_weight": args.bigram_weight,
            "skip_weight": args.skip_weight,
            "pos_weight": args.pos_weight,
            "f_type": args.f_type,
            "exp_temperature": args.exp_temperature,
        },
        "results": {
            "training_time_sec": t_train,
            "quick_ppl": quick_ppl,
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
    print(f"DONE — Attractor Language Machine v52")
    print(f"Total time: {t_total:.1f}s ({t_total/60:.1f}min)")
    print(f"PPL: {full_ppl:.2f}")
    print(f"Results: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
