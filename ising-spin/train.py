#!/usr/bin/env python3
"""
Integer Language Model — Training Script (v81 — Data-Driven Classes)

Pure integer language model. No neural nets. No torch dependency.
Runs on a Pi 5. Produces grammatically coherent text.

KEY CHANGE v81: ALL POS dependency removed from features.
  - Word classes are DATA-DRIVEN (frequency buckets, K=20)
  - NOT static POS tags (K=13, 88% NOUN = degenerate)
  - Balanced classes: ~100 words/bucket instead of 1762 in one tag
  - hash(word, class) has 40000 keys vs hash(word, pos) ≈ 2000

Usage:
  python -u train.py                           # Full run (50K texts, default features)
  python -u train.py --samples 5000 --vocab 1000  # Quick test
  python -u train.py --features lex_bi,word_cls_bi,cls_word_bi  # Custom feature set
  python -u train.py --features all             # All 9 features
  python -u train.py --n-buckets 30             # More buckets for larger vocab
"""

import os, sys, argparse, json, time, traceback
os.environ["PYTHONUNBUFFERED"] = "1"

from pathlib import Path
import numpy as np

# Ensure src/ is on path
_src = str(Path(__file__).parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# Clear stale __pycache__
import shutil
for root, dirs, files in os.walk(Path(__file__).parent / "src"):
    if "__pycache__" in dirs:
        shutil.rmtree(os.path.join(root, "__pycache__"), ignore_errors=True)
        dirs.remove("__pycache__")

from ising_spin import IntegerLM, Vocabulary, IDX2POS
from ising_spin.feature_hash_energy import (
    FeatureSpec, default_features,
    LexBigramFeature, LexSkipFeature, LexTrigramFeature,
    ClassWordBigramFeature, WordClassBigramFeature,
    ClassWordSkipFeature, WordClassSkipFeature,
    ClassTrigramFeature,
)
from ising_spin.utils import get_rss_mb

CACHE_DIR = Path(__file__).parent
OUTPUT_DIR = CACHE_DIR / "output"


# All available feature names
ALL_FEATURE_NAMES = [
    "lex_bi", "word_cls_bi", "cls_word_bi",
    "lex_skip", "word_cls_skip", "cls_word_skip",
    "cls_tri", "lex_tri",
]


def build_features(feature_names, args):
    """Build feature list from names and CLI args."""
    features = []

    for name in feature_names:
        if name == "lex_bi":
            features.append(LexBigramFeature(
                n_hashes=args.lex_n_hashes, table_size=args.lex_table_size,
                eta=args.lex_eta, clip=args.lex_clip, weight=1.0,
            ))
        elif name == "word_cls_bi":
            features.append(WordClassBigramFeature(
                n_hashes=args.cls_n_hashes, table_size=args.cls_table_size,
                eta=args.cls_eta, clip=args.cls_clip, weight=0.5,
            ))
        elif name == "cls_word_bi":
            features.append(ClassWordBigramFeature(
                n_hashes=args.cls_n_hashes, table_size=args.cls_table_size,
                eta=args.cls_eta, clip=args.cls_clip, weight=0.5,
            ))
        elif name == "lex_skip":
            features.append(LexSkipFeature(
                n_hashes=args.skip_n_hashes, table_size=args.skip_table_size,
                eta=args.skip_eta, clip=args.skip_clip, weight=0.3,
            ))
        elif name == "word_cls_skip":
            features.append(WordClassSkipFeature(
                n_hashes=args.skip_n_hashes, table_size=args.skip_table_size,
                eta=args.skip_eta, clip=args.skip_clip, weight=0.3,
            ))
        elif name == "cls_word_skip":
            features.append(ClassWordSkipFeature(
                n_hashes=args.skip_n_hashes, table_size=args.skip_table_size,
                eta=args.skip_eta, clip=args.skip_clip, weight=0.3,
            ))
        elif name == "cls_tri":
            features.append(ClassTrigramFeature(
                n_hashes=args.cls_n_hashes, table_size=args.cls_table_size,
                eta=args.cls_eta, clip=args.cls_clip, weight=0.5,
            ))
        elif name == "lex_tri":
            features.append(LexTrigramFeature(
                n_hashes=args.lex_n_hashes, table_size=args.lex_table_size,
                eta=args.lex_eta, clip=args.lex_clip, weight=0.3,
            ))
        else:
            print(f"    WARNING: Unknown feature '{name}' — skipping", flush=True)

    return features


def load_data(n_samples):
    """Load TinyStories dataset."""
    cache_files = {}
    for f in CACHE_DIR.glob("cached_tiny_stories_*.json"):
        parts = f.stem.split("_")
        size_str = parts[-1]
        try:
            size = int(size_str[:-1]) * (1000 if size_str.endswith("k") else 1000000)
            cache_files[size] = str(f)
        except (ValueError, IndexError):
            continue

    cache_path = cache_files.get(n_samples) or min(
        (s for s in cache_files if s >= n_samples), default=None, key=lambda s: s
    )
    if cache_path is None:
        sufficient = {s: p for s, p in cache_files.items() if s >= n_samples}
        if sufficient:
            cache_path = sufficient[min(sufficient.keys())]

    if cache_path:
        print(f"Loading cache: {cache_path}")
        with open(cache_path) as f:
            texts = json.load(f)
        if len(texts) >= n_samples:
            return texts[:n_samples]

    print("Downloading TinyStories from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split=f"train[:{n_samples}]")
    texts = [s["text"] for s in ds]
    cache_name = f"cached_tiny_stories_{n_samples // 1000}k.json"
    with open(CACHE_DIR / cache_name, "w") as f:
        json.dump(texts, f)
    return texts


def main():
    parser = argparse.ArgumentParser(description="Integer Language Model (v81 — Data-Driven Classes)")

    # Data
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--vocab", type=int, default=2000)
    parser.add_argument("--max-seq-len", type=int, default=30)
    parser.add_argument("--vocab-min-freq", type=int, default=5)

    # Word classes (v81: variable, data-driven)
    parser.add_argument("--n-buckets", type=int, default=20,
                        help="Number of frequency buckets for word classes. "
                             "K=20 means ~100 words/bucket for V=2000. "
                             "NOT the old static 13 POS tags.")

    # Feature selection
    parser.add_argument("--features", type=str, default="default",
                        help="Comma-separated feature names, 'default', or 'all'. "
                             f"Options: {', '.join(ALL_FEATURE_NAMES)}")

    # Shared feature parameters
    parser.add_argument("--cls-n-hashes", type=int, default=2)
    parser.add_argument("--cls-table-size", type=int, default=65537)
    parser.add_argument("--cls-eta", type=int, default=1)
    parser.add_argument("--cls-clip", type=int, default=50)

    parser.add_argument("--lex-n-hashes", type=int, default=3)
    parser.add_argument("--lex-table-size", type=int, default=65537)
    parser.add_argument("--lex-eta", type=int, default=1)
    parser.add_argument("--lex-clip", type=int, default=100)

    parser.add_argument("--skip-n-hashes", type=int, default=2)
    parser.add_argument("--skip-table-size", type=int, default=65537)
    parser.add_argument("--skip-eta", type=int, default=1)
    parser.add_argument("--skip-clip", type=int, default=80)

    # NCE
    parser.add_argument("--nce-epochs", type=int, default=3)
    parser.add_argument("--nce-negatives", type=int, default=3)

    # Generation
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--rep-penalty", type=float, default=3.0)
    parser.add_argument("--rep-window", type=int, default=5)
    parser.add_argument("--gen-length", type=int, default=100)

    args = parser.parse_args()

    # Resolve feature names
    if args.features == "default":
        feature_names = ["lex_bi", "word_cls_bi", "cls_word_bi", "lex_skip", "cls_tri", "lex_tri"]
    elif args.features == "all":
        feature_names = ALL_FEATURE_NAMES
    else:
        feature_names = [f.strip() for f in args.features.split(",")]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / f"ilm_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("INTEGER LANGUAGE MODEL v81 — Data-Driven Classes", flush=True)
    print(f"Started: {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"  Features: {', '.join(feature_names)}", flush=True)
    print(f"  Word classes: K={args.n_buckets} (frequency buckets, NOT POS)", flush=True)
    print(f"  Class table: size={args.cls_table_size}, hashes={args.cls_n_hashes}, "
          f"eta={args.cls_eta}, clip={args.cls_clip}", flush=True)
    print(f"  Lex table:   size={args.lex_table_size}, hashes={args.lex_n_hashes}, "
          f"eta={args.lex_eta}, clip={args.lex_clip}", flush=True)
    print(f"  Skip table:  size={args.skip_table_size}, hashes={args.skip_n_hashes}, "
          f"eta={args.skip_eta}, clip={args.skip_clip}", flush=True)
    print(f"  Gen: alpha={args.alpha}, T={args.temperature}, rep={args.rep_penalty}", flush=True)
    rss = get_rss_mb()
    if rss:
        print(f"Memory (RSS): {rss} MB", flush=True)
    print("=" * 70, flush=True)

    # [1] Load data
    print("\n[1/6] Loading data...", flush=True)
    texts = load_data(args.samples)
    print(f"  {len(texts):,} texts", flush=True)

    # [2] Build vocabulary + DATA-DRIVEN word classes
    print("\n[2/6] Building vocabulary + frequency buckets...", flush=True)
    vocab = Vocabulary(
        max_size=args.vocab,
        min_freq=args.vocab_min_freq,
        max_seq_len=args.max_seq_len,
        n_buckets=args.n_buckets,
    )
    vocab.build(texts)
    print(f"  {vocab.V} words, K={vocab.n_buckets} frequency buckets", flush=True)

    # Show frequency bucket distribution (the NEW class system)
    print(f"\n  === Frequency Bucket Distribution (DATA-DRIVEN, replaces POS) ===", flush=True)
    bucket_dist = vocab.bucket_distribution()
    for b in sorted(bucket_dist.keys()):
        count = bucket_dist[b]
        # Show example words for each bucket
        examples = [vocab.words[w] for w in range(4, vocab.V) if vocab.word_bucket[w] == b][:5]
        print(f"    Bucket {b:2d}: {count:4d} words  (e.g. {', '.join(examples)})", flush=True)

    # Show POS distribution (diagnostics only — NOT used in features)
    print(f"\n  === POS Distribution (diagnostics only, NOT used in features) ===", flush=True)
    for name, count in sorted(vocab.pos_distribution().items(), key=lambda x: -x[1]):
        print(f"    {name}: {count}", flush=True)

    # [3] Tokenize
    print("\n[3/6] Tokenizing...", flush=True)
    sequences = vocab.tokenize(texts)
    n_train = int(0.9 * len(sequences))
    train_seqs, test_seqs = sequences[:n_train], sequences[n_train:]
    print(f"  Train: {len(train_seqs):,}, Test: {len(test_seqs):,}", flush=True)

    # [4] Build model with dynamic features
    print("\n[4/6] Building Integer Language Model...", flush=True)
    features = build_features(feature_names, args)
    print(f"  Registered {len(features)} features:", flush=True)
    for feat in features:
        print(f"    {feat}", flush=True)

    model = IntegerLM(
        vocab=vocab,
        features=features,
        top_k=args.top_k,
        alpha=args.alpha,
        temperature=args.temperature,
        rep_penalty=args.rep_penalty,
        rep_window=args.rep_window,
        seed=42,
    )
    print(f"  Energy memory: {model.energy.memory_mb():.2f} MB", flush=True)
    print(f"  Bigram memory: {model.bigram.statistics().get('memory_mb', 0):.1f} MB", flush=True)

    # [5] Train + Calibrate
    print("\n[5/6] Training + calibrating...", flush=True)
    t0 = time.time()

    train_stats = model.train(train_seqs, n_epochs=args.nce_epochs, n_negatives=args.nce_negatives)
    cal_stats = model.calibrate(train_seqs)
    t_train = time.time() - t0

    # Show class transition matrix (v81: replaces POS matrix)
    print("\n  === Class Transition Matrix (from frequency buckets) ===", flush=True)
    cls_matrix = model.class_transition_matrix()
    K = cls_matrix.shape[0]
    # Show abbreviated matrix (max 10×10 for readability)
    show_K = min(K, 10)
    header = "        " + " ".join(f"B{j:>2d}" for j in range(show_K))
    print(header, flush=True)
    for i in range(show_K):
        row = f"  B{i:>2d}  "
        for j in range(show_K):
            row += f"{cls_matrix[i,j]:>5.0f}"
        print(row, flush=True)

    # [6] Evaluate
    print("\n[6/6] Evaluating...", flush=True)

    # Discriminative accuracy
    print(f"\n{'='*70}", flush=True)
    print("DISCRIMINATIVE ACCURACY", flush=True)
    disc = model.discriminative_accuracy(test_seqs, n_samples=500)
    print(f"  {disc['accuracy']:.3f} ({disc['comparisons']} comparisons)", flush=True)

    # Perplexity
    print(f"\n{'='*70}", flush=True)
    print("PERPLEXITY", flush=True)
    ppl = model.perplexity(test_seqs, n_samples=min(100, len(test_seqs)))
    print(f"  Base PPL: {ppl['base_ppl']:.2f}", flush=True)
    print(f"  LEGD PPL: {ppl['legd_ppl']:.2f}", flush=True)
    delta = ppl['base_ppl'] - ppl['legd_ppl']
    print(f"  {'IMPROVEMENT' if delta > 0 else 'REGRESSION'}: {abs(delta):.2f} PPL", flush=True)

    # Generation
    print(f"\n{'='*70}", flush=True)
    print("GENERATION", flush=True)
    for prompt in ["once upon a time", "there was a little", "the little girl"]:
        text = model.generate_text(prompt, length=args.gen_length)
        print(f"\n  '{prompt}':", flush=True)
        print(f"  {text[:300]}", flush=True)

    # Feature diagnostics
    print(f"\n{'='*70}", flush=True)
    print("FEATURE DIAGNOSTICS", flush=True)
    for feat in model.energy.features.values():
        fs = feat.statistics()
        print(f"  {feat.name}: weight={fs['weight']:.1f}, "
              f"range=[{fs['range'][0]},{fs['range'][1]}], "
              f"mean={fs['mean']:.1f}, std={fs['std']:.1f}, "
              f"nnz={fs['nnz']}, mem={fs['memory_kb']:.0f}KB", flush=True)

    # Save results
    diag = model.diagnostics()
    t_total = time.time() - t0
    results = {
        "version": "2.1.0",
        "architecture": "Integer Language Model v81 — Data-Driven Classes",
        "timestamp": timestamp,
        "config": {
            "features": feature_names,
            "samples": args.samples,
            "vocab_size": args.vocab,
            "n_buckets": args.n_buckets,
            "nce_epochs": args.nce_epochs,
            "nce_negatives": args.nce_negatives,
        },
        "results": {
            "training_time_s": t_train,
            "total_time_s": t_total,
            "disc_accuracy": disc['accuracy'],
            "base_ppl": ppl['base_ppl'],
            "legd_ppl": ppl['legd_ppl'],
            "vocab_size": vocab.V,
            "n_buckets": vocab.n_buckets,
            "alpha": diag['alpha'],
            "temperature": diag['temperature'],
            "metropolis_threshold": diag['metropolis_threshold'],
            "feature_weights": diag['feature_weights'],
            "energy_mean": diag['energy_mean'],
            "energy_std": diag['energy_std'],
        },
    }
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}", flush=True)
    print(f"DONE — Integer Language Model v81")
    print(f"  Time: {t_total:.1f}s | Disc: {disc['accuracy']:.3f} | "
          f"Base PPL: {ppl['base_ppl']:.2f} | LEGD PPL: {ppl['legd_ppl']:.2f}")
    print(f"  Alpha: {diag['alpha']:.3f} | T: {diag['temperature']} | "
          f"Features: {diag['n_features']} | Buckets: {diag['n_classes']}")
    print(f"  Energy: mean={diag['energy_mean']:.1f}, std={diag['energy_std']:.1f}")
    print(f"  Results: {output_dir}")
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
