#!/usr/bin/env python3
"""
Integer Language Model — Training Script (v90 — Architectural Fix)

Pure integer language model. No neural nets. No torch dependency.
Runs on a Pi 5. Produces grammatically coherent text.

v90 FUNDAMENTAL FIXES over v89 (PPL=13.40, only 3/9 features active):
  7 independent code reviews identified critical architectural flaws:

  BIGRAM BASE MODEL (BIGGEST IMPACT):
  - Laplace alpha=1.0→0.01 + Jelinek-Mercer interpolation
  - Expected base PPL improvement: 27.74 → ~15-18

  FEATURE HASH ENERGY:
  - Hash independence fixed (double-hashing, collision correlation 65%→3%)
  - LexBigramFeature REMOVED (redundant with P_base)
  - Table sizes increased: lex 65537→262147 (load factor 3.05→0.76)
  - Class clip reduced: 50→20 (prevents binary saturation)
  - POS class system added (syntactically meaningful class→word features)
  - Per-feature balanced negatives (aligned to each feature's class_key)

  LEGD INFERENCE:
  - Metropolis gate REMOVED (was killing 25% of correct candidates)
  - top_k=200 (was 50 — 15-25% of tokens were invisible)
  - Alpha search expanded to [0, 5.0] (was [0, 1.0])
  - Repetition penalty reduced: 3.0→1.0

Usage:
  python -u train.py                           # Full run (50K texts, default features)
  python -u train.py --samples 5000 --vocab 1000  # Quick test
  python -u train.py --features default            # Default 8 features (2 lex + 3 POS + 1 freq + 2 dist)
  python -u train.py --features all                # All feature variants
  python -u train.py --n-clusters 40               # More distributional clusters
  python -u train.py --no-dist                     # Disable distributional clusters
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


# All available feature names (with class_key variants)
ALL_FEATURE_NAMES = [
    "lex_skip", "lex_tri",
    "cls_word_bi_freq", "cls_word_bi_dist", "cls_word_bi_pos",
    "word_cls_bi_freq", "word_cls_bi_dist", "word_cls_bi_pos",
    "cls_tri_dist", "cls_tri_freq", "cls_tri_pos",
    "cls_word_skip_freq", "cls_word_skip_dist", "cls_word_skip_pos",
    "word_cls_skip_freq", "word_cls_skip_dist", "word_cls_skip_pos",
]


def build_features(feature_names, args, has_dist=True):
    """Build feature list from names and CLI args."""
    features = []
    cls_nce_rate = args.cls_nce_rate

    for name in feature_names:
        if name == "lex_skip":
            features.append(LexSkipFeature(
                n_hashes=args.skip_n_hashes, table_size=args.skip_table_size,
                eta=args.skip_eta, clip=args.skip_clip, weight=0.3, nce_rate=1.0,
            ))
        elif name == "lex_tri":
            features.append(LexTrigramFeature(
                n_hashes=args.lex_n_hashes, table_size=args.lex_table_size,
                eta=args.lex_eta, clip=args.lex_clip, weight=0.5, nce_rate=1.0,
            ))
        elif name == "cls_word_bi_freq":
            features.append(ClassWordBigramFeature(
                n_hashes=args.cls_n_hashes, table_size=args.cls_table_size,
                eta=args.cls_eta, clip=args.cls_clip, weight=0.5, class_key="freq",
                nce_rate=cls_nce_rate,
            ))
        elif name == "cls_word_bi_dist":
            if has_dist:
                features.append(ClassWordBigramFeature(
                    n_hashes=args.cls_n_hashes, table_size=args.cls_table_size,
                    eta=args.cls_eta, clip=args.cls_clip, weight=0.5, class_key="dist",
                    nce_rate=cls_nce_rate,
                ))
        elif name == "cls_word_bi_pos":
            features.append(ClassWordBigramFeature(
                n_hashes=args.cls_n_hashes, table_size=args.cls_table_size,
                eta=args.cls_eta, clip=args.cls_clip, weight=0.5, class_key="pos",
                nce_rate=cls_nce_rate,
            ))
        elif name == "word_cls_bi_freq":
            features.append(WordClassBigramFeature(
                n_hashes=args.cls_n_hashes, table_size=args.cls_table_size,
                eta=args.cls_eta, clip=args.cls_clip, weight=0.5, class_key="freq",
                nce_rate=cls_nce_rate,
            ))
        elif name == "word_cls_bi_dist":
            if has_dist:
                features.append(WordClassBigramFeature(
                    n_hashes=args.cls_n_hashes, table_size=args.cls_table_size,
                    eta=args.cls_eta, clip=args.cls_clip, weight=0.5, class_key="dist",
                    nce_rate=cls_nce_rate,
                ))
        elif name == "word_cls_bi_pos":
            features.append(WordClassBigramFeature(
                n_hashes=args.cls_n_hashes, table_size=args.cls_table_size,
                eta=args.cls_eta, clip=args.cls_clip, weight=0.5, class_key="pos",
                nce_rate=cls_nce_rate,
            ))
        elif name == "cls_tri_dist":
            if has_dist:
                features.append(ClassTrigramFeature(
                    n_hashes=args.cls_n_hashes, table_size=args.cls_table_size,
                    eta=args.cls_eta, clip=args.cls_clip, weight=0.5, class_key="dist",
                    nce_rate=cls_nce_rate,
                ))
        elif name == "cls_tri_freq":
            features.append(ClassTrigramFeature(
                n_hashes=args.cls_n_hashes, table_size=args.cls_table_size,
                eta=args.cls_eta, clip=args.cls_clip, weight=0.5, class_key="freq",
                nce_rate=cls_nce_rate,
            ))
        elif name == "cls_tri_pos":
            features.append(ClassTrigramFeature(
                n_hashes=args.cls_n_hashes, table_size=args.cls_table_size,
                eta=args.cls_eta, clip=args.cls_clip, weight=0.5, class_key="pos",
                nce_rate=cls_nce_rate,
            ))
        elif name == "cls_word_skip_freq":
            features.append(ClassWordSkipFeature(
                n_hashes=args.skip_n_hashes, table_size=args.skip_table_size,
                eta=args.skip_eta, clip=args.skip_clip, weight=0.3, class_key="freq",
                nce_rate=cls_nce_rate,
            ))
        elif name == "cls_word_skip_dist":
            if has_dist:
                features.append(ClassWordSkipFeature(
                    n_hashes=args.skip_n_hashes, table_size=args.skip_table_size,
                    eta=args.skip_eta, clip=args.skip_clip, weight=0.3, class_key="dist",
                    nce_rate=cls_nce_rate,
                ))
        elif name == "cls_word_skip_pos":
            features.append(ClassWordSkipFeature(
                n_hashes=args.skip_n_hashes, table_size=args.skip_table_size,
                eta=args.skip_eta, clip=args.skip_clip, weight=0.3, class_key="pos",
                nce_rate=cls_nce_rate,
            ))
        elif name == "word_cls_skip_freq":
            features.append(WordClassSkipFeature(
                n_hashes=args.skip_n_hashes, table_size=args.skip_table_size,
                eta=args.skip_eta, clip=args.skip_clip, weight=0.3, class_key="freq",
                nce_rate=cls_nce_rate,
            ))
        elif name == "word_cls_skip_dist":
            if has_dist:
                features.append(WordClassSkipFeature(
                    n_hashes=args.skip_n_hashes, table_size=args.skip_table_size,
                    eta=args.skip_eta, clip=args.skip_clip, weight=0.3, class_key="dist",
                    nce_rate=cls_nce_rate,
                ))
        elif name == "word_cls_skip_pos":
            features.append(WordClassSkipFeature(
                n_hashes=args.skip_n_hashes, table_size=args.skip_table_size,
                eta=args.skip_eta, clip=args.skip_clip, weight=0.3, class_key="pos",
                nce_rate=cls_nce_rate,
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
    parser = argparse.ArgumentParser(description="Integer Language Model (v90 — Architectural Fix)")

    # Data
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--vocab", type=int, default=2000)
    parser.add_argument("--max-seq-len", type=int, default=30)
    parser.add_argument("--vocab-min-freq", type=int, default=5)

    # Word classes (v82: multiple class systems)
    parser.add_argument("--n-buckets", type=int, default=20,
                        help="Number of frequency buckets (freq class system)")
    parser.add_argument("--n-clusters", type=int, default=30,
                        help="Number of distributional clusters (dist class system)")
    parser.add_argument("--no-dist", action="store_true",
                        help="Disable distributional clusters (freq-only mode)")

    # Feature selection
    parser.add_argument("--features", type=str, default="default",
                        help="Comma-separated feature names, 'default', or 'all'. "
                             f"Options: {', '.join(ALL_FEATURE_NAMES)}")

    # Shared feature parameters
    parser.add_argument("--cls-n-hashes", type=int, default=2)
    parser.add_argument("--cls-table-size", type=int, default=65537)
    parser.add_argument("--cls-eta", type=int, default=1)
    parser.add_argument("--cls-clip", type=int, default=20,
                        help="Clip for class features (v90: clip=20, prevents binary saturation)")
    parser.add_argument("--cls-nce-rate", type=float, default=1.0,
                        help="NCE subsampling rate for class features (v88: 1.0 = update on all pairs, coordinated training)")

    parser.add_argument("--lex-n-hashes", type=int, default=3)
    parser.add_argument("--lex-table-size", type=int, default=262147)
    parser.add_argument("--lex-eta", type=int, default=1)
    parser.add_argument("--lex-clip", type=int, default=100)

    parser.add_argument("--skip-n-hashes", type=int, default=2)
    parser.add_argument("--skip-table-size", type=int, default=262147)
    parser.add_argument("--skip-eta", type=int, default=1)
    parser.add_argument("--skip-clip", type=int, default=80)

    # NCE
    parser.add_argument("--nce-epochs", type=int, default=2)
    parser.add_argument("--nce-negatives", type=int, default=25)

    # Generation
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--rep-penalty", type=float, default=1.0)
    parser.add_argument("--rep-window", type=int, default=5)
    parser.add_argument("--bigram-block-window", type=int, default=3,
                        help="How many recent bigrams to block repeating")
    parser.add_argument("--gen-length", type=int, default=100)

    args = parser.parse_args()

    use_dist = not args.no_dist

    # Resolve feature names
    if args.features == "default":
        feature_names = [
            "lex_skip",
            "lex_tri",
            "cls_word_bi_pos",
            "cls_word_skip_pos",
            "cls_tri_pos",
            "cls_word_bi_freq",
            "cls_word_bi_dist",
            "cls_tri_dist",
        ]
        if not use_dist:
            feature_names = [n for n in feature_names if "dist" not in n]
    elif args.features == "all":
        feature_names = ALL_FEATURE_NAMES
        if not use_dist:
            feature_names = [n for n in feature_names if "dist" not in n]
    else:
        feature_names = [f.strip() for f in args.features.split(",")]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / f"ilm_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("INTEGER LANGUAGE MODEL v90 — Architectural Fix", flush=True)
    print(f"Started: {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"  Features: {', '.join(feature_names)}", flush=True)
    class_str = f"freq(K={args.n_buckets}) + pos(K=13)"
    if use_dist:
        class_str += f" + dist(K={args.n_clusters})"
    print(f"  Word classes: {class_str}", flush=True)
    print(f"  Class table: size={args.cls_table_size}, hashes={args.cls_n_hashes}, "
          f"eta={args.cls_eta}, clip={args.cls_clip}, nce_rate={args.cls_nce_rate}", flush=True)
    print(f"  Lex table:   size={args.lex_table_size}, hashes={args.lex_n_hashes}, "
          f"eta={args.lex_eta}, clip={args.lex_clip}", flush=True)
    print(f"  Skip table:  size={args.skip_table_size}, hashes={args.skip_n_hashes}, "
          f"eta={args.skip_eta}, clip={args.skip_clip}", flush=True)
    print(f"  Gen: alpha={args.alpha}, T={args.temperature}, rep={args.rep_penalty}, "
          f"bigram_block={args.bigram_block_window}", flush=True)
    rss = get_rss_mb()
    if rss:
        print(f"Memory (RSS): {rss} MB", flush=True)
    print("=" * 70, flush=True)

    # [1] Load data
    print("\n[1/6] Loading data...", flush=True)
    texts = load_data(args.samples)
    print(f"  {len(texts):,} texts", flush=True)

    # [2] Build vocabulary + frequency buckets
    print("\n[2/6] Building vocabulary + word classes...", flush=True)
    vocab = Vocabulary(
        max_size=args.vocab,
        min_freq=args.vocab_min_freq,
        max_seq_len=args.max_seq_len,
        n_buckets=args.n_buckets,
        n_clusters=args.n_clusters,
    )
    vocab.build(texts)
    print(f"  {vocab.V} words", flush=True)

    # Show frequency bucket distribution
    print(f"\n  === Frequency Bucket Distribution (freq class system) ===", flush=True)
    bucket_dist = vocab.bucket_distribution()
    for b in sorted(bucket_dist.keys()):
        count = bucket_dist[b]
        examples = [vocab.words[w] for w in range(4, vocab.V) if vocab.word_bucket[w] == b][:5]
        print(f"    Bucket {b:2d}: {count:4d} words  (e.g. {', '.join(examples)})", flush=True)

    # Show POS distribution (diagnostics only)
    print(f"\n  === POS Distribution (diagnostics only, NOT used in features) ===", flush=True)
    for name, count in sorted(vocab.pos_distribution().items(), key=lambda x: -x[1]):
        print(f"    {name}: {count}", flush=True)

    # [3] Tokenize
    print("\n[3/6] Tokenizing...", flush=True)
    sequences = vocab.tokenize(texts)
    n_train = int(0.9 * len(sequences))
    train_seqs, test_seqs = sequences[:n_train], sequences[n_train:]
    print(f"  Train: {len(train_seqs):,}, Test: {len(test_seqs):,}", flush=True)

    # Build distributional clusters (v83: sorted partition clustering)
    if use_dist:
        print("\n  Building distributional clusters...", flush=True)
        vocab.build_distributional_clusters(train_seqs)

        # Show distributional cluster distribution
        print(f"\n  === Distributional Cluster Distribution (dist class system) ===", flush=True)
        cluster_dist = vocab.cluster_distribution()
        for c in sorted(cluster_dist.keys())[:15]:
            words = cluster_dist[c]
            count = sum(1 for w in range(4, vocab.V) if vocab.word_cluster[w] == c)
            print(f"    Cluster {c:2d}: {count:4d} words  (e.g. {', '.join(words)})", flush=True)
        if len(cluster_dist) > 15:
            print(f"    ... ({len(cluster_dist)} total clusters)", flush=True)

    # [4] Build model with dynamic features
    print("\n[4/6] Building Integer Language Model...", flush=True)
    has_dist = use_dist and vocab.word_cluster is not None
    features = build_features(feature_names, args, has_dist=has_dist)
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
        bigram_block_window=args.bigram_block_window,
        seed=42,
    )
    print(f"  Energy memory: {model.energy.memory_mb():.2f} MB", flush=True)
    print(f"  Bigram memory: {model.bigram.statistics().get('memory_mb', 0):.1f} MB", flush=True)
    print(f"  Class systems: {list(model.energy.word_classes.keys())}", flush=True)

    # [5] Train + Calibrate
    print("\n[5/6] Training + calibrating...", flush=True)
    t0 = time.time()

    train_stats = model.train(train_seqs, n_epochs=args.nce_epochs, n_negatives=args.nce_negatives)
    cal_stats = model.calibrate(train_seqs)
    t_train = time.time() - t0

    # Show class transition matrices for EACH class system
    for class_key in model.energy.word_classes.keys():
        print(f"\n  === Class Transition Matrix ({class_key}) ===", flush=True)
        cls_matrix = model.class_transition_matrix(class_key=class_key)
        K = cls_matrix.shape[0]
        show_K = min(K, 12)
        header = "        " + " ".join(f"C{j:>2d}" for j in range(show_K))
        print(header, flush=True)
        for i in range(show_K):
            row = f"  C{i:>2d}  "
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
        ck = fs.get('class_key', None)
        nr = fs.get('nce_rate', 1.0)
        print(f"  {feat.name}: weight={fs['weight']:.1f}, class={ck}, "
              f"range=[{fs['range'][0]},{fs['range'][1]}], "
              f"mean={fs['mean']:.1f}, std={fs['std']:.1f}, "
              f"nnz={fs['nnz']}, nce_rate={nr}, mem={fs['memory_kb']:.0f}KB", flush=True)

    # Save results
    diag = model.diagnostics()
    t_total = time.time() - t0
    results = {
        "version": "2.5.0",
        "architecture": "Integer Language Model v89 — Normalized Energy",
        "timestamp": timestamp,
        "config": {
            "features": feature_names,
            "samples": args.samples,
            "vocab_size": args.vocab,
            "n_buckets": args.n_buckets,
            "n_clusters": args.n_clusters if use_dist else 0,
            "use_dist": use_dist,
            "nce_epochs": args.nce_epochs,
            "nce_negatives": args.nce_negatives,
            "cls_nce_rate": args.cls_nce_rate,
        },
        "results": {
            "training_time_s": t_train,
            "total_time_s": t_total,
            "disc_accuracy": disc['accuracy'],
            "base_ppl": ppl['base_ppl'],
            "legd_ppl": ppl['legd_ppl'],
            "vocab_size": vocab.V,
            "class_systems": diag.get('class_systems', []),
            "n_classes_map": diag.get('n_classes_map', {}),
            "alpha": diag['alpha'],
            "temperature": diag['temperature'],
            "metropolis_threshold": diag['metropolis_threshold'],
            "feature_weights": diag['feature_weights'],
            "feature_class_keys": diag.get('feature_class_keys', {}),
            "energy_mean": diag['energy_mean'],
            "energy_std": diag['energy_std'],
        },
    }
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}", flush=True)
    print(f"DONE — Integer Language Model v90")
    print(f"  Time: {t_total:.1f}s | Disc: {disc['accuracy']:.3f} | "
          f"Base PPL: {ppl['base_ppl']:.2f} | LEGD PPL: {ppl['legd_ppl']:.2f}")
    print(f"  Alpha: {diag['alpha']:.3f} | T: {diag['temperature']} | "
          f"Features: {diag['n_features']} | "
          f"Classes: {diag.get('n_classes_map', {})}")
    print(f"  Energy: mean={diag['energy_mean']:.1f}, std={diag['energy_std']:.1f}")
    print(f"  Results: {output_dir}")
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
