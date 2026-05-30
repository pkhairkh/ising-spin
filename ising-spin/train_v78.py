#!/usr/bin/env python3
"""
Integer EBM v78 — Feature-Hashed Energy with POS Generalization

LEVEL 1 UPGRADE from v77:

  v77's HashEnergyTable hashes raw token IDs. This gives ZERO generalization:
  learning "the cat" tells you nothing about "a dog" because they hash to
  completely different slots. The table only memorizes FACTS, not RULES.

  v78 adds POS (Part-of-Speech) hash tables that learn CATEGORY-LEVEL rules:

    E = pos_weight * E_pos(POS(prev), POS(target))   <-- NEW: generalizable
      + lex_weight * E_lex(prev_id, target_id)        <-- Same as v77: specific

  With 13 POS types, there are only 13×13 = 169 possible POS bigram pairs.
  Every DET→NOUN transition ("the cat", "a dog", "this house", ...) maps
  to the SAME hash slot. Training on "the cat" automatically improves
  the score for "a dog" — without ever seeing that pair.

  The POS table learns RULES:
    DET→NOUN = strongly negative (good)
    DET→VERB = weakly positive (bad)
    NOUN→VERB = negative (good)
    NOUN→DET = strongly positive (bad)

  These rules apply to ALL words with those POS tags, including words
  the model has never seen together. That's GENERALIZATION.

  The lexical table handles token-specific knowledge (like v77).
  Both are trained simultaneously with integer NCE.

ARCHITECTURE:
  PHASE 1: Feature-Hashed Energy (POS + Lexical hash tables)
    - POS bigram + trigram tables (small prime ~1009, fast convergence)
    - Lexical bigram + trigram tables (large prime ~65537, same as v77)
    - Combined energy: pos_weight * E_pos + lex_weight * E_lex
    - Pure integer NCE training (no gradients, no BLAS, no float32)

  PHASE 2: LEGD with Feature-Weighted Energy
    - Same per-step guided decoding as v77
    - Now also calibrates pos_weight vs lex_weight
    - POS energy provides generalizable rejection signal
    - Lexical energy provides token-specific rejection signal

EXPECTED IMPROVEMENTS over v77:
  1. Generalization: Unseen word pairs scored by POS category rules
  2. Faster POS convergence: 169 slots × millions of updates = stable rules
  3. Better disc_acc on held-out vocabulary: rules apply to new words
  4. Complementary signals: POS catches syntactic errors, LEX catches semantic

Pipeline:
  1. Load base model (GPT-2 or DummyBaseLM)
  2. Load corpus, build vocabulary, tokenize
  3. Build POS type system (13 categories from vocabulary/pos.py)
  4. Build FeatureHashEnergyTable (POS + lexical tables)
  5. Build LEGD v2 decoder
  6. NCE training (simultaneous POS + lexical updates)
  7. Calibrate alpha/beta/pos_weight/lex_weight/metropolis_threshold
  8. Evaluate: disc_acc (POS vs lexical breakdown), PPL, generation

Usage:
  python -u train_v78.py --nce-epochs 3 --table-size 131071 --clip-value 500
  python -u train_v78.py --samples 5000 --vocab 1000 --no-base-model --nce-epochs 1
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

# Clear stale __pycache__
import shutil
for root, dirs, files in os.walk(Path(__file__).parent / "src"):
    if "__pycache__" in dirs:
        shutil.rmtree(os.path.join(root, "__pycache__"), ignore_errors=True)
        dirs.remove("__pycache__")

import argparse
import json
import time
import traceback
import numpy as np
from pathlib import Path

# --- Configuration ---
DEFAULT_SAMPLES = 50000
DEFAULT_VOCAB = 2000
DEFAULT_DATASET = "tinystories"

# Cache directory
CACHE_DIR = Path(__file__).parent
OUTPUT_DIR = CACHE_DIR / "output"


def get_rss_mb() -> int:
    """Get current process RSS in MB."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    except Exception:
        return 0


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

    print(f"Downloading {dataset_name} from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split=f"train[:{n_samples}]")
        texts = [s["text"] for s in ds]
        cache_name = f"cached_{cache_dataset}_{n_samples // 1000}k.json"
        with open(CACHE_DIR / cache_name, "w") as f:
            json.dump(texts, f)
        print(f"  Downloaded and cached {len(texts):,} texts")
        return texts
    except Exception as e:
        print(f"  Error downloading: {e}")
        return []


def build_vocabulary(texts: list, max_vocab: int = 2000, min_freq: int = 5):
    """Build vocabulary from texts."""
    from collections import Counter
    word_counts = Counter()
    for text in texts:
        words = text.lower().split()
        word_counts.update(words)

    filtered = {w: c for w, c in word_counts.items() if c >= min_freq}
    sorted_words = sorted(filtered.items(), key=lambda x: -x[1])
    vocab_words = ["<pad>", "<unk>", "<bos>", "<eos>"] + [w for w, _ in sorted_words[:max_vocab-4]]

    word2idx = {w: i for i, w in enumerate(vocab_words)}
    idx2word = {i: w for i, w in enumerate(vocab_words)}
    word_freq = np.zeros(len(vocab_words), dtype=np.int32)
    for w, c in sorted_words[:max_vocab-4]:
        if w in word2idx:
            word_freq[word2idx[w]] = c

    return vocab_words, word2idx, idx2word, word_freq


def tokenize_texts(texts: list, word2idx: dict, max_seq_len: int = 30):
    """Tokenize texts into word ID sequences."""
    sequences = []
    for text in texts:
        words = text.lower().split()
        ids = [word2idx.get(w, 1) for w in words]
        ids = [id for id in ids if id >= 4]
        if len(ids) >= 2:
            sequences.append(ids[:max_seq_len])
    return sequences


def build_word_pos_array(vocab_words, word2idx, idx2word):
    """
    Build word_pos array using POSTypeSystem from vocabulary/pos.py.

    Each word gets its PRIMARY POS type (the most specific one).
    This array is the key input for FeatureHashEnergyTable.
    """
    from ising_spin.vocabulary.pos import POSTypeSystem, POS2IDX

    pos_system = POSTypeSystem(vocab_size=len(vocab_words))
    pos_system.build_from_vocabulary(word2idx, idx2word)

    # Assign primary POS type per word (pick the most specific / lowest priority)
    TAG_PRIORITY = {
        POS2IDX["PUNCT"]: 0, POS2IDX["DET"]: 1, POS2IDX["PRON"]: 2,
        POS2IDX["AUX"]: 3, POS2IDX["CONJ"]: 4, POS2IDX["PART"]: 5,
        POS2IDX["PREP"]: 6, POS2IDX["NUM"]: 7, POS2IDX["ADV"]: 8,
        POS2IDX["ADJ"]: 9, POS2IDX["NOUN"]: 10, POS2IDX["VERB"]: 11,
        POS2IDX["X"]: 12,
    }

    word_pos = np.zeros(len(vocab_words), dtype=np.int32)
    for idx in range(len(vocab_words)):
        if idx in pos_system.allowed_types and pos_system.allowed_types[idx]:
            tags = list(pos_system.allowed_types[idx])
            # Pick the most specific (highest priority = lowest number)
            best_t = min(tags, key=lambda t: TAG_PRIORITY.get(t, 99))
            word_pos[idx] = best_t
        else:
            word_pos[idx] = POS2IDX["X"]

    return word_pos, pos_system


def main():
    parser = argparse.ArgumentParser(
        description="Integer EBM v78 — Feature-Hashed Energy + POS Generalization"
    )

    # Core parameters
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)

    # Base model
    parser.add_argument("--base-model", type=str, default="gpt2")
    parser.add_argument("--no-base-model", action="store_true", default=False,
                        help="Use DummyBaseLM (no torch required)")

    # POS hash table parameters
    parser.add_argument("--n-pos-hashes", type=int, default=2,
                        help="Number of hash functions for POS tables (2 is enough for 169 pairs)")
    parser.add_argument("--pos-table-size", type=int, default=1009,
                        help="Prime for POS hash table (1009 is plenty for 13x13=169 pairs)")
    parser.add_argument("--pos-eta", type=int, default=3,
                        help="NCE learning rate for POS tables (higher = faster rule convergence)")
    parser.add_argument("--pos-clip", type=int, default=500,
                        help="Clip value for POS table entries")
    parser.add_argument("--pos-weight", type=float, default=1.0,
                        help="Weight for POS energy in combined score")

    # Lexical hash table parameters (same as v77)
    parser.add_argument("--n-lex-hashes", type=int, default=3,
                        help="Number of hash functions for lexical tables")
    parser.add_argument("--lex-table-size", type=int, default=65537,
                        help="Prime for lexical hash table")
    parser.add_argument("--lex-eta", type=int, default=1,
                        help="NCE learning rate for lexical tables")
    parser.add_argument("--lex-clip", type=int, default=1000,
                        help="Clip value for lexical table entries")
    parser.add_argument("--lex-weight", type=float, default=1.0,
                        help="Weight for lexical energy in combined score")

    # Shared parameters
    parser.add_argument("--use-trigram", action="store_true", default=True,
                        help="Enable trigram hash energy")
    parser.add_argument("--no-trigram", action="store_true", default=False,
                        help="Disable trigram hash energy")
    parser.add_argument("--trigram-weight", type=int, default=1,
                        help="Relative weight of trigram vs bigram energy")

    # LEGD parameters
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--log-prob-scale", type=int, default=100)
    parser.add_argument("--metropolis-threshold", type=int, default=0)

    # NCE
    parser.add_argument("--nce-epochs", type=int, default=3)
    parser.add_argument("--nce-negatives", type=int, default=3)

    # Repetition penalty
    parser.add_argument("--rep-penalty", type=float, default=50.0)
    parser.add_argument("--rep-window", type=int, default=5)

    # Generation
    parser.add_argument("--gen-length", type=int, default=100)
    parser.add_argument("--max-seq-len", type=int, default=30)
    parser.add_argument("--vocab-min-freq", type=int, default=5)

    args = parser.parse_args()

    # --- Header ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / f"v78_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    use_dummy = args.no_base_model or args.base_model == "dummy"
    use_trigram = args.use_trigram and not args.no_trigram

    print("=" * 70, flush=True)
    print("INTEGER EBM v78 — Feature-Hashed Energy + POS Generalization", flush=True)
    print(f"Started: {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Key architectural changes from v77:", flush=True)
    print(f"  1. POS hash tables — category-level RULES, not token-specific facts", flush=True)
    print(f"  2. POS table: {args.n_pos_hashes} hashes, size={args.pos_table_size}, "
          f"13x13={13*13} possible pairs", flush=True)
    print(f"  3. Lexical table: {args.n_lex_hashes} hashes, size={args.lex_table_size}", flush=True)
    print(f"  4. Combined energy: {args.pos_weight}*E_pos + {args.lex_weight}*E_lex", flush=True)
    print(f"  5. POS generalizes: 'the cat' improves 'a dog' (same DET->NOUN slot)", flush=True)
    print(f"  6. Trigram: {'ON' if use_trigram else 'OFF'}", flush=True)
    rss = get_rss_mb()
    if rss > 0:
        print(f"Memory (RSS): {rss:,} MB", flush=True)
    print("=" * 70, flush=True)

    # --- [1] Load Data ---
    print("\n[1/8] Loading data...", flush=True)
    texts = load_data(args.samples, dataset_name=args.dataset)
    n_texts = len(texts)
    print(f"  Using {n_texts:,} texts for training", flush=True)

    # --- [2] Build Vocabulary ---
    print("\n[2/8] Building vocabulary...", flush=True)
    vocab_words, word2idx, idx2word, word_freq = build_vocabulary(
        texts, max_vocab=args.vocab, min_freq=args.vocab_min_freq
    )
    V = len(vocab_words)
    print(f"  Vocabulary: {V} words", flush=True)

    # --- [3] Tokenize ---
    print("\n[3/8] Tokenizing...", flush=True)
    sequences = tokenize_texts(texts, word2idx, max_seq_len=args.max_seq_len)
    n_train = int(0.9 * len(sequences))
    train_seqs = sequences[:n_train]
    test_seqs = sequences[n_train:]
    print(f"  Train: {len(train_seqs):,}, Test: {len(test_seqs):,}", flush=True)

    # --- [4] Build POS type system ---
    print("\n[4/8] Building POS type system...", flush=True)
    word_pos, pos_system = build_word_pos_array(vocab_words, word2idx, idx2word)

    # Count POS distribution
    from ising_spin.vocabulary.pos import IDX2POS
    pos_counts = {}
    for idx in range(V):
        pt = int(word_pos[idx])
        name = IDX2POS.get(pt, f"T{pt}")
        pos_counts[name] = pos_counts.get(name, 0) + 1

    print(f"  POS types: {len(pos_counts)} categories", flush=True)
    for name, count in sorted(pos_counts.items(), key=lambda x: -x[1]):
        print(f"    {name}: {count}", flush=True)

    # --- [5] Build Base Model ---
    print("\n[5/8] Building base model...", flush=True)
    base_model = None
    if use_dummy:
        from ising_spin.attractor.base_model import DummyBaseLM
        base_model = DummyBaseLM(
            vocab_words=vocab_words,
            word_freq=word_freq,
            seed=42,
        )
        base_model.build_bigrams(train_seqs)
        base_model.build_word_alignment(idx2word, word2idx)
        print(f"  DummyBaseLM: {V} words, bigrams built", flush=True)
    else:
        try:
            from ising_spin.attractor.base_model import BaseLMInterface
            base_model = BaseLMInterface(model_name=args.base_model)
            base_model.build_word_alignment(idx2word, word2idx)
            print(f"  Base model: {args.base_model} loaded (word-level candidates enabled)", flush=True)
        except (ImportError, Exception) as e:
            print(f"  Cannot load GPT-2: {e}", flush=True)
            print(f"  Falling back to DummyBaseLM", flush=True)
            from ising_spin.attractor.base_model import DummyBaseLM
            base_model = DummyBaseLM(
                vocab_words=vocab_words,
                word_freq=word_freq,
                seed=42,
            )
            base_model.build_bigrams(train_seqs)
            base_model.build_word_alignment(idx2word, word2idx)
            use_dummy = True

    # --- [6] Build FeatureHashEnergyTable ---
    print("\n[6/8] Building FeatureHashEnergyTable...", flush=True)
    from ising_spin.attractor.feature_hash_energy import FeatureHashEnergyTable
    from ising_spin.attractor.legd_decoder_v2 import LEGDDecoderV2
    from ising_spin.attractor.corruptions import Corruptor

    hash_energy = FeatureHashEnergyTable(
        vocab_size=V,
        word_pos=word_pos,
        n_pos_types=13,
        # POS tables
        n_pos_hashes=args.n_pos_hashes,
        pos_table_size=args.pos_table_size,
        pos_eta=args.pos_eta,
        pos_clip=args.pos_clip,
        # Lexical tables
        n_lex_hashes=args.n_lex_hashes,
        lex_table_size=args.lex_table_size,
        lex_eta=args.lex_eta,
        lex_clip=args.lex_clip,
        # Shared
        use_trigram=use_trigram,
        trigram_weight=args.trigram_weight,
        pos_weight=args.pos_weight,
        lex_weight=args.lex_weight,
        seed=42,
    )

    # Build simple pos_types array for Corruptor (uses the same word_pos)
    corruptor = Corruptor(
        vocab_words=vocab_words,
        word2idx=word2idx,
        idx2word=idx2word,
        pos_types=word_pos,
        word_freq=word_freq,
        seed=42,
    )

    decoder = LEGDDecoderV2(
        base_model=base_model,
        hash_energy=hash_energy,
        vocab_words=vocab_words,
        word2idx=word2idx,
        idx2word=idx2word,
        top_k=args.top_k,
        alpha=args.alpha,
        temperature=args.temperature,
        metropolis_threshold=args.metropolis_threshold,
        rep_penalty=args.rep_penalty,
        rep_window=args.rep_window,
        log_prob_scale=args.log_prob_scale,
        seed=42,
    )

    print(f"  POS tables: {args.n_pos_hashes} hashes, size={args.pos_table_size}", flush=True)
    print(f"  Lex tables: {args.n_lex_hashes} hashes, size={args.lex_table_size}", flush=True)
    print(f"  Trigram: {'ON' if use_trigram else 'OFF'}", flush=True)
    print(f"  Memory estimate: {hash_energy.memory_mb():.2f} MB", flush=True)
    print(f"  Feature weights: pos_weight={args.pos_weight}, lex_weight={args.lex_weight}", flush=True)

    # --- [7] Train + Calibrate + Evaluate ---
    print("\n[7/8] Training feature hash energy + calibrating + evaluating...", flush=True)
    t_start = time.time()

    # Phase 1: Train feature hash energy
    print("\n  === Phase 1: Feature-Hashed Energy NCE Training ===", flush=True)
    train_stats = decoder.train_hash_energy(
        sequences=train_seqs,
        n_epochs=args.nce_epochs,
        n_negatives=args.nce_negatives,
        corruptor=corruptor,
    )
    t_train = time.time() - t_start
    print(f"\n  Feature hash energy training complete: {t_train:.1f}s", flush=True)

    # --- Show learned POS rules ---
    print(f"\n  === Learned POS Transition Rules ===", flush=True)
    pos_matrix = hash_energy.get_pos_transition_matrix()

    print(f"\n  POS Transition Energy Matrix (lower = more likely):", flush=True)
    # Print header
    from ising_spin.vocabulary.pos import IDX2POS, COARSE_POS_TAGS
    header = "        " + " ".join(f"{IDX2POS.get(j, 'X'):>6s}" for j in range(13))
    print(header, flush=True)
    for i in range(13):
        row = f"  {IDX2POS.get(i, 'X'):>6s} "
        for j in range(13):
            val = pos_matrix[i, j]
            row += f"{val:>6.0f}"
        print(row, flush=True)

    # Find the most and least likely transitions
    print(f"\n  Top 10 MOST LIKELY POS transitions:", flush=True)
    flat_idx = np.argsort(pos_matrix, axis=None)
    for k in range(min(10, len(flat_idx))):
        i, j = np.unravel_index(flat_idx[k], pos_matrix.shape)
        print(f"    {IDX2POS.get(i, 'X'):>6s} -> {IDX2POS.get(j, 'X'):>6s}: "
              f"energy={pos_matrix[i,j]:.0f}", flush=True)

    print(f"\n  Top 10 LEAST LIKELY POS transitions:", flush=True)
    for k in range(min(10, len(flat_idx))):
        idx = flat_idx[-(k+1)]
        i, j = np.unravel_index(idx, pos_matrix.shape)
        print(f"    {IDX2POS.get(i, 'X'):>6s} -> {IDX2POS.get(j, 'X'):>6s}: "
              f"energy={pos_matrix[i,j]:.0f}", flush=True)

    # Phase 2: Calibrate LEGD decoder
    print("\n  === Phase 2: LEGD Calibration ===", flush=True)
    cal_stats = decoder.calibrate(
        sequences=train_seqs,
        corruptor=corruptor,
    )

    # --- [8] Evaluation ---
    print(f"\n[8/8] Evaluating...", flush=True)

    # --- Discriminative Accuracy ---
    print(f"\n{'=' * 70}", flush=True)
    print("DISCRIMINATIVE ACCURACY (KEY METRIC)", flush=True)
    print(f"{'=' * 70}", flush=True)

    try:
        disc_acc = decoder.compute_discriminative_accuracy(
            test_seqs, corruptor=corruptor, n_samples=500
        )
        print(f"  Overall: {disc_acc['overall_accuracy']:.3f}", flush=True)
        for key in sorted(disc_acc.keys()):
            if key.endswith("_accuracy"):
                print(f"    {key}: {disc_acc[key]:.3f}", flush=True)
    except Exception as e:
        print(f"  Error: {e}", flush=True)
        traceback.print_exc()
        disc_acc = {"overall_accuracy": 0.0}

    # --- Perplexity ---
    print(f"\n{'=' * 70}", flush=True)
    print("PERPLEXITY", flush=True)
    print(f"{'=' * 70}", flush=True)

    try:
        ppl_stats = decoder.compute_perplexity(test_seqs, n_samples=min(100, len(test_seqs)))
        print(f"  Base PPL:    {ppl_stats['base_ppl']:.2f}", flush=True)
        print(f"  LEGD PPL:    {ppl_stats['legd_ppl']:.2f}", flush=True)
        print(f"  Evaluated on: {ppl_stats['n_tokens']} tokens", flush=True)
        improvement = ppl_stats['base_ppl'] - ppl_stats['legd_ppl']
        if improvement > 0:
            print(f"  Feature hash energy IMPROVES the base model by {improvement:.2f} PPL", flush=True)
        else:
            print(f"  Feature hash energy HURTS the base model by {-improvement:.2f} PPL", flush=True)
    except Exception as e:
        print(f"  Error: {e}", flush=True)
        traceback.print_exc()
        ppl_stats = {"base_ppl": float("inf"), "legd_ppl": float("inf"), "n_tokens": 0}

    # --- Generation ---
    print(f"\n{'=' * 70}", flush=True)
    print("GENERATION", flush=True)
    print(f"{'=' * 70}", flush=True)

    prompts = ["once upon a time", "there was a little", "the little girl"]

    for i, prompt in enumerate(prompts):
        print(f"\n  --- '{prompt}' ({args.gen_length} words) ---", flush=True)
        try:
            prompt_words = prompt.lower().split()
            prompt_ids = [word2idx.get(w, 1) for w in prompt_words]
            prompt_ids = [pid for pid in prompt_ids if pid >= 4]

            if len(prompt_ids) == 0:
                print("  (prompt words not in vocab)", flush=True)
                continue

            # LEGD generation
            generated_ids = decoder.generate(
                prompt_ids, length=args.gen_length
            )

            text = " ".join(idx2word.get(wid, "<unk>") for wid in generated_ids)
            print(f"  LEGD v78: {text[:300]}", flush=True)

            gen_file = output_dir / f"generated_{i}.txt"
            with open(gen_file, "w") as f:
                f.write(text)

        except Exception as e:
            print(f"  Generation error: {e}", flush=True)
            traceback.print_exc()

    # --- Diagnostics ---
    print(f"\n{'=' * 70}", flush=True)
    print("DIAGNOSTICS", flush=True)
    print(f"{'=' * 70}", flush=True)

    diag = decoder.diagnostics()
    print(f"  Alpha: {diag['alpha']}", flush=True)
    print(f"  Beta: {diag['beta']}", flush=True)
    print(f"  Metropolis threshold: {diag['metropolis_threshold']}", flush=True)
    print(f"  Rep penalty: {diag['rep_penalty']}, window={diag['rep_window']}", flush=True)
    print(f"  Energy: mean={diag['energy_mean']:.1f}, std={diag['energy_std']:.1f}", flush=True)
    print(f"  Base: mean={diag['base_energy_mean']:.1f}, std={diag['base_energy_std']:.1f}", flush=True)

    if diag.get('has_feature_hash'):
        print(f"  POS weight: {diag['pos_weight']}", flush=True)
        print(f"  Lex weight: {diag['lex_weight']}", flush=True)
        print(f"  POS energy: mean={diag['pos_energy_mean']:.1f}, std={diag['pos_energy_std']:.1f}", flush=True)
        print(f"  Lex energy: mean={diag['lex_energy_mean']:.1f}, std={diag['lex_energy_std']:.1f}", flush=True)

    print(f"  POS bigram: nnz={diag.get('hash_pos_bigram_nnz', 0):,}, "
          f"range=[{diag.get('hash_pos_bigram_min', 0)},{diag.get('hash_pos_bigram_max', 0)}], "
          f"density={diag.get('hash_pos_bigram_density', 0):.4f}", flush=True)
    print(f"  Lex bigram: nnz={diag.get('hash_lex_bigram_nnz', 0):,}, "
          f"range=[{diag.get('hash_lex_bigram_min', 0)},{diag.get('hash_lex_bigram_max', 0)}], "
          f"density={diag.get('hash_lex_bigram_density', 0):.4f}", flush=True)
    print(f"  Hash memory: {hash_energy.memory_mb():.2f} MB", flush=True)
    print(f"  Calibrated: {diag['calibrated']}", flush=True)

    # --- Save Results ---
    t_total = time.time() - t_start
    results = {
        "version": "78.0.0",
        "architecture": "Feature-Hashed Energy + LEGD (POS generalization + lexical specificity)",
        "timestamp": timestamp,
        "config": vars(args),
        "results": {
            "training_time_s": t_train,
            "total_time_s": t_total,
            "discriminative_accuracy": disc_acc.get("overall_accuracy", 0.0),
            "base_ppl": ppl_stats.get("base_ppl", float("inf")),
            "legd_ppl": ppl_stats.get("legd_ppl", float("inf")),
            "vocab_size": V,
            "alpha": diag['alpha'],
            "beta": diag['beta'],
            "metropolis_threshold": diag['metropolis_threshold'],
            "pos_weight": diag.get('pos_weight', 1.0),
            "lex_weight": diag.get('lex_weight', 1.0),
            "pos_bigram_nnz": diag.get('hash_pos_bigram_nnz', 0),
            "pos_bigram_range": f"[{diag.get('hash_pos_bigram_min', 0)},{diag.get('hash_pos_bigram_max', 0)}]",
            "lex_bigram_nnz": diag.get('hash_lex_bigram_nnz', 0),
            "lex_bigram_range": f"[{diag.get('hash_lex_bigram_min', 0)},{diag.get('hash_lex_bigram_max', 0)}]",
            "base_model": "dummy" if use_dummy else args.base_model,
            "energy_mean": diag['energy_mean'],
            "energy_std": diag['energy_std'],
        },
    }

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {results_file}")

    root_results = CACHE_DIR / "training_results_v78.json"
    with open(root_results, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {root_results}")

    # Save POS transition matrix
    pos_matrix_file = output_dir / "pos_transition_matrix.json"
    pos_data = {}
    for i in range(13):
        for j in range(13):
            key = f"{IDX2POS.get(i, 'X')}->{IDX2POS.get(j, 'X')}"
            pos_data[key] = float(pos_matrix[i, j])
    with open(pos_matrix_file, "w") as f:
        json.dump(pos_data, f, indent=2)
    print(f"  Saved: {pos_matrix_file}")

    print(f"\n{'=' * 70}", flush=True)
    print(f"DONE — Integer EBM v78 — Feature-Hashed Energy + POS Generalization")
    print(f"Total time: {t_total:.1f}s ({t_total/60:.1f}min)")
    print(f"Discriminative accuracy: {disc_acc.get('overall_accuracy', 0.0):.3f}")
    print(f"Base PPL: {ppl_stats.get('base_ppl', float('inf')):.2f}")
    print(f"LEGD PPL: {ppl_stats.get('legd_ppl', float('inf')):.2f}")
    print(f"Alpha: {diag['alpha']:.1f}, Beta: {diag['beta']}")
    print(f"POS weight: {diag.get('pos_weight', 1.0)}, Lex weight: {diag.get('lex_weight', 1.0)}")
    print(f"Results: {output_dir}")
    print(f"{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()
