#!/usr/bin/env python3
"""
Integer EBM v77 — Hash-Compressed Energy + Local Energy-Guided Decoding

ARCHITECTURAL RETHINK from v76h:

  The DAM is clinically dead. Alpha=0.0 in v76h means the entire Ising
  model (512-dim J-matrix, SDR encoding, three-band spin state, VSA
  binding, episodic memory) contributed NOTHING to the 0.799 rerank_acc.
  That score came entirely from bigram + repetition penalty.

  v77 KILLS the DAM and replaces it with:

  PHASE 1: Hash-Compressed Integer Energy Table
    - No SDR, no J-matrix, no DAM layer
    - Pure integer hash table: O(1) lookup per candidate
    - Multi-hash ensemble (3 independent hash functions)
    - Bigram + trigram energy via double/triple hashing
    - Implicit generalization through hash collisions
    - Pure integer NCE training: table[h] -= eta for positive, += eta for negative

  PHASE 2: Local Energy-Guided Decoding (LEGD)
    - No reranker — energy guides at EACH generation step
    - delta_E = hash_energy(prev, candidate) + tri * hash_energy(prev2, prev, candidate)
    - P(s_t) proportional to P_base(s_t) * exp(-alpha * delta_E / T)
    - Adaptive Metropolis gate: hard-reject if delta_E > threshold
    - Repetition penalty carried over from v76h

  WHY THIS SHOULD WORK:
    1. Local delta_E has ~100x better SNR than global E
    2. Hash table was trained on EXACTLY these local decisions
    3. O(1) per candidate vs O(D^2) for DAM
    4. No variance explosion from summing thousands of couplings
    5. Implicit generalization via hash collisions
    6. Memory: ~1.2 MB vs 520 KB for DAM (negligible difference)

  WHAT WE LOSE:
    - Long-range context beyond trigrams
    - Hierarchical/abstract representations
    - The "physics" of the Ising model

  WHAT WE GAIN:
    - A signal that actually helps (alpha != 0)
    - 100x faster decoding
    - Simpler, debuggable pipeline
    - No z-score normalization hacks

Pipeline:
  1. Load base model (GPT-2 or DummyBaseLM)
  2. Load corpus, build vocabulary, tokenize
  3. Build POS type system (for NCE corruption types)
  4. Build HashEnergyTable (bigram + trigram hash tables)
  5. Build LEGD decoder
  6. NCE training (3 corruption types, pure integer updates)
  7. Calibrate alpha/beta/metropolis_threshold
  8. Evaluate: disc_acc, PPL, generation

Usage:
  python -u train_v77.py --samples 5000 --vocab 1000 --no-base-model --nce-epochs 1
  python -u train_v77.py  # Default: 50K samples, GPT-2 base model
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


def build_pos_types(vocab_words, word_freq):
    """Build simple POS type system based on word suffixes."""
    n_types = 13
    pos_types = np.zeros(len(vocab_words), dtype=np.int32)

    for i, word in enumerate(vocab_words):
        if i < 4:
            pos_types[i] = 0
        elif word.endswith(("ed",)):
            pos_types[i] = 1
        elif word.endswith(("ing",)):
            pos_types[i] = 2
        elif word.endswith(("ly",)):
            pos_types[i] = 3
        elif word.endswith(("tion", "ment", "ness", "ity")):
            pos_types[i] = 4
        elif word.endswith(("ous", "ful", "less", "ive", "able")):
            pos_types[i] = 5
        elif word in ("the", "a", "an"):
            pos_types[i] = 6
        elif word in ("is", "was", "were", "are", "am", "be", "been", "being"):
            pos_types[i] = 7
        elif word in ("he", "she", "it", "they", "we", "i", "you"):
            pos_types[i] = 8
        elif word in ("and", "but", "or", "so", "because", "when", "if"):
            pos_types[i] = 9
        elif word in ("to", "from", "in", "on", "at", "with", "by", "for"):
            pos_types[i] = 10
        elif word in (".", ",", "!", "?", ";", ":"):
            pos_types[i] = 11
        else:
            pos_types[i] = 12

    return pos_types


def main():
    parser = argparse.ArgumentParser(
        description="Integer EBM v77 — Hash-Compressed Energy + LEGD"
    )

    # Core parameters
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)

    # Base model
    parser.add_argument("--base-model", type=str, default="gpt2")
    parser.add_argument("--no-base-model", action="store_true", default=False,
                        help="Use DummyBaseLM (no torch required)")

    # Hash energy parameters
    parser.add_argument("--n-hashes", type=int, default=3,
                        help="Number of independent hash functions (ensemble size)")
    parser.add_argument("--table-size", type=int, default=65537,
                        help="Prime number for hash table size")
    parser.add_argument("--use-trigram", action="store_true", default=True,
                        help="Enable trigram hash energy")
    parser.add_argument("--no-trigram", action="store_true", default=False,
                        help="Disable trigram hash energy")
    parser.add_argument("--trigram-weight", type=int, default=1,
                        help="Relative weight of trigram vs bigram energy")
    parser.add_argument("--hash-eta", type=int, default=1,
                        help="NCE learning rate for hash energy updates")
    parser.add_argument("--clip-value", type=int, default=1000,
                        help="Max absolute value for hash table entries (0=no clipping)")

    # LEGD parameters
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Mixing weight for hash energy (0=pure base, >0=energy correction)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for energy correction")
    parser.add_argument("--log-prob-scale", type=int, default=100)
    parser.add_argument("--metropolis-threshold", type=int, default=0,
                        help="Hard rejection threshold for delta_E (0=disabled)")

    # NCE
    parser.add_argument("--nce-epochs", type=int, default=3,
                        help="NCE training epochs")
    parser.add_argument("--nce-negatives", type=int, default=3,
                        help="Number of NCE negatives per positive")

    # Repetition penalty
    parser.add_argument("--rep-penalty", type=float, default=50.0,
                        help="Energy penalty for repeated words (0=disabled)")
    parser.add_argument("--rep-window", type=int, default=5,
                        help="How many recent words to check for repetition")

    # Generation
    parser.add_argument("--gen-length", type=int, default=100)
    parser.add_argument("--max-seq-len", type=int, default=30)
    parser.add_argument("--context-window", type=int, default=10)
    parser.add_argument("--vocab-min-freq", type=int, default=5)

    args = parser.parse_args()

    # --- Header ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / f"legd_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    use_dummy = args.no_base_model or args.base_model == "dummy"
    use_trigram = args.use_trigram and not args.no_trigram

    print("=" * 70, flush=True)
    print("INTEGER EBM v77 — Hash-Compressed Energy + Local Energy-Guided Decoding", flush=True)
    print(f"Started: {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Key architectural changes from v76h:", flush=True)
    print(f"  1. NO SDR, NO DAM, NO J-matrix — replaced by hash energy table", flush=True)
    print(f"  2. NO reranker — replaced by Local Energy-Guided Decoding", flush=True)
    print(f"  3. Hash table: n_hashes={args.n_hashes}, table_size={args.table_size}, "
          f"trigram={'ON' if use_trigram else 'OFF'}", flush=True)
    print(f"  4. Pure integer NCE training (no BLAS, no float32 intermediates)", flush=True)
    print(f"  5. O(1) per candidate lookup vs O(D^2) for DAM", flush=True)
    rss = get_rss_mb()
    if rss > 0:
        print(f"Memory (RSS): {rss:,} MB", flush=True)
    print("=" * 70, flush=True)

    # --- [1] Load Data ---
    print("\n[1/7] Loading data...", flush=True)
    texts = load_data(args.samples, dataset_name=args.dataset)
    n_texts = len(texts)
    print(f"  Using {n_texts:,} texts for training", flush=True)

    # --- [2] Build Vocabulary ---
    print("\n[2/7] Building vocabulary...", flush=True)
    vocab_words, word2idx, idx2word, word_freq = build_vocabulary(
        texts, max_vocab=args.vocab, min_freq=args.vocab_min_freq
    )
    V = len(vocab_words)
    print(f"  Vocabulary: {V} words", flush=True)

    # --- [3] Tokenize ---
    print("\n[3/7] Tokenizing...", flush=True)
    sequences = tokenize_texts(texts, word2idx, max_seq_len=args.max_seq_len)
    n_train = int(0.9 * len(sequences))
    train_seqs = sequences[:n_train]
    test_seqs = sequences[n_train:]
    print(f"  Train: {len(train_seqs):,}, Test: {len(test_seqs):,}", flush=True)

    # --- [4] Build POS types ---
    print("\n[4/7] Building POS type system...", flush=True)
    pos_types = build_pos_types(vocab_words, word_freq)
    print(f"  POS system: {len(set(pos_types))} types, {V} words typed", flush=True)

    # --- [5] Build Base Model ---
    print("\n[5/7] Building base model...", flush=True)
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

    # --- [6] Build HashEnergyTable and LEGD Decoder ---
    print("\n[6/7] Building HashEnergyTable + LEGD decoder...", flush=True)
    from ising_spin.attractor.hash_energy import HashEnergyTable
    from ising_spin.attractor.legd_decoder import LEGDDecoder
    from ising_spin.attractor.corruptions import Corruptor

    hash_energy = HashEnergyTable(
        vocab_size=V,
        n_hashes=args.n_hashes,
        table_size=args.table_size,
        use_trigram=use_trigram,
        trigram_weight=args.trigram_weight,
        eta=args.hash_eta,
        clip_value=args.clip_value,
        seed=42,
    )

    corruptor = Corruptor(
        vocab_words=vocab_words,
        word2idx=word2idx,
        idx2word=idx2word,
        pos_types=pos_types,
        word_freq=word_freq,
        seed=42,
    )

    decoder = LEGDDecoder(
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

    print(f"  HashEnergyTable: n_hashes={args.n_hashes}, "
          f"table_size={args.table_size}, "
          f"trigram={'ON' if use_trigram else 'OFF'}", flush=True)
    print(f"  Memory estimate: {hash_energy.memory_mb():.2f} MB", flush=True)

    # --- [7] Train + Calibrate + Evaluate ---
    print("\n[7/7] Training hash energy + calibrating + evaluating...", flush=True)
    t_start = time.time()

    # Phase 1: Train hash energy
    print("\n  === Phase 1: Hash Energy NCE Training ===", flush=True)
    train_stats = decoder.train_hash_energy(
        sequences=train_seqs,
        n_epochs=args.nce_epochs,
        n_negatives=args.nce_negatives,
        corruptor=corruptor,
    )
    t_train = time.time() - t_start
    print(f"\n  Hash energy training complete: {t_train:.1f}s", flush=True)

    # Phase 2: Calibrate LEGD decoder
    print("\n  === Phase 2: LEGD Calibration ===", flush=True)
    cal_stats = decoder.calibrate(
        sequences=train_seqs,
        corruptor=corruptor,
    )

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
            print(f"  Hash energy IMPROVES the base model by {improvement:.2f} PPL", flush=True)
        else:
            print(f"  Hash energy HURTS the base model by {-improvement:.2f} PPL", flush=True)
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
            print(f"  LEGD: {text[:300]}", flush=True)

            gen_file = output_dir / f"generated_{i}.txt"
            with open(gen_file, "w") as f:
                f.write(text)

            # Also generate WITHOUT hash energy for comparison
            base_ids = list(prompt_ids)
            for step in range(args.gen_length):
                if hasattr(base_model, 'get_top_k_words') and not isinstance(base_model, type(base_model)):
                    try:
                        candidates, log_probs = base_model.get_top_k_words(
                            base_ids, k=min(args.top_k, V)
                        )
                    except Exception:
                        candidates, log_probs = base_model.get_top_k(
                            base_ids, k=min(args.top_k, V)
                        )
                else:
                    candidates, log_probs = base_model.get_top_k(
                        base_ids, k=min(args.top_k, V)
                    )

                valid = (candidates >= 4) & (candidates < V)
                if not np.any(valid):
                    break
                candidates = candidates[valid]
                log_probs = log_probs[valid]
                best = int(np.argmax(log_probs))
                base_ids.append(int(candidates[best]))

            base_text = " ".join(idx2word.get(wid, "<unk>") for wid in base_ids)
            print(f"\n  --- Base model only (no hash energy) ---", flush=True)
            print(f"  {base_text[:300]}", flush=True)

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
    print(f"  Hash bigram: nnz={diag['hash_bigram_nnz']:,}, "
          f"range=[{diag['hash_bigram_min']},{diag['hash_bigram_max']}], "
          f"density={diag['hash_bigram_density']:.4f}", flush=True)
    if 'hash_trigram_nnz' in diag:
        print(f"  Hash trigram: nnz={diag['hash_trigram_nnz']:,}, "
              f"density={diag['hash_trigram_density']:.4f}", flush=True)
    print(f"  Hash memory: {hash_energy.memory_mb():.2f} MB", flush=True)
    print(f"  Calibrated: {diag['calibrated']}", flush=True)

    # --- Save Results ---
    t_total = time.time() - t_start
    results = {
        "version": "77.0.0",
        "architecture": "Hash-Compressed Energy + LEGD (no DAM, no SDR, no J-matrix)",
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
            "hash_bigram_nnz": diag['hash_bigram_nnz'],
            "hash_bigram_range": f"[{diag['hash_bigram_min']},{diag['hash_bigram_max']}]",
            "base_model": "dummy" if use_dummy else args.base_model,
            "energy_mean": diag['energy_mean'],
            "energy_std": diag['energy_std'],
        },
    }

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {results_file}")

    root_results = CACHE_DIR / "training_results_v77.json"
    with open(root_results, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {root_results}")

    print(f"\n{'=' * 70}", flush=True)
    print(f"DONE — Integer EBM v77")
    print(f"Total time: {t_total:.1f}s ({t_total/60:.1f}min)")
    print(f"Discriminative accuracy: {disc_acc.get('overall_accuracy', 0.0):.3f}")
    print(f"Base PPL: {ppl_stats.get('base_ppl', float('inf')):.2f}")
    print(f"LEGD PPL: {ppl_stats.get('legd_ppl', float('inf')):.2f}")
    print(f"Alpha: {diag['alpha']:.1f}, Beta: {diag['beta']}")
    print(f"Results: {output_dir}")
    print(f"{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()
