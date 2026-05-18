# Ising Spin Language Model

**Text generation with zero floating-point operations in the generation loop — now with accurate POS tagging, dependency couplings, and integer matrix factorization.**

A proof-of-concept demonstrating that grammatically and semantically structured text can be generated using only integer arithmetic — no `exp()`, no `softmax()`, no floating-point multiplications in the generation path.

## Architecture v3: Enhanced Typed Ising-Potts Model

The model uses a **coupled Ising-Potts architecture** where each position has:
- A **type** (POS tag, ~13 states) — Potts spin (spaCy-accurate)
- A **value** (specific word, ~5000+ states) — Ising-like spin (NMF-factorized)

The energy function is decomposed into six integer terms:

```
E(types, words) = E_type(types)        — POS sequence coupling (spaCy-accurate)
               + E_emit(words|types)   — word-type compatibility
               + E_lexical(words)      — PMI word-word coupling (NMF-factorized)
               + E_semantic(words)     — semantic type gating
               + E_grammar(types)      — grammar constraint penalties
               + E_dep(types, words)   — long-range dependency coupling [NEW]
```

### What's New in v3

| Enhancement | Description | Impact |
|------------|-------------|--------|
| **spaCy POS tagger** | Replaces rule-based POS with accurate spaCy tagging | "the"→DET (was NOUN), "is"→AUX (was NOUN) |
| **Dependency couplings (J_tree)** | Long-range subject-verb agreement from parse trees | 197K non-zero entries, 13 agreement rules |
| **Integer NMF (J ≈ W×H)** | Factorized coupling matrix for vocab scaling | 93.6% memory savings, enables 3K→10K+ vocab |
| **Larger corpus** | Default 100K samples (was 30K) | Denser PMI matrix, better coverage |

### Generation: Staged Annealing

| Phase | Temperature | What Resolves | Operations |
|-------|-------------|---------------|------------|
| **Phase 1** | High T | POS tag sequence | Exact enumeration over 13 states |
| **Phase 2** | Medium T | Types + Words | Gibbs with emission + PMI + deps |
| **Phase 3** | Low T | Specific words | Gibbs with full coupling (NMF O(K)) |

### How It Works

```
Corpus (fineweb-edu, 50K+ samples)
    │
    ▼  [spaCy POS + dependency parsing]
┌──────────────────────────────────────────────┐
│  spaCy tagger (one-time, training only):     │
│  - Accurate POS: DET, NOUN, VERB, AUX, etc. │
│  - Dependency edges: nsubj, dobj, amod, etc. │
│  - All stored as integer counts              │
└──────────────────────────────────────────────┘
    │
    ▼  [Integer counting + bit_length()]
┌──────────────────────────────────────────────┐
│  Compute log-floor PMI couplings:            │
│  J[i,j] = sign(cooc*N - marg_i*marg_j)      │
│         * (ratio.bit_length() - 1)           │
│  → Pure integer: bit_length = floor(log2)    │
└──────────────────────────────────────────────┘
    │
    ▼  [Dependency parse → J_tree]
┌──────────────────────────────────────────────┐
│  Build dependency couplings:                 │
│  J_tree[head, dep] = log-floor(dep count)    │
│  - 13 dependency labels (nsubj, dobj, etc.)  │
│  - Agreement rules from POS patterns         │
│  - LONG-RANGE: no distance limit!            │
└──────────────────────────────────────────────┘
    │
    ▼  [NMF factorization]
┌──────────────────────────────────────────────┐
│  Integer NMF: J ≈ W × H, W,H ∈ ℤ           │
│  - Memory: O(V²) → O(V×K), K << V           │
│  - Energy per position: O(K) not O(V)        │
│  - 93.6% memory savings                      │
└──────────────────────────────────────────────┘
    │
    ▼  [Saved as integer arrays]
┌──────────────────────────────────────────────┐
│        GENERATION LOOP (ZERO FP)             │
│                                              │
│  Phase 1: Update types via exact enum (13)   │
│  Phase 2: Update types + words via Gibbs     │
│  Phase 3: Update words only (types frozen)   │
│                                              │
│  All operations: integer add, compare,       │
│  cumsum + searchsorted, threshold lookup     │
│  + dependency coupling (sparse J_tree)       │
│  + NMF factorized energy (O(K) per pos)      │
└──────────────────────────────────────────────┘
    │
    ▼
  Generated text with POS annotations
```

## Simulation Results (v2 vs v3)

Trained on 20K fineweb-edu samples, 3K vocabulary, 5K spaCy-tagged texts:

### POS Assignment Accuracy

| Word | v2 (rule-based) | v3 (spaCy) | Correct? |
|------|----------------|------------|----------|
| the | NOUN | **DET** | ✓ |
| is | NOUN | **AUX** | ✓ |
| in | NOUN | **PREP** | ✓ |
| learn | NOUN | **VERB** | ✓ |
| students | NOUN | NOUN | ✓ |
| research | NOUN | NOUN | ✓ |

### Grammar Metrics (15 samples each)

| Metric | v2 | v3 | Change |
|--------|----|----|--------|
| aux_verb | 2 | 9 | **+7** |
| noun_verb | 0 | 2 | +2 |
| double_prep | 3 | 0 | **-3 (eliminated)** |
| det_noun | 5 | 4 | -1 |
| prep_noun | 9 | 6 | -3 |

### Dependency Coupling Statistics

| Dependency | Edge Count | Description |
|-----------|-----------|-------------|
| det | 207,432 | Determiner-noun coupling |
| amod | 110,340 | Adjective-noun coupling |
| nsubj | 117,561 | Subject-verb agreement |
| aux | 80,364 | Auxiliary-verb coupling |
| dobj | 77,124 | Verb-object coupling |
| compound | 83,955 | Noun compound coupling |
| advcl | 23,394 | Adverbial clause |
| ccomp | 24,525 | Clausal complement |

**13 agreement rules** extracted from dependency statistics, enabling long-range grammatical constraints.

### Integer NMF Performance

| Metric | Value |
|--------|-------|
| Memory savings | 93.6% |
| Full matrix elements | 9,024,016 |
| Factorized elements | 576,768 |
| K (latent factors) | 96 (48 pos + 48 neg) |
| Relative error | 0.755 |

## What Makes This Different From a Markov Chain?

1. **Bidirectional context**: Each word is conditioned on BOTH left AND right neighbors via Gibbs sampling — not just left-context like autoregressive models.

2. **Grammatical structure**: The type layer enforces POS constraints (DET→NOUN, AUX→VERB, etc.) through integer penalty terms — Marcolli's implicational couplings.

3. **Long-range dependencies**: J_tree enables subject-verb agreement across clause boundaries — beyond the local window. This is the key insight from Reinhart & De las Coves: long-range couplings enable context-sensitive generative capacity.

4. **Semantic coupling**: Log-floor PMI captures true association (not just frequency). "quantum"→"entanglement" gets high PMI; "the"→"cat" gets low PMI.

5. **Staged annealing**: Types resolve first (global structure), then words (local choices) — coarse-to-fine generation.

6. **Ising-Potts gating**: Following Haydarov et al. (arXiv:2502.12014), word-word coupling is active only when types match — syntactically-modulated generation.

## Theoretical Foundation

| Result | Reference | Implication |
|--------|-----------|-------------|
| Ising topology = grammar class | Reinhart & De las Coves (arXiv:2208.08301) | 1D chain → context-free; long-range → context-sensitive |
| Implicational couplings = syntax | Marcolli et al. (arXiv:1508.00504) | Grammar rules → integer penalty terms |
| Coupled Ising-Potts gating | Haydarov et al. (arXiv:2502.12014) | Type agreement gates word coupling |
| PMI ≈ Word2Vec | Levy & Goldberg (2014) | Integer PMI ≈ integer word2vec |
| NMF for matrices | Lee & Seung (2001) | J ≈ W×H enables scaling |

## Installation

```bash
pip install datasets numpy spacy
python -m spacy download en_core_web_sm
```

## Usage

### Train enhanced v3 model (recommended)

```bash
python run.py enhanced-train --n_samples 100000
```

### Generate text

```bash
python run.py enhanced-generate --prompt "the" --length 25
```

### Full demo with evaluation

```bash
python run.py enhanced-demo --n_samples 50000
```

### Thorough evaluation

```bash
python run.py enhanced-eval --n_samples 50000
```

### v2 model (legacy, no spaCy)

```bash
python run.py typed-train --n_samples 30000
python run.py typed-generate --prompt "the" --length 25
```

### Legacy character-level model

```bash
python run.py train --n_samples 50000
python run.py generate --prompt "the " --length 300
```

## The Zero-FP Claim

**In the generation loop, every operation is integer:**

1. **Energy computation**: `int64 + int64` — integer addition of coupling terms
2. **NMF-factorized energy**: `W[word,:] @ H_sum` — integer dot product, O(K)
3. **Type update**: exact enumeration over 13 states — integer comparison
4. **Word proposal**: cumulative sum + binary search — integer comparison
5. **Dependency coupling**: sparse J_tree lookup + integer addition
6. **Acceptance**: precomputed threshold table lookup + integer comparison
7. **Annealing**: switch between precomputed threshold tables — no runtime FP

The ONLY floating-point in the system occurs during **one-time precomputation** (offline):
- Building PMI threshold tables (uses `math.exp()` — could be replaced with integer LUT)
- NMF factorization (training only — the resulting W, H are pure integer)
- The actual PMI computation uses `bit_length()` — pure integer

In a production deployment, ALL tables would be precomputed constants. The entire generation loop would run on integer-only hardware (no FPU needed).

## Project Structure

```
ising-spin/
├── src/ising_spin/
│   ├── __init__.py          # Package init (v3.0.0)
│   ├── spacy_tagger.py      # [NEW] SpaCy POS tagger + dependency parser
│   ├── dep_couplings.py     # [NEW] Dependency tree couplings (J_tree)
│   ├── int_nmf.py           # [NEW] Integer matrix factorization (J ≈ W×H)
│   ├── enhanced_model.py    # [NEW] Unified v3 model + sampler
│   ├── pmi_couplings.py     # Log-floor PMI via bit_length()
│   ├── type_system.py       # POS type layer + grammar penalties
│   ├── semantic_types.py    # Semantic compatibility + Hebbian gating
│   ├── typed_sampler.py     # Staged annealing sampler (v2)
│   ├── typed_model.py       # Unified model pipeline (v2)
│   ├── char_model.py        # Legacy character-level model
│   ├── couplings.py         # Legacy word-level couplings
│   ├── sampler.py           # Legacy Gibbs sampler
│   ├── vocabulary.py        # Word-level vocabulary
│   ├── data_loader.py       # fineweb-edu loader
│   └── model.py             # Legacy word-level model
├── run.py                   # CLI entry point
└── README.md
```

## Limitations & Future Work

- **NMF approximation error**: Relative error of ~0.75 is significant; better initialization or more iterations would help
- **Slow convergence**: Gibbs sampling with type+word states needs many sweeps
- **No FP-free training**: Training still uses FP for NMF and probability table precomputation
- **No discourse coherence**: Model captures sentence-level structure but not paragraph/document structure
- **Subword units**: BPE within the Ising framework for open vocabulary

### Improvements to explore:
1. **Better NMF**: SVD-based initialization, more iterations, or alternating least squares for integer matrices
2. **Subword units**: BPE within the Ising framework for open vocabulary
3. **Hardware implementation**: Map integer operations to FPGA or ASIC
4. **Analog Ising solver**: Physical Ising machines for the Gibbs sampling step
5. **Multi-scale couplings**: Local (word-word), mid-range (type-type), long-range (dependency edges)

## Citation

If you use this work, please cite the underlying research:

- Grammar of the Ising Model: Reinhart & De las Coves (arXiv:2208.08301, Proc. Roy. Soc. A 2025)
- Spin Glass Syntax: Marcolli et al. (arXiv:1508.00504)
- Coupled Ising-Potts: Haydarov, Omirov & Rozikov (arXiv:2502.12014)
- PMI ≈ Word2Vec: Levy & Goldberg (2014)
- NMF: Lee & Seung (2001)
- Thermodynamic computing: Whitelam (arXiv:2506.15121, PRL 2026)
- MatMul-free LM: Zhu et al. (arXiv:2406.02528)

## License

MIT
