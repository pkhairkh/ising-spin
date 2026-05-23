# GFST-Ising v14.0: Grassmann Flag Architecture

## Design Document

### Problem Statement

The v12.x Ising Spin Glass Language Model achieves PPL~50, far from the target of PPL~20. The fundamental issue is **expressivity**: the model is essentially a smoothed n-gram model with Boltzmann sampling. Cranking PMI weights (v13 approach) doesn't help because:

1. **Same-window PMI is redundant** with n-gram recall (both encode the same local context)
2. **Long-range PMI is too noisy** with only 500K training texts (4000×4000 sparse matrix)
3. **Symmetric PMI cannot capture word order** — PMI("the","dog") = PMI("dog","the"), but language is directional

This is not a parameter tuning problem — it's an architectural problem.

### Inspiration: gfst-hmb (pkhairkh)

The `gfst-hmb` repository on HuggingFace introduces three key concepts:

1. **Grassmann-flag state tracking**: Structured state representation using flag manifolds (nested subspaces)
2. **Block-based exact-token memory with sparse readout**: Explicit memory mechanism with integer-only retrieval
3. **MPUC-first student**: Minimal Parameter Unit Component design

The key insight from Grassmann algebra is that **states can be structured as nested subspaces** (flags), and **interactions between states can be antisymmetric** (wedge products). This is fundamentally more expressive than flat integer states with symmetric couplings.

### Architecture: Three Structural Innovations

#### 1. Flag State Representation (Grassmann Flags)

**Current model**: Each word is an atomic integer index. No internal structure.

**New model**: Each word carries a **flag** — a nested hierarchy:
```
V₀ ⊂ V₁ ⊂ V₂
word ∈ cluster ∈ topic
```

Concrete instantiation:
- `word` ∈ {1..V} (vocab index, V=4000)
- `cluster` = cluster[word] ∈ {1..C} (C=64 syntactic-semantic clusters)
- `topic` = topic[cluster] ∈ {1..S} (S=16 semantic topics)

**Why this helps PPL**:
- Cluster-level statistics (64×64) have **much better coverage** than word-level (4000×4000)
- Topic-level (16×16) is essentially **noise-free** even with 500K texts
- The hierarchy provides **natural backoff**: word → cluster → topic
- When word-level n-grams are sparse, cluster-level coupling provides reliable signal

**Integer-only implementation**:
- Cluster assignment: `word_to_cluster[V]` — integer lookup table
- Topic assignment: `cluster_to_topic[C]` — integer lookup table
- Word-to-topic: cached `word_to_topic[V]` — integer lookup table
- Cluster assignment via integer K-means (L1 distance, integer medians)

#### 2. Antisymmetric (Wedge) Couplings

**Current model**: PMI is symmetric — J[w1,w2] = J[w2,w1]. Cannot distinguish "the dog" from "dog the".

**New model**: Grassmann wedge product — J_fwd[c1,c2] ≠ J_bwd[c1,c2].

The wedge product in exterior algebra is antisymmetric:
```
a ∧ b = -b ∧ a
```

Applied to language:
```
J_wedge[c_i, c_j] = J_fwd[c_i, c_j] - J_bwd[c_j, c_i]
```

This is **antisymmetric by construction**:
```
J_wedge[c_i, c_j] = -J_wedge[c_j, c_i]
```

**Why this helps PPL**:
- Word ORDER is the most basic linguistic structure
- "DET → NOUN" is forward-likely but backward-unlikely
- "NOUN → VERB" is forward-likely but backward-unlikely
- Symmetric PMI CANNOT capture this — it averages forward and backward
- At cluster level (64×64), forward/backward counts are well-estimated

**Distance-dependent wedge coupling**:
- Distance 1 (adjacent): full strength (256/256)
- Distance 2: 3/4 strength (192/256)
- Distance 3: 1/2 strength (128/256)
- Distance 4: 1/4 strength (64/256)
- Distance 5: 1/8 strength (32/256)

**Integer-only implementation**:
- `J_fwd_dist[d]`: int16 matrix (64×64 = 8KB) for each distance d
- `J_bwd_dist[d]`: int16 matrix (64×64 = 8KB) for each distance d
- `J_wedge[d]` = fwd - bwd^T: precomputed int16 matrix
- Energy computation: integer multiply + shift (Q8 fixed point)

#### 3. Block Memory with Sparse Readout

**Current model**: Only sees last 5 words (5-gram window). No long-range context.

**New model**: Training text stored in blocks, retrieved by flag matching.

During training:
- Text split into blocks of 32 words
- Each block tagged with: topic_id, cluster_signature
- Blocks indexed by topic for fast retrieval

During generation:
- Current context's topic and cluster pattern computed
- Readout cache looks up: (topic, cluster_context_hash) → Counter of next words
- Provides "what followed this pattern in the training data?"
- This is **retrieval-augmented generation, integer-only**

**Why this helps PPL**:
- N-gram recall is limited to 5-word context
- Block memory provides **topic-consistent** word suggestions from anywhere in the training data
- Even when the exact 5-gram hasn't been seen, the topic+cluster pattern may match many blocks
- Readout cache is precomputed — O(1) lookup during generation

**Integer-only implementation**:
- Block storage: list of (topic, cluster_sig, word_array) tuples
- Topic index: `topic_id → list[block_indices]` (integer lists)
- Readout cache: `dict[(topic, ctx_hash), Counter[word → count]]` (integer counts)
- Memory energy: `weight × floor(log2(total/count))` (integer log2)

### Energy Function

The total energy for a candidate word w given context ctx is:

```
E(w|ctx) = E_recall(w, ctx)            [n-gram recall, PRIMARY]
         + E_flag_cluster(w)            [cluster consistency]
         + E_flag_topic(w)              [topic coherence]
         + E_wedge(w, ctx)              [antisymmetric coupling]
         + E_memory(w, ctx)             [block readout]
         + E_hard(w, ctx)               [hard constraints]
```

Where:
- `E_recall` remains the PRIMARY term (n-gram log-probability)
- `E_flag_cluster` penalizes words whose cluster doesn't follow the context's cluster pattern
- `E_flag_topic` penalizes off-topic words (topic ≠ current_topic)
- `E_wedge` provides direction-dependent coupling bonus for well-ordered cluster pairs
- `E_memory` provides bonus for words that followed similar patterns in training data
- `E_hard` keeps existing constraints (same-word penalty, grammar, etc.)

### Why This Is NOT "Just Cranking PMI Weight"

| Aspect | v13 (cranking PMI) | v14 (Grassmann Flag) |
|--------|-------------------|---------------------|
| State representation | Flat integer | Nested flag (word/cluster/topic) |
| Coupling structure | Symmetric (J=J^T) | Antisymmetric (J=-J^T) |
| Context mechanism | Local n-gram only | n-gram + block retrieval |
| Coupling dimension | 4000×4000 (sparse) | 64×64 (dense, reliable) |
| Word order | Cannot distinguish | Captured by wedge product |
| Long-range context | None (5-word window) | Block memory (full corpus) |
| Information type | Redundant with recall | Independent of recall |

### Implementation Files

1. **`src/ising_spin/grassmann_flag.py`** (~1100 lines): New module
   - `FlagState`: Word → Cluster → Topic hierarchy
   - `WedgeCoupling`: Antisymmetric distance-dependent couplings
   - `BlockMemory`: Integer-only retrieval-augmented generation
   - `GrassmannFlagLayer`: Unified interface

2. **`src/ising_spin/model.py`**: Modified
   - Added `GrassmannFlagLayer` import
   - Added `grassmann_flag_layer` parameter to `IsingLM.__init__`
   - Added Grassmann energy term to `_compute_word_energy`
   - Added topic state update in `generate`
   - Added `grassmann_flag_*` parameters to `IsingLMModel`
   - Added build step (11b/13) in `train()`

3. **`train_v14.py`** (~350 lines): New training script
   - Grassmann Flag Layer enabled by default
   - Ablation modes: `--only-flag`, `--only-wedge`, `--only-memory`
   - Full disable: `--no-grassmann`

### Training Command

```bash
# Full Grassmann Flag architecture
python -u train_v14.py --samples 500000

# Ablation: only flag energy (no wedge, no memory)
python -u train_v14.py --samples 500000 --only-flag

# Ablation: only wedge coupling (no flag, no memory)
python -u train_v14.py --samples 500000 --only-wedge

# Ablation: only block memory (no flag, no wedge)
python -u train_v14.py --samples 500000 --only-memory

# Baseline: without Grassmann (equivalent to v12)
python -u train_v14.py --samples 500000 --no-grassmann
```

### Expected PPL Improvement

Current v12 best: **PPL ≈ 50**

Conservative estimates:
- Flag state (cluster consistency): **-5 to -10 PPL** (better statistics at cluster level)
- Wedge coupling (word order): **-5 to -10 PPL** (captures directional structure)
- Block memory (long-range context): **-5 to -15 PPL** (retrieval beyond n-gram window)
- **Combined target: PPL ≈ 20-35**

The key reason for confidence: each component provides **independent information** that the current model completely lacks. This is not marginal improvement from parameter tuning — it's adding entirely new dimensions of linguistic structure.

### Memory Footprint

| Component | Size |
|-----------|------|
| FlagState lookup tables | V × (2 + 1) bytes = ~12 KB |
| Cluster bigram (fwd+bwd) | 64 × 64 × 2 × 8 bytes = ~65 KB |
| Wedge couplings (5 distances) | 5 × 64 × 64 × 2 bytes = ~40 KB |
| Block storage (500K blocks) | ~64 MB |
| Readout cache | ~10 MB |
| **Total additional** | **~75 MB** |

This is well within the Pi's 16GB RAM budget.
