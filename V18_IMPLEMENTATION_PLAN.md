# ISG-LM v18 Implementation Plan — 10x Expressivity Gain

**Version**: v18.0 Draft  
**Date**: 2026-05-24  
**Source**: `isg_lm_v18_proposal_final.pdf`  
**Baseline**: v17.4 (PPL=19.19 best, incoherent generation)  
**Target**: v18.3 (PPL<10, coherent 400-token generation, integer-only on Pi 5)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Phase 1 — Quick Wins (v18.0)](#2-phase-1--quick-wins-v180)
3. [Phase 2 — Dense Memory (v18.1)](#3-phase-2--dense-memory-v181)
4. [Phase 3 — Temporal Dynamics (v18.2)](#4-phase-3--temporal-dynamics-v182)
5. [Phase 4 — Cross-Scale Features (v18.3)](#5-phase-4--cross-scale-features-v183)
6. [Validation Gates](#6-validation-gates)
7. [Risk Register](#7-risk-register)
8. [File Map](#8-file-map)

---

## 1. Architecture Overview

The v18 energy function replaces the flat additive v17 energy with a Product of Experts where each expert captures a distinct linguistic dependency:

```
E_total(w | ctx) = E_dense_am(w|ctx)       // Nonlinear pattern matching (replaces linear n-gram)
                 + E_vsa_bind(w|ctx)        // Compositional binding (word+POS+topic)
                 + E_reservoir(w|h(t))       // Long-range temporal dynamics
                 + E_factorial_state(w|s(t)) // Coupled discourse state
                 + E_rff(w|ctx)             // Cross-scale random features
                 + E_kn_backoff(w|ctx)       // KN-smoothed n-gram (kept as fallback)
                 + E_hard(w)                // Hard constraints (POS, same-word, closed-class)
```

**Budget**: ~34 MB memory, ~13 ms/token on Pi 5, ~75 tokens/sec. All integer-only (Q15/Q30/uint8/int16).

---

## 2. Phase 1 — Quick Wins (v18.0)

**Duration**: 1-2 weeks  
**Cumulative Expressivity Gain**: 5-8x  
**Risk**: Low

### Task 1.1: Create VSA Binding Module (`src/ising_spin/vsa/`)

**Files to create**:
- `src/ising_spin/vsa/__init__.py`
- `src/ising_spin/vsa/qfhrr.py` — Core qFHRR implementation

**Subtasks**:

| ID | Task | DoD | Est. |
|----|------|-----|------|
| 1.1.1 | Implement `QFHRRVectors` class with random uint8 vector generation (D=512, Q=256) | Unit test: generate 1000 vectors, verify each component in [0,255], verify D=512 | 2h |
| 1.1.2 | Implement `bind()` — element-wise modular addition: `(a + b) mod 256` | Unit test: bind(a, b) then unbind = b; verify round-trip for 100 random pairs | 1h |
| 1.1.3 | Implement `unbind()` — element-wise modular subtraction: `(a - b) mod 256` | Unit test: unbind(bind(a,b), b) ≈ a (within similarity threshold) | 1h |
| 1.1.4 | Implement `superpose()` — saturating addition with clip at 255 | Unit test: superpose of two known vectors matches expected sum; overflow clips correctly | 1h |
| 1.1.5 | Implement `similarity()` — sum of phase-difference lookups from 256-entry LUT | Unit test: self-similarity >> cross-similarity for 100 random vectors; similarity of bound pair preserves structure | 3h |
| 1.1.6 | Build 256-entry phase-difference lookup table at init time | Unit test: LUT[0] = max similarity; LUT[128] = min similarity; monotonic decrease | 2h |
| 1.1.7 | Implement `VSAEncoder` — encodes a token as `(hash_word ⊕ role_word) ⊎ (hash_pos ⊕ role_pos) ⊎ (hash_topic ⊕ role_topic)` | Unit test: encode("bank", VERB, SPORTS) ≠ encode("bank", NOUN, POLITICS); same word same POS same topic = same vector | 3h |
| 1.1.8 | Implement precomputed readout matrix: `R[w] = VSA_encode(w, pos_w, topic_w)` for all V words | Unit test: readout matrix shape (V, D) uint8; row sums within expected range; memory < 30 MB for V=49000 | 2h |
| 1.1.9 | Implement `compute_vsa_energy()` — `E_vsa(w) = -sim(context_encoding, R[w])` | Unit test: known context gives lower energy for compatible words; energy is int32; no overflow | 2h |
| 1.1.10 | Integration: Wire VSA energy into `EnergyComputer` as additive term `E_vsa_bind` | Integration test: v18.0 pipeline produces energies including VSA term; VSA term is non-zero and varies across candidates | 3h |

**Phase 1.1 DoD**:
- [ ] All 10 subtasks pass their unit tests
- [ ] VSA module imported and used by EnergyComputer
- [ ] VSA energy term is additive, integer-only, Q30-safe
- [ ] Memory for readout matrix < 30 MB at V=49000
- [ ] No float operations in VSA hot path

### Task 1.2: State Scale Rebalance

**Files to modify**: `src/ising_spin/energy/computer.py`, `src/ising_spin/model_v17.py` → `model_v18.py`, `train_v18.py`

| ID | Task | DoD | Est. |
|----|------|-----|------|
| 1.2.1 | Change `state_scale` default from 50 to 400 in `EnergyComputer.__init__` | Unit test: state_scale=400 in default config; energy contribution of state is 8-16% of total (not <3%) | 0.5h |
| 1.2.2 | Update `IsingLMModel.__init__` default `state_scale=400` | Config test: model_v18 uses state_scale=400 by default | 0.5h |
| 1.2.3 | Update `IsingLMGenerator.__init__` default `state_scale=400` | Generator test: generator passes state_scale=400 to energy computer | 0.5h |
| 1.2.4 | Add CLI arg `--state-scale` to `train_v18.py` with default 400 | CLI test: `python train_v18.py --help` shows --state-scale with default 400 | 0.5h |
| 1.2.5 | Verify state energy doesn't corrupt predictions at scale=400 | Regression test: PPL at scale=400 is not worse than scale=50 (may be different, but not catastrophically worse) | 1h |

**Phase 1.2 DoD**:
- [ ] `state_scale=400` is the new default across all modules
- [ ] State energy contributes 8-16% of total energy (not <3%)
- [ ] No PPL regression >20% from scale change alone
- [ ] CLI arg `--state-scale` works

### Task 1.3: Model v18 Scaffold

**Files to create/modify**:
- `src/ising_spin/model_v18.py` (copy from model_v17.py, extend)
- `train_v18.py` (copy from train_v17.py, extend)

| ID | Task | DoD | Est. |
|----|------|-----|------|
| 1.3.1 | Create `model_v18.py` from `model_v17.py` with v18 header, imports for VSA | Import test: `from ising_spin.model_v18 import IsingLMModel` works | 1h |
| 1.3.2 | Add VSA module construction to `IsingLMModel.train()` step 12.5 | Integration test: training pipeline builds VSA encoder and readout matrix | 2h |
| 1.3.3 | Create `train_v18.py` with v18 config defaults (vocab=49000, state_scale=400) | CLI test: `python train_v18.py --help` shows all v18 args | 1h |
| 1.3.4 | Add `--no-vsa` ablation flag to train_v18.py | Ablation test: `--no-vsa` runs without VSA module, PPL matches v17.4 baseline | 1h |

**Phase 1.3 DoD**:
- [ ] model_v18.py and train_v18.py exist and run
- [ ] VSA module constructed during training
- [ ] Ablation flag `--no-vsa` works
- [ ] Full pipeline: train → PPL → generate works end-to-end

### Task 1.4: Phase 1 Validation

| ID | Task | DoD | Est. |
|----|------|-----|------|
| 1.4.1 | Run PPL evaluation on 100-sequence test set with v18.0 | PPL < 15 (target from proposal Table 7) | 2h |
| 1.4.2 | Run beta sweep to find optimal beta for new energy landscape | Best beta found, PPL at best beta < 13 | 1h |
| 1.4.3 | Generate 400-token samples from 3 prompts | Qualitative: at least 2 coherent sentence-to-sentence transitions per sample | 1h |
| 1.4.4 | Run ambiguity test: "bank" in river vs financial context gets different VSA energies | Test: E_vsa("bank", NOUN, FINANCE) ≠ E_vsa("bank", NOUN, NATURE); difference > 10% of VSA scale | 1h |
| 1.4.5 | Memory profiling: RSS < 12 GB during training, < 2 GB during inference | Profiler output documented | 1h |
| 1.4.6 | Latency profiling: per-token generation < 6 ms on Pi 5 | Timer output documented | 1h |

**Phase 1 Complete DoD**:
- [ ] PPL < 15 on 100-sequence evaluation
- [ ] VSA binding distinguishes ambiguous words across contexts
- [ ] State energy contributes meaningfully (8-16% of total)
- [ ] Memory < 12 GB training, < 2 GB inference
- [ ] Generation latency < 6 ms/token on Pi 5
- [ ] All unit and integration tests pass
- [ ] Code committed with tag `v18.0`

---

## 3. Phase 2 — Dense Memory (v18.1)

**Duration**: 1-2 weeks  
**Cumulative Expressivity Gain**: 10-13x  
**Risk**: Low-Medium

### Task 2.1: Dense AM Energy Module (`src/ising_spin/dense_am/`)

**Files to create**:
- `src/ising_spin/dense_am/__init__.py`
- `src/ising_spin/dense_am/energy.py` — Dense AM energy with random feature approximation

| ID | Task | DoD | Est. |
|----|------|-----|------|
| 2.1.1 | Implement `RandomFeatureProjector` — fixed random integer matrix of shape (D, context_hash_dim) with D=256 | Unit test: projection shape correct; output is int8/int16; deterministic given same seed | 2h |
| 2.1.2 | Implement polynomial nonlinearity `F(x) = min(x*x >> k, MAX_VAL)` in integer arithmetic | Unit test: F(0)=0; F(MAX_VAL) = MAX_VAL; monotonic; no overflow for Q30 inputs | 1h |
| 2.1.3 | Implement `preaggregate_readout()` — compute Phi(w) = sum of psi(ctx_mu) over all contexts where w_mu = w | Unit test: Phi matrix shape (V, D) int16; row sums non-negative; memory < 25 MB for V=49K | 4h |
| 2.1.4 | Implement `compute_dense_am_energy()` — `E(w) = -phi(ctx) · Phi(w)` single D-dim dot product | Unit test: known context gives lower energy for frequent continuations; energy is int32; no overflow | 2h |
| 2.1.5 | Build cos/sin lookup tables (256 entries) for random feature projection | Unit test: LUT[0] = max; LUT[64] ≈ 0; LUT[128] = min; values in [-127, 127] | 1h |
| 2.1.6 | Add temperature parameter for controlling energy sharpness (F_degree) | Unit test: degree=1 recovers linear energy; degree=2 gives sharper landscape; std(energies) increases 2x with degree=2 | 2h |
| 2.1.7 | Integration: Wire Dense AM energy into `EnergyComputer` as `E_dense_am` replacing linear n-gram recall for word scale | Integration test: word-level recall uses Dense AM; POS/topic recall still use linear (for now) | 3h |

**Phase 2.1 DoD**:
- [ ] Dense AM module computes energy via random feature dot product
- [ ] Polynomial nonlinearity creates sharper energy landscape (std 2x vs linear)
- [ ] Pre-aggregated readout matrix computed offline during training
- [ ] Per-candidate computation: 256 integer multiply-accumulates
- [ ] All integer, Q30-safe, no overflow

### Task 2.2: Training Integration

| ID | Task | DoD | Est. |
|----|------|-----|------|
| 2.2.1 | Add Dense AM pre-aggregation step to `IsingLMModel.train()` after n-gram index build | Step runs in < 5 min for 500K samples; Phi matrix saved | 3h |
| 2.2.2 | Add `--dense-am-dim` CLI arg (default 256) and `--dense-am-degree` (default 2) | CLI test: args work; degree=1 recovers linear baseline | 1h |
| 2.2.3 | Add `--no-dense-am` ablation flag | Ablation test: without Dense AM, energy = linear n-gram only | 0.5h |

**Phase 2.2 DoD**:
- [ ] Dense AM pre-aggregation integrated into training pipeline
- [ ] CLI args for dimension and degree
- [ ] Ablation flag works

### Task 2.3: Phase 2 Validation

| ID | Task | DoD | Est. |
|----|------|-----|------|
| 2.3.1 | PPL evaluation on 100-sequence test set | PPL < 10 (target from proposal) | 2h |
| 2.3.2 | Energy distribution analysis: std(energies) across candidates increases 2x vs v17 | Metric documented | 1h |
| 2.3.3 | Generate 400-token samples from 3 prompts | Qualitative: fewer copied n-grams (<80 per 400 tokens); some coherent transitions | 1h |
| 2.3.4 | Memory profiling: RSS < 12 GB | Profiler output documented | 0.5h |
| 2.3.5 | Latency profiling: per-token < 8 ms on Pi 5 (Dense AM adds ~2.5 ms) | Timer output documented | 0.5h |

**Phase 2 Complete DoD**:
- [ ] PPL < 10 on 100-sequence evaluation
- [ ] Energy landscape is sharper than linear (2x std)
- [ ] Generation shows noticeable improvement in coherence
- [ ] Memory < 12 GB, latency < 8 ms/token on Pi 5
- [ ] Code committed with tag `v18.1`

---

## 4. Phase 3 — Temporal Dynamics (v18.2)

**Duration**: 2-3 weeks  
**Cumulative Expressivity Gain**: 12-18x  
**Risk**: Medium

### Task 3.1: Integer ESN Reservoir (`src/ising_spin/reservoir/`)

**Files to create**:
- `src/ising_spin/reservoir/__init__.py`
- `src/ising_spin/reservoir/integer_esn.py` — Integer Echo State Network

| ID | Task | DoD | Est. |
|----|------|-----|------|
| 3.1.1 | Implement `IntegerESN` class with fixed random binary W_in matrix {-1, 0, +1}^(512 × V) | Unit test: W_in shape correct; values in {-1,0,+1}; sparsity ~33% | 2h |
| 3.1.2 | Implement state update: `h(t) = clip(alpha * h(t-1) + W_in * one_hot(w_t), -2^15, 2^15)` | Unit test: state stays in int16 range; alpha=31130 (Q15 ≈ 0.95); update is pure integer | 3h |
| 3.1.3 | Implement precomputed readout matrix R (V × 512, int16) via ridge regression on training data | Unit test: R shape correct; values in int16 range; trained via integer ridge regression | 4h |
| 3.1.4 | Implement `compute_reservoir_energy()` — `E_reservoir(w) = -h(t) · R[w]` | Unit test: recent words get lower energy; energy is int32; no overflow | 2h |
| 3.1.5 | Implement `reset()` for new document | Unit test: h(t) = zero vector after reset | 0.5h |
| 3.1.6 | Integration: Wire reservoir energy into `EnergyComputer` as `E_reservoir` | Integration test: reservoir contributes energy; total energy includes reservoir term | 2h |

**Phase 3.1 DoD**:
- [ ] Integer ESN maintains int16 state vector of size 512
- [ ] State update is pure integer (Q15 alpha + int16 clip)
- [ ] Readout matrix precomputed offline via ridge regression
- [ ] Per-candidate computation: 512 integer multiply-accumulates
- [ ] Reservoir captures long-range dependencies (verified by energy correlation with distant words)

### Task 3.2: Factorial State Coupling

**Files to modify**: `src/ising_spin/state/document_state.py`

| ID | Task | DoD | Est. |
|----|------|-----|------|
| 3.2.1 | Add pairwise compatibility tables: `topic_mode_compat` (16×8), `topic_tense_compat` (16×4), `mode_tense_compat` (8×4), etc. (7 variables → C(7,2)=21 pairs, but only 5 most correlated) | Unit test: tables exist; shape correct; values are int16; built from training co-occurrence | 3h |
| 3.2.2 | Implement mean-field inference loop with 5 iterations | Unit test: loop converges in ≤5 iterations; energies decrease monotonically; computation is O(K) per variable | 4h |
| 3.2.3 | Implement coupling energy: `E_coupling(var_i, var_j) = lambda * compat_table[val_i, val_j]` | Unit test: coupling energy is int32; lambda is a Q15 parameter; total coupling energy < 20% of state energy | 2h |
| 3.2.4 | Add `--mf-iterations` CLI arg (default 5) and `--mf-lambda` (default 0.5 in Q15) | CLI test: args work | 1h |
| 3.2.5 | Integration: Replace independent state energy with coupled mean-field state energy | Integration test: coupled state produces different energies than independent state; difference is meaningful | 2h |

**Phase 3.2 DoD**:
- [ ] Pairwise compatibility tables built from training data
- [ ] Mean-field inference runs 5 iterations per token
- [ ] Coupling energy is weak (lambda-controlled) but non-zero
- [ ] Factorial state utilizes more of the 2^22 joint state space
- [ ] Computation is ~65K integer ops per token (negligible)

### Task 3.3: Phase 3 Validation

| ID | Task | DoD | Est. |
|----|------|-----|------|
| 3.3.1 | PPL evaluation on 100-sequence test set | PPL < 8 | 2h |
| 3.3.2 | Reservoir memory test: verify reservoir distinguishes "the X the Y" from "the Y the X" (position-sensitive) | Test: energies differ for same words at different positions | 1h |
| 3.3.3 | Factorial state test: verify coupled state produces different topic predictions for same word in different mode contexts | Test: topic prediction after "however" (ARGUMENT mode) differs from after "the" (NARRATIVE mode) | 1h |
| 3.3.4 | Generate 400-token samples | Qualitative: at least 5 coherent cross-sentence transitions; <20 copied n-grams | 1h |
| 3.3.5 | Memory: RSS < 12 GB; latency < 13 ms/token on Pi 5 | Profiler output documented | 1h |

**Phase 3 Complete DoD**:
- [ ] PPL < 8 on 100-sequence evaluation
- [ ] Reservoir provides position-sensitive energy
- [ ] Factorial state coupling produces context-dependent predictions
- [ ] Generation is noticeably more coherent
- [ ] Memory < 12 GB, latency < 13 ms/token
- [ ] Code committed with tag `v18.2`

---

## 5. Phase 4 — Cross-Scale Features (v18.3)

**Duration**: 1-2 weeks  
**Cumulative Expressivity Gain**: 15-25x  
**Risk**: Very Low

### Task 4.1: Random Fourier Feature Cross-Scale Layer (`src/ising_spin/rff/`)

**Files to create**:
- `src/ising_spin/rff/__init__.py`
- `src/ising_spin/rff/cross_scale.py` — Low-precision random Fourier features

| ID | Task | DoD | Est. |
|----|------|-----|------|
| 4.1.1 | Implement `CrossScaleRFF` — fixed random integer projection matrix combining word+POS+topic context hashes | Unit test: phi(ctx) shape (D,); D=256; values in int8 range | 2h |
| 4.1.2 | Implement integer cos lookup table (256 entries) for random feature computation | Unit test: LUT values in [-127, 127]; LUT[0]=127; LUT[64]≈0; LUT[128]=-127 | 1h |
| 4.1.3 | Implement pre-aggregated Theta matrix: `Theta[w] = sum of phi(ctx_mu)` for all contexts where w_mu=w | Unit test: Theta shape (V, D) int8; memory < 15 MB for V=49K | 3h |
| 4.1.4 | Implement `compute_rff_energy()` — `E_rff(w) = -phi(ctx) · Theta[w]` | Unit test: cross-scale features give different energies than independent scale energies; energy is int32 | 2h |
| 4.1.5 | Integration: Wire RFF energy into `EnergyComputer` as `E_rff` | Integration test: total energy includes RFF term; RFF captures combinations that independent scales miss | 2h |

**Phase 4.1 DoD**:
- [ ] RFF module creates cross-scale features from word+POS+topic context
- [ ] 256-dimensional int8 feature vector per position
- [ ] Pre-aggregated Theta matrix computed offline
- [ ] Per-candidate: 256 integer multiply-accumulates
- [ ] Captures combinations that independent scales cannot

### Task 4.2: Phase 4 Validation

| ID | Task | DoD | Est. |
|----|------|-----|------|
| 4.2.1 | Full PPL evaluation: 100-sequence test set, beta sweep | PPL < 8 (conservative); target < 5 if lucky | 2h |
| 4.2.2 | Cross-scale feature test: verify "NOUN in SPORTS" gives different energy than "NOUN in POLITICS" via RFF | Test: energy difference > 0 for same word across topic contexts | 1h |
| 4.2.3 | Generation quality: 400-token samples from 5 prompts | Qualitative: >5 coherent cross-sentence transitions; <20 copied n-grams; 50% sentences rated "plausible English" | 2h |
| 4.2.4 | Full Pi 5 benchmark: memory, latency, tokens/sec | Memory < 12 GB RSS; latency < 13 ms/token; >75 tokens/sec | 1h |
| 4.2.5 | Integer arithmetic audit: no float operations in hot path | Grep test: no `float`, `math.exp`, `math.log` in energy/sampling/reservoir/vsa/dense_am/rff code | 1h |

**Phase 4 Complete DoD**:
- [ ] PPL < 8 (target < 5) on 100-sequence evaluation
- [ ] Cross-scale features produce context-dependent energies
- [ ] Generation is mostly coherent with <20 copied n-grams
- [ ] Full Pi 5 budget met: <12 GB, <13 ms/token, >75 tok/s
- [ ] Zero float operations in inference hot path
- [ ] Code committed with tag `v18.3`

---

## 6. Validation Gates

Each phase must pass ALL of the following before proceeding:

| Gate | Tier 1 (Quantitative) | Tier 2 (Qualitative) | Tier 3 (Computational) |
|------|----------------------|---------------------|----------------------|
| v18.0 | PPL < 15 | ≥2 cross-sentence transitions | RSS < 12 GB, <6 ms/tok |
| v18.1 | PPL < 10 | <80 copied n-grams/400tok | RSS < 12 GB, <8 ms/tok |
| v18.2 | PPL < 8 | <40 copied n-grams/400tok | RSS < 12 GB, <13 ms/tok |
| v18.3 | PPL < 8 (target <5) | <20 copied n-grams/400tok | RSS < 12 GB, <13 ms/tok |

**Regression gate**: If any phase causes PPL to increase by >50% over previous phase, STOP and diagnose before proceeding.

---

## 7. Risk Register

| Risk | Severity | Probability | Mitigation | Trigger |
|------|----------|-------------|------------|---------|
| VSA binding loses word identity | High | Low | Ambiguity test; fallback to independent scales | Same word gets same energy in all POS/topic contexts |
| Dense AM over-sharp energy | Medium | Medium | Tune polynomial degree; add temperature parameter | Energy std > 10x linear baseline |
| ESN reservoir chaotic dynamics | Medium | Low | Clip state to int16; calibrate spectral radius to 0.95 | State overflows or oscillates wildly |
| Factorial state oscillation | Low | Low | Cap coupling lambda; use damping in mean-field | Energy doesn't converge in 5 iterations |
| Memory exceeds Pi 5 limits | High | Low | Profile at each phase; 34 MB total is well within 16 GB | RSS > 12 GB during training |
| Integer overflow in Q30 | High | Low | All energy sums capped at 2^30; saturating arithmetic | Energy values exceed 2^30 |
| Generation still incoherent | Medium | High | Validate at every phase; diagnose specific failure modes | PPL improves but generation remains fragmented |

---

## 8. File Map

### New Files (v18)

```
src/ising_spin/
├── vsa/
│   ├── __init__.py              # VSA module init
│   └── qfhrr.py                 # qFHRR binding implementation
├── dense_am/
│   ├── __init__.py              # Dense AM module init
│   └── energy.py                # Dense AM energy + random features
├── reservoir/
│   ├── __init__.py              # Reservoir module init
│   └── integer_esn.py           # Integer ESN implementation
├── rff/
│   ├── __init__.py              # RFF module init
│   └── cross_scale.py           # Cross-scale random Fourier features
├── model_v18.py                 # v18 training orchestrator (extends model_v17)
└── (existing files modified as needed)

train_v18.py                     # v18 training script (extends train_v17)
tests/
├── test_qfhrr.py                # VSA binding unit tests
├── test_dense_am.py             # Dense AM unit tests
├── test_integer_esn.py          # Integer ESN unit tests
├── test_rff.py                  # RFF unit tests
├── test_factorial_state.py      # Factorial state coupling tests
├── test_v18_integration.py      # End-to-end integration tests
├── test_v18_regression.py       # Regression tests vs v17 baseline
└── test_v18_property.py         # Property-based tests
```

### Modified Files (v18)

```
src/ising_spin/
├── energy/computer.py           # Add E_vsa, E_dense_am, E_reservoir, E_rff terms
├── state/document_state.py      # Add compatibility tables, mean-field coupling
├── generator.py                 # Update reservoir state tracking, v18 energy pipeline
└── sampling/boltzmann.py        # Extend max_delta to 50000 for wider energy range
```
