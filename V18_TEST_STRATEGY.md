# ISG-LM v18 Test Strategy

**Version**: v18.0 Draft  
**Date**: 2026-05-24  
**Scope**: Comprehensive test plan covering unit, integration, property-based, regression, performance, and acceptance testing for all v18 architectural upgrades.

---

## 1. Test Architecture Overview

```
Level 5: ACCEPTANCE TESTS        ← Pi 5 deployment, 400-token generation quality
Level 4: END-TO-END TESTS        ← Full train→eval→generate pipeline
Level 3: INTEGRATION TESTS       ← Module interactions (VSA+Energy, Reservoir+State)
Level 2: PROPERTY-BASED TESTS    ← Invariants (integer-only, no overflow, monotonicity)
Level 1: UNIT TESTS              ← Individual functions and classes
```

**Test Count Target**: 150+ tests across all levels  
**CI Integration**: All Level 1-3 tests must pass before any commit to main  
**Pi 5 Validation**: Level 4-5 run manually on Pi 5 before release tags

---

## 2. Level 1: Unit Tests

### 2.1 VSA / qFHRR Module (`tests/test_qfhrr.py`)

| ID | Test Name | What It Verifies | Critical Assertions |
|----|-----------|-------------------|---------------------|
| UT-VSA-01 | `test_vector_generation_shape` | Random uint8 vectors are correct shape | `vectors.shape == (n, 512)`, all values in `[0, 255]` |
| UT-VSA-02 | `test_vector_generation_deterministic` | Same seed produces same vectors | `v1 == v2` for same seed |
| UT-VSA-03 | `test_bind_unbind_roundtrip` | Binding then unbinding recovers original | `similarity(unbind(bind(a,b),b), a) > threshold` |
| UT-VSA-04 | `test_bind_commutativity` | Modular addition is commutative | `bind(a,b) == bind(b,a)` element-wise |
| UT-VSA-05 | `test_bind_associativity` | Modular addition is associative | `bind(bind(a,b),c) == bind(a,bind(b,c))` |
| UT-VSA-06 | `test_superpose_clipping` | Saturating addition clips at 255 | No value > 255 in output |
| UT-VSA-07 | `test_similarity_self_max` | Self-similarity is maximum | `similarity(v, v) > similarity(v, random_v)` for 100 trials |
| UT-VSA-08 | `test_similarity_orthogonal_random` | Random vectors have low similarity | `abs(similarity(v1, v2)) < threshold` for independent random v1, v2 |
| UT-VSA-09 | `test_binding_preserves_similarity` | Bound pairs preserve structure | `similarity(bind(a,b), bind(a,c))` correlates with `similarity(b,c)` |
| UT-VSA-10 | `test_lut_monotonic` | Phase-difference LUT is monotonically decreasing | `LUT[i] >= LUT[i+1]` for all i |
| UT-VSA-11 | `test_encoder_distinguishes_contexts` | Same word in different contexts produces different vectors | `encode("bank", VERB, SPORTS) != encode("bank", NOUN, POLITICS)` |
| UT-VSA-12 | `test_encoder_consistent` | Same encoding inputs produce same output | `encode("the", DET, 1) == encode("the", DET, 1)` |
| UT-VSA-13 | `test_readout_matrix_shape` | Readout matrix has correct dimensions | `R.shape == (V, 512)`, dtype uint8 |
| UT-VSA-14 | `test_readout_matrix_memory` | Readout matrix fits in memory budget | `R.nbytes < 30 * 1024 * 1024` for V=49000 |
| UT-VSA-15 | `test_vsa_energy_integer_only` | VSA energy computation uses no floats | Mock float functions; verify no calls |
| UT-VSA-16 | `test_vsa_energy_range` | VSA energy fits in int32 | All energies in `[-2^30, 2^30]` |
| UT-VSA-17 | `test_vsa_energy_context_sensitivity` | Different contexts produce different energy rankings | Word "run" gets different rank in SPORTS vs POLITICS |

### 2.2 Dense AM Module (`tests/test_dense_am.py`)

| ID | Test Name | What It Verifies | Critical Assertions |
|----|-----------|-------------------|---------------------|
| UT-DAM-01 | `test_projector_shape` | Random projection output shape | `phi.shape == (256,)` for any input |
| UT-DAM-02 | `test_projector_deterministic` | Same input, same projection | `phi1 == phi2` for same context |
| UT-DAM-03 | `test_projector_integer_only` | No float operations in projection | Mock floats; verify no calls |
| UT-DAM-04 | `test_polynomial_nonlinearity_degree1` | F(x)=x for degree=1 (linear) | `F(x, degree=1) == x` for range of x |
| UT-DAM-05 | `test_polynomial_nonlinearity_degree2` | F(x)=x² for degree=2 (quadratic) | `F(x, degree=2) ≈ x*x >> k` |
| UT-DAM-06 | `test_polynomial_no_overflow` | F(x) never exceeds MAX_VAL | `F(x) <= MAX_VAL` for all x in int32 range |
| UT-DAM-07 | `test_preaggregate_shape` | Phi matrix dimensions correct | `Phi.shape == (V, 256)`, dtype int16 |
| UT-DAM-08 | `test_preaggregate_memory` | Phi matrix in budget | `Phi.nbytes < 25 * 1024 * 1024` for V=49K |
| UT-DAM-09 | `test_dense_am_energy_shape` | Energy output shape matches candidates | `len(energies) == len(candidates)` |
| UT-DAM-10 | `test_dense_am_energy_range` | Energies fit in int32 | All values in `[-2^30, 2^30]` |
| UT-DAM-11 | `test_dense_am_sharper_than_linear` | Nonlinear energy has higher std than linear | `std(dense_energies) > 1.5 * std(linear_energies)` |
| UT-DAM-12 | `test_cos_lut_values` | Cos lookup table has correct values | `LUT[0] > 0`, `LUT[64] ≈ 0`, `LUT[128] < 0` |

### 2.3 Integer ESN Reservoir (`tests/test_integer_esn.py`)

| ID | Test Name | What It Verifies | Critical Assertions |
|----|-----------|-------------------|---------------------|
| UT-ESN-01 | `test_win_shape` | W_in matrix has correct shape | `W_in.shape == (512, V)` |
| UT-ESN-02 | `test_win_values` | W_in values are in {-1, 0, +1} | `set(unique(W_in)) ⊆ {-1, 0, 1}` |
| UT-ESN-03 | `test_win_sparsity` | W_in is ~33% sparse | `abs(nonzero_fraction - 0.33) < 0.1` |
| UT-ESN-04 | `test_state_update_shape` | State vector shape | `h.shape == (512,)`, dtype int16 |
| UT-ESN-05 | `test_state_update_range` | State stays in int16 | `all(-2^15 <= h[i] <= 2^15-1)` after 100 updates |
| UT-ESN-06 | `test_state_decay` | State decays with alpha | `norm(h(t+1)) < norm(alpha*h(t) + input)` for zero input |
| UT-ESN-07 | `test_state_integer_only` | No float in state update | Mock floats; verify no calls |
| UT-ESN-08 | `test_readout_shape` | Readout matrix R has correct shape | `R.shape == (V, 512)`, dtype int16 |
| UT-ESN-09 | `test_readout_memory` | Readout matrix in budget | `R.nbytes < 50 * 1024 * 1024` |
| UT-ESN-10 | `test_reservoir_energy_shape` | Energy output matches candidates | `len(energies) == len(candidates)` |
| UT-ESN-11 | `test_reservoir_energy_range` | Energy fits in int32 | All values in `[-2^30, 2^30]` |
| UT-ESN-12 | `test_reservoir_position_sensitivity` | Same word at different positions gets different energy | After "the cat the" vs "the the cat": energy("cat") differs |
| UT-ESN-13 | `test_reservoir_reset` | Reset zeros state | `all(h == 0)` after reset |
| UT-ESN-14 | `test_reservoir_recency` | Recent words have more influence | `similarity(h, R[recent_word]) > similarity(h, R[old_word])` |

### 2.4 Cross-Scale RFF Module (`tests/test_rff.py`)

| ID | Test Name | What It Verifies | Critical Assertions |
|----|-----------|-------------------|---------------------|
| UT-RFF-01 | `test_phi_shape` | Feature vector shape | `phi.shape == (256,)`, dtype int8 |
| UT-RFF-02 | `test_phi_integer_only` | No float in feature computation | Mock floats; verify no calls |
| UT-RFF-03 | `test_theta_shape` | Theta matrix dimensions | `Theta.shape == (V, 256)`, dtype int8 |
| UT-RFF-04 | `test_theta_memory` | Theta matrix in budget | `Theta.nbytes < 15 * 1024 * 1024` for V=49K |
| UT-RFF-05 | `test_rff_energy_shape` | Energy output matches candidates | `len(energies) == len(candidates)` |
| UT-RFF-06 | `test_rff_energy_range` | Energy fits in int32 | All values in `[-2^30, 2^30]` |
| UT-RFF-07 | `test_rff_cross_scale_sensitivity` | Combining different scales gives different features | `phi(word=5, pos=3, topic=1) != phi(word=5, pos=3, topic=2)` |
| UT-RFF-08 | `test_rff_cos_lut` | Cos lookup table correctness | `LUT[0]=127`, `LUT[128]=-127` |

### 2.5 Factorial State (`tests/test_factorial_state.py`)

| ID | Test Name | What It Verifies | Critical Assertions |
|----|-----------|-------------------|---------------------|
| UT-FS-01 | `test_compatibility_table_shape` | All pairwise tables correct shape | `topic_mode.shape == (16, 8)`, etc. |
| UT-FS-02 | `test_compatibility_table_values` | Table values are int16 | `dtype == int16` |
| UT-FS-03 | `test_mf_convergence` | Mean-field converges in ≤5 iterations | Energy change < threshold after 5 iterations |
| UT-FS-04 | `test_mf_monotonic` | Energy decreases during MF iterations | `E(i+1) <= E(i)` for each iteration |
| UT-FS-05 | `test_coupling_lambda_control` | Lambda=0 recovers independent state | `E_coupled(lambda=0) == E_independent` |
| UT-FS-06 | `test_coupling_effect` | Lambda>0 changes predictions | `state_pred(lambda=0.5) != state_pred(lambda=0)` for some contexts |
| UT-FS-07 | `test_mf_integer_only` | No float in mean-field loop | Mock floats; verify no calls |
| UT-FS-08 | `test_state_scale_rebalance` | scale=400 gives meaningful contribution | `state_energy_pct > 8%` of total energy |

---

## 3. Level 2: Property-Based Tests

### 3.1 Integer-Only Invariant (`tests/test_v18_property.py`)

| ID | Property | How Tested | Failure Condition |
|----|----------|------------|-------------------|
| PT-INT-01 | No float operations in energy hot path | Monkeypatch `float`, `math.exp`, `math.log`; run full energy computation | Any patched function called |
| PT-INT-02 | All energy values fit in int32 Q30 | Generate random contexts, compute energies, check range | Any energy > 2^30 or < -2^30 |
| PT-INT-03 | Energy sum never overflows int64 | Sum all energy terms for 50K candidates | sum > 2^63 - 1 |
| PT-INT-04 | VSA vectors are always uint8 | Generate 10K vectors, check all components | Any value not in [0, 255] |
| PT-INT-05 | Reservoir state always int16 | Run 1000 updates, check state after each | Any value not in [-2^15, 2^15-1] |
| PT-INT-06 | Dense AM Phi values are int16 | Build Phi matrix, check all entries | Any value not in [-2^15, 2^15-1] |
| PT-INT-07 | RFF Theta values are int8 | Build Theta matrix, check all entries | Any value not in [-127, 127] |

### 3.2 Monotonicity and Ordering Properties

| ID | Property | How Tested | Failure Condition |
|----|----------|------------|-------------------|
| PT-MON-01 | Lower energy = higher probability (Boltzmann) | For 100 random energy arrays, verify P(i) ~ exp(-beta*E(i)) | Sample distribution doesn't match Boltzmann |
| PT-MON-02 | More frequent n-gram = lower recall energy | Word with count 100 gets lower energy than count 1 | `E(count=100) > E(count=1)` |
| PT-MON-03 | VSA self-similarity > cross-similarity | 1000 random vector pairs | `sim(v,v) < sim(v,w)` for some v,w |
| PT-MON-04 | Reservoir recency effect | Verify recent words have higher activation | Older word has higher similarity than recent |
| PT-MON-05 | Dense AM sharpness increases with degree | std(energies, degree=2) > std(energies, degree=1) | Degree=2 is not sharper |

### 3.3 Consistency Properties

| ID | Property | How Tested | Failure Condition |
|----|----------|------------|-------------------|
| PT-CON-01 | Deterministic energy for same input | Compute energy twice for same context | `E1 != E2` |
| PT-CON-02 | VSA encode/decode round-trip | Encode then verify readout similarity | Self-similarity below threshold |
| PT-CON-03 | State update is deterministic | Update state twice with same word | Different resulting state |
| PT-CON-04 | Readout matrix is dense (no all-zero rows) | Check all rows of Phi, R, Theta matrices | Any row with all zeros |
| PT-CON-05 | Energy is translation-invariant | E(w) + constant doesn't change ranking | Ranking changes after adding constant |

---

## 4. Level 3: Integration Tests

### 4.1 Module Interaction Tests (`tests/test_v18_integration.py`)

| ID | Test Name | What It Verifies | Setup | Assertions |
|----|-----------|-------------------|-------|------------|
| IT-01 | `test_vsa_in_energy_computer` | VSA energy term integrated correctly | Small model (V=500) | Total energy includes VSA term; VSA term varies across candidates |
| IT-02 | `test_dense_am_in_energy_computer` | Dense AM energy term integrated | Small model | Total energy includes Dense AM term; sharper than linear |
| IT-03 | `test_reservoir_in_energy_computer` | Reservoir energy term integrated | Small model | Total energy includes reservoir term; reservoir tracks context |
| IT-04 | `test_rff_in_energy_computer` | RFF energy term integrated | Small model | Total energy includes RFF term; cross-scale interactions present |
| IT-05 | `test_factorial_state_in_energy_computer` | Coupled state energy integrated | Small model | State energy changes with coupling; lambda=0 matches independent |
| IT-06 | `test_all_experts_together` | Full v18 energy function | Small model | All terms contribute; energy is additive; no overflow |
| IT-07 | `test_kn_backoff_still_works` | KN backoff preserved as fallback | Small model | KN backoff gives reasonable energy for unseen words |
| IT-08 | `test_generator_with_v18_energy` | Generator produces text with v18 energy | Small model | Generate 50 tokens without crash; words are valid vocab IDs |
| IT-09 | `test_ppl_with_v18_energy` | PPL computation works with v18 energy | Small model | PPL is finite and positive; not inf or nan |
| IT-10 | `test_reservoir_state_persistence` | Reservoir state persists across tokens | Generate 20 tokens | Reservoir state h(t) ≠ h(0) after generation |
| IT-11 | `test_document_state_with_coupling` | Document state evolves with coupling | Generate 20 tokens | State values change during generation; coupling affects transitions |
| IT-12 | `test_ablation_no_vsa` | --no-vsa flag removes VSA term | Train with flag | VSA energy = 0; PPL matches v17.4 baseline |
| IT-13 | `test_ablation_no_dense_am` | --no-dense-am removes Dense AM | Train with flag | Dense AM energy = 0; linear n-gram used instead |
| IT-14 | `test_ablation_no_reservoir` | --no-reservoir removes reservoir | Train with flag | Reservoir energy = 0 |
| IT-15 | `test_ablation_no_rff` | --no-rff removes RFF | Train with flag | RFF energy = 0 |

---

## 5. Level 4: End-to-End Tests

### 5.1 Full Pipeline Tests (run on development machine)

| ID | Test Name | What It Verifies | Duration Target | Assertions |
|----|-----------|-------------------|-----------------|------------|
| E2E-01 | `test_train_small_corpus` | Train on 1000 texts, all 14+ steps complete | < 5 min | Model builds; all components initialized; no errors |
| E2E-02 | `test_train_medium_corpus` | Train on 50K texts | < 30 min | PPL computed; finite value; generation works |
| E2E-03 | `test_beta_sweep_v18` | Beta sweep finds optimal beta | < 10 min | Best beta found; PPL varies with beta |
| E2E-04 | `test_generate_400_tokens` | Generate 400-token sample | < 30 sec | 400 tokens generated; text is non-empty |
| E2E-05 | `test_ppl_improvement_v18_vs_v17` | v18 PPL is better than v17 | < 1 hr | `PPL_v18 < PPL_v17` on same test set |
| E2E-06 | `test_memory_under_budget` | Training stays within 12 GB RSS | Full training | `RSS < 12 GB` at all steps |

---

## 6. Level 5: Acceptance Tests (Pi 5 Only)

### 6.1 Pi 5 Deployment Validation

| ID | Test Name | What It Verifies | Target | Measurement |
|----|-----------|-------------------|--------|-------------|
| AT-01 | Pi 5 training completes | Full training on Pi 5 without OOM | RSS < 12 GB | `/proc/pid/status` VmRSS |
| AT-02 | Pi 5 generation latency | Per-token generation speed | < 13 ms/tok | `time.perf_counter()` per token |
| AT-03 | Pi 5 generation throughput | Tokens per second | > 75 tok/s | 400 tokens / wall clock |
| AT-04 | Pi 5 PPL target | PPL meets target | < 8 (target <5) | 100-sequence evaluation |
| AT-05 | Pi 5 generation coherence | Qualitative quality | < 20 copied n-grams/400tok | Automated n-gram copy count |
| AT-06 | Pi 5 integer-only verification | No float in hot path | Zero float calls | `grep -c "float\|math.exp\|math.log" *.py` on core modules |
| AT-07 | Pi 5 400-token generation | Full 400-token generation from 5 prompts | At least 3/5 samples have coherent transitions | Human review + automated metrics |
| AT-08 | Pi 5 stress test | 10 consecutive 400-token generations | No crash, no memory leak | RSS stable across runs |

---

## 7. Regression Test Suite (`tests/test_v18_regression.py`)

### 7.1 v17 Baseline Preservation

| ID | Test | What Must Not Break |
|----|------|---------------------|
| RT-01 | KN backoff energy computed correctly | KN backoff gives lower energy than unigram for continuation words |
| RT-02 | Interpolated smoothing works | Interpolated energy ≤ longest-only energy |
| RT-03 | Context weight capping at 16 | 10-gram match weight = 16 (not 512) |
| RT-04 | Sentence boundary `<S>` prevents cross-sentence n-grams | No n-gram spans across `<S>` tokens |
| RT-05 | Multi-type candidate assignment | "run" appears in both NOUN and VERB type buckets |
| RT-06 | Tokenizer consistency | `vocab.encode()` matches `vocab._tokenize()` |
| RT-07 | Same-word penalty applies | `E(w_prev) += same_word_penalty` |
| RT-08 | Closed-class run limit works | After 2 closed-class words, next is open-class |
| RT-09 | PPL computation is valid | `PPL > 1` and finite |
| RT-10 | Boltzmann sampler handles wide energy range | `max_delta=50000` works without overflow |

---

## 8. Performance Test Suite (`tests/test_v18_performance.py`)

### 8.1 Latency Benchmarks

| ID | Benchmark | Target | Measurement |
|----|-----------|--------|-------------|
| PERF-01 | VSA similarity (512 dims, 49K candidates) | < 5 ms | Timer per token |
| PERF-02 | Dense AM energy (256 dims, 49K candidates) | < 2.5 ms | Timer per token |
| PERF-03 | Reservoir update + readout (512 dims, 49K candidates) | < 5 ms | Timer per token |
| PERF-04 | RFF energy (256 dims, 49K candidates) | < 0.5 ms | Timer per token |
| PERF-05 | Mean-field state coupling (5 iters, 7 vars) | < 0.1 ms | Timer per token |
| PERF-06 | Total v18 energy computation | < 13 ms | Timer per token |
| PERF-07 | Full generation loop (energy + sampling + state update) | < 15 ms | Timer per token |

### 8.2 Memory Benchmarks

| ID | Benchmark | Target | Measurement |
|----|-----------|--------|-------------|
| MEM-01 | VSA readout matrix | < 25 MB | `R.nbytes` |
| MEM-02 | Dense AM Phi matrix | < 25 MB | `Phi.nbytes` |
| MEM-03 | Reservoir readout matrix | < 50 MB | `R.nbytes` |
| MEM-04 | RFF Theta matrix | < 15 MB | `Theta.nbytes` |
| MEM-05 | Total model size (all matrices) | < 200 MB | Sum of all `.nbytes` |
| MEM-06 | Peak RSS during training | < 12 GB | `/proc/pid/status` |
| MEM-07 | Peak RSS during inference | < 2 GB | `/proc/pid/status` |

---

## 9. Test Execution Plan

### 9.1 Continuous (Every Commit)

```bash
# Level 1 + Level 2: Unit + Property tests (< 2 min)
pytest tests/test_qfhrr.py tests/test_dense_am.py tests/test_integer_esn.py \
       tests/test_rff.py tests/test_factorial_state.py tests/test_v18_property.py \
       -v --timeout=120
```

### 9.2 Pre-Merge (Before PR to main)

```bash
# Level 1 + 2 + 3: + Integration tests (< 10 min)
pytest tests/test_v18_integration.py tests/test_v18_regression.py -v --timeout=600
```

### 9.3 Phase Gate (Before version tag)

```bash
# Level 1-4: + End-to-end tests (< 1 hour)
pytest tests/test_v18_integration.py tests/test_v18_regression.py \
       tests/test_v18_performance.py -v --timeout=3600
python train_v18.py --samples 50000  # Full E2E training run
```

### 9.4 Release (Before Pi 5 deployment)

- Run all Level 1-5 tests
- Manual human evaluation of 400-token generation samples
- Document PPL, latency, memory metrics

---

## 10. Test Infrastructure

### 10.1 Test Fixtures

```python
# conftest.py — shared fixtures for all v18 tests

@pytest.fixture
def small_vocab():
    """Vocabulary with 500 words from 100 short texts."""
    ...

@pytest.fixture
def small_model(small_vocab):
    """Fully trained v18 model with 500-word vocab on 1000 texts."""
    ...

@pytest.fixture
def vsa_encoder(small_vocab):
    """VSA encoder with D=512 for the small vocab."""
    ...

@pytest.fixture
def random_context():
    """Random context of 10 word IDs for energy testing."""
    ...
```

### 10.2 Property-Based Testing with Hypothesis

```python
from hypothesis import given, strategies as st

@given(
    dim=st.integers(min_value=64, max_value=1024),
    n_vectors=st.integers(min_value=2, max_value=100),
)
def test_vsa_similarity_bounded(dim, n_vectors):
    """Similarity of random vectors is within expected bounds."""
    ...
```

### 10.3 Regression Test Automation

```python
# Store v17.4 baseline PPL for comparison
V17_BASELINE_PPL = 19.19

def test_v18_ppl_not_worse_than_v17():
    """v18 PPL should not be worse than v17.4 baseline."""
    model = train_small_v18()
    ppl = model.compute_perplexity(n_samples=10)
    # Allow 50% regression for small corpus; strict for full
    assert ppl < V17_BASELINE_PPL * 1.5
```

---

## 11. Success Criteria Summary

| Metric | v17.4 Baseline | v18.0 Target | v18.3 Target | Test Level |
|--------|---------------|-------------|-------------|------------|
| PPL (100-seq) | 21.11 | < 15 | < 8 | AT-04 |
| PPL (best beta) | 19.19 | < 13 | < 5 | AT-04 |
| Copied n-grams/400tok | 254 | < 80 | < 20 | AT-05 |
| Cross-sentence transitions | 0-1 | 2-3 | 5+ | AT-07 |
| Memory (RSS training) | 10 GB | < 12 GB | < 12 GB | AT-01 |
| Per-token latency (Pi 5) | ~1 ms | ~6 ms | ~13 ms | AT-02 |
| Tokens/sec (Pi 5) | ~800 | ~160 | ~75 | AT-03 |
| Integer-only inference | Yes | Yes | Yes | AT-06 |
