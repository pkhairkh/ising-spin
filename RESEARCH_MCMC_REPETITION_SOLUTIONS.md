# Principled Mathematical Solutions for MCMC Repetition/Trapping in Discrete Ising-Potts Models

## Research Report — Integer-Only Arithmetic Constraint

**Problem**: Coupled Ising-Potts model for text generation (V=3000 word vocabulary, ~20-30 positions) gets trapped in repetitive states. The sampler falls into local energy minima where the same word persists at a position across sweeps, and the Metropolis-within-Gibbs dynamics lacks the energy to escape.

**Constraint**: All generation-path computation must use integer-only arithmetic. No floating-point operations during sampling.

**Current Workarounds to Replace**:
- Self-repulsion (ad-hoc penalty for repeated words)
- History-driven target (tracking visited states and penalizing)
- Momentum (tracking direction of word index changes)
- Anti-ferromagnetic self-coupling (J[i,i] < 0)

---

## Diagnosis: Why Does Repetition Occur?

### The Physics of the Trap

Your model has the Hamiltonian:

```
E(state) = -Σ_{i<j} J[w_i, w_j] - Σ_i h[i, w_i] - Σ_i I_emit[w_i, t_i] + penalties
```

The repetition problem has three root causes identifiable from spin glass theory:

**1. Rugged Energy Landscape with Multiple Metastable States**

For a 20-position sequence with V=3000, the state space is 3000^20 ≈ 10^69. The coupling matrix J derived from PMI/co-occurrence creates a **frustrated** system: word w_i may have strong positive coupling to w_j but negative coupling to w_k, while w_j also couples positively to w_k. This frustration creates an exponential number of local energy minima separated by barriers.

In spin glass terminology, your system exhibits **replica symmetry breaking** — there is no single "funnel" leading to a global minimum, but rather a hierarchical landscape of valleys within valleys (Parisi, 1980; Mezard, Parisi & Virasoro, 1987). The 1-step replica symmetry breaking (1RSB) picture applies: the state space partitions into many pure states, and transitions between them require crossing energy barriers that scale with system size.

**2. Single-Site Update Dynamics Cannot Cross Barriers**

Your Metropolis-within-Gibbs sampler updates one position at a time (single-spin-flip dynamics). When word w_i at position i is in a local minimum, changing it to any alternative word w' requires:

```
ΔE = E(w_i → w') = [coupling terms with neighbors] + [field change] + [emission change]
```

If w_i has strong positive couplings to its neighbors (which is the typical case for a local minimum), ALL alternatives have ΔE > 0, often by large amounts (hundreds to thousands in your integer scale). At β=1000 (your phase 3 temperature), acceptance probability for ΔE=500 is:

```
P(accept) ≈ exp(-500 × 1.0) ≈ 10^{-217}
```

This is effectively zero. The chain is **metastable** — it will eventually escape (ergodicity is preserved) but the escape time is astronomically large (exponential in ΔE × β).

**3. Proposal Distribution Mismatch**

Your proposal strategy (40% field-based, 40% coupling-neighbor, 20% uniform) concentrates proposals near the current state. But the "exit paths" from a local minimum may require jumping to a distant, seemingly unrelated word. The proposal distribution systematically misses these escape routes.

### Quantitative Diagnosis

From your code, the energy scale is:
- Coupling values: J entries up to ~100 (scaled counts)
- Field values: h entries up to ~50
- Emission bonuses: up to emit_val × 100
- Emission penalties: -500 for incompatible types
- Grammar penalties: up to -10000 for forbidden transitions

The total energy landscape has barriers of order 500-5000 between metastable states. At your cold-chain temperature (β=1000), barrier crossings are effectively impossible via single-site Metropolis.

---

## Analysis of Current Workarounds (Why They're Problematic)

### History-Driven Target (Hu et al. 2025)
- **Problem**: Violates detailed balance. The energy function depends on the trajectory, not just the current state. This means the stationary distribution is **not** the Boltzmann distribution — you're sampling from an unknown distribution.
- **Problem**: The quadratic penalty `min(count² × 5, 5000)` is ad-hoc. There's no principled way to choose the decay factor or the cap.
- **Problem**: Requires O(L × V) memory for visit counts and O(L) computation per sweep for decay.

### Momentum Tracker
- **Problem**: Breaks detailed balance entirely. The proposal distribution depends on trajectory history.
- **Problem**: "Direction" in word-index space is meaningless — word index 42 and word index 43 are not "close" in any semantic sense.
- **Problem**: This is essentially a heuristic from continuous optimization (momentum/SGD) misapplied to discrete spaces.

### Anti-Ferromagnetic Self-Coupling (J[i,i] < 0)
- **Problem**: Modifies the energy function itself, so you're sampling from a **different** model. The generated text no longer reflects the learned PMI statistics.
- **Problem**: May create artificial oscillations (word keeps flipping between two states).

### Self-Repulsion
- **Problem**: Same as anti-ferromagnetic coupling — changes the model, not just the sampler.

---

## TOP 5 PRINCIPLED SOLUTIONS

### ═══════════════════════════════════════════════════════════
### SOLUTION 1: Swendsen-Wang / Wolff Cluster Algorithms for Potts Models
### ═══════════════════════════════════════════════════════════

**Rank**: ⭐⭐⭐⭐⭐ (HIGHEST PRIORITY — most principled, most impactful)

#### Mathematical Principle

The key insight (Fortuin & Kasteleyn, 1972; Swendsen & Wang, 1987; Wolff, 1989) is the **FK random-cluster representation** of the Potts model. For a q-state Potts model with Hamiltonian:

```
H = -J Σ_{<i,j>} δ(s_i, s_j)
```

The partition function factorizes as:

```
Z = Σ_{states} exp(βJ Σ δ(s_i,s_j)) = Σ_{bonds ⊆ E} p^|bonds| (1-p)^|E\bonds| q^{C(bonds)}
```

where p = 1 - exp(-βJ) is the bond activation probability and C(bonds) is the number of connected components.

This means: instead of flipping individual spins, we can:
1. **Activate bonds**: For each pair of neighboring sites with the same spin value, add a "bond" with probability p = 1 - exp(-βJ)
2. **Identify clusters**: Find connected components of the bond graph
3. **Flip clusters**: Assign each cluster a NEW random spin value uniformly from {1,...,q}

This makes global moves that preserve detailed balance and can cross energy barriers in a single step.

#### Why This Solves Repetition

In your problem, when word w appears at positions i and j, and J[w,w] > 0 (same-word coupling), the FK bond between (i,j) is activated with probability p = 1 - exp(-β × J[w,w]). The entire cluster of positions containing word w gets reassigned simultaneously to a new word. **The repetition is broken because the cluster gets a new word value, not because of any repulsion — it's the natural dynamics of the model.**

For the Ising-Potts language model, the extension is:
- The "same spin" condition becomes "same word" or "compatible types"
- The bond probability p_ij depends on J[w_i, w_j], not just a uniform J
- The cluster flip assigns a new word from the emission-compatible set

#### Integer-Only Implementation

**Precomputation (one-time, FP allowed)**:
```python
# For each possible coupling value J_val, precompute:
# p = 1 - exp(-beta * J_val / 1000)
# threshold[J_val] = int(2^31 * p)
# This is a lookup table: same as your existing ProbabilityTable
cluster_thresholds = {}
for j_val in range(-max_J, max_J + 1):
    if j_val > 0:
        p = 1.0 - math.exp(-j_val * beta / 1000.0)
        cluster_thresholds[j_val] = int((2**31 - 1) * p)
    else:
        cluster_thresholds[j_val] = 0  # No bond for non-positive coupling
```

**Generation loop (pure integer)**:
```python
def swendsen_wang_sweep(state_words, state_types, J_combined, cluster_thresholds):
    L = len(state_words)
    
    # Step 1: Activate bonds (INTEGER: compare random int to threshold)
    # Union-Find data structure for clusters
    parent = list(range(L))
    
    for i in range(L):
        for j_offset in range(1, window + 1):
            j = i + j_offset
            if j >= L:
                break
            
            # Bond activation: same-type neighbors with positive coupling
            coupling = int(J_combined[state_words[i], state_words[j]])
            if coupling > 0 and state_types[i] == state_types[j]:
                threshold = cluster_thresholds.get(coupling, 0)
                rand_val = random.randint(0, 2**31 - 2)
                if rand_val < threshold:
                    union(parent, i, j)  # Merge clusters (integer ops only)
    
    # Step 2: Identify clusters (pure integer graph traversal)
    clusters = find_clusters(parent, L)  # Dict: root -> [positions]
    
    # Step 3: Flip each cluster to a new random word (INTEGER sampling)
    for root, positions in clusters.items():
        if len(positions) == 1:
            # Single-site: use regular Metropolis for efficiency
            continue
        
        # Sample new word for this cluster from emission-compatible set
        # Use precomputed emission cumsum (same as existing code)
        current_type = state_types[positions[0]]
        new_word = sample_from_emission_cumsum(current_type)
        new_type = get_type_for_word(new_word)
        
        # Accept/reject entire cluster flip using integer energy comparison
        delta_e = compute_cluster_energy_change(
            positions, state_words, state_types, new_word, new_type, J_combined
        )
        rand_val = random.randint(0, 2**31 - 2)
        if accept(delta_e, rand_val, prob_table):
            for pos in positions:
                state_words[pos] = new_word
                state_types[pos] = new_type
```

**Union-Find (all integer)**:
```python
def find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]  # Path compression
        i = parent[i]
    return i

def union(parent, i, j):
    ri, rj = find(parent, i), find(parent, j)
    if ri != rj:
        parent[ri] = rj
```

#### Detailed Balance

**YES, detailed balance is preserved** (Swendsen & Wang, 1987). The proof relies on the FK representation: the bond activation step creates the correct conditional distribution over clusters, and the cluster reassignment step is symmetric. The composite transition kernel satisfies detailed balance with respect to the Potts Boltzmann distribution.

For the heterogeneous coupling case (your problem, where J[w_i, w_j] varies), the proof still holds as long as:
- Bond probability p_ij = 1 - exp(-β × J[w_i, w_j]) depends only on the current state
- The new spin assignment is symmetric (uniform or proportional to field weights)

#### How It Addresses Repetition

1. **Cluster flips break correlations**: If positions 3, 7, 15 all have word "the" and are coupled, they form a cluster that gets reassigned together. No more individual-site trapping.

2. **Non-local moves**: A cluster can span the entire sequence, enabling barrier crossings that would require O(L) single-site flips.

3. **Automatic adaptation**: In ordered phases (low T), clusters are large → big moves. In disordered phases (high T), clusters are small → appropriate exploration. The algorithm self-tunes its move size.

4. **No ad-hoc parameters**: The only parameter is β, which is already part of the model.

#### Computational Cost

- **Bond activation**: O(L × window) integer comparisons — same order as current sweep
- **Union-Find**: O(L × α(L)) where α is inverse Ackermann ≈ 1
- **Cluster identification**: O(L)
- **Cluster energy computation**: O(|cluster| × window) per cluster

**Total**: O(L × window) — **same asymptotic cost as current Metropolis-within-Gibbs**, with a small constant factor overhead for Union-Find operations.

For V=3000, L=20, window=5: about 100 bond checks, 20 union-find operations, ~5-10 clusters per sweep. **Negligible overhead.**

#### Caveat for Heterogeneous Systems

The standard SW algorithm is exact for homogeneous Potts models (uniform J). For your heterogeneous J (different J[w_i, w_j] values), the bond probability varies per edge, but the algorithm remains valid — you just use p_ij = 1 - exp(-β × J_ij) instead of uniform p. This is the **heterogeneous Swendsen-Wang** algorithm (Edwards & Sokal, 1988).

However, for models with strong disorder (your J has high variance), cluster sizes may be small, reducing effectiveness. In this case, the **Wolff single-cluster** variant is preferable:
1. Pick a random site i
2. Grow one cluster from i using FK bond activation
3. Flip only that one cluster
4. Repeat

The Wolff variant is more efficient for disordered systems because it focuses computational effort on the largest clusters.

---

### ═══════════════════════════════════════════════════════════
### SOLUTION 2: Wang-Landau Flat-Histogram Sampling (Modified Density of States)
### ═══════════════════════════════════════════════════════════

**Rank**: ⭐⭐⭐⭐ (High — fundamentally changes the sampling distribution)

#### Mathematical Principle

Wang-Landau sampling (Wang & Landau, Phys. Rev. Lett. 2001; Phys. Rev. E 2001) replaces the Boltzmann distribution with the **microcanonical** distribution. Instead of sampling P(state) ∝ exp(-βE), it samples uniformly over energy levels:

```
P(E) = 1/g(E) × g(E) = constant for all E
```

where g(E) is the density of states (number of configurations with energy E).

The algorithm works by:
1. Maintaining an estimator ĝ(E) of the density of states
2. Visiting energy E → multiply ĝ(E) by a modification factor f > 1
3. Accepting moves with probability min(1, ĝ(E_old) / ĝ(E_new))
4. When the energy histogram is "flat" (all bins visited roughly equally), reduce f → √f
5. Continue until f ≈ 1 + 10^{-8}

The mathematical theorem (proven by Zhou & Bhatt, 2005; Belardinelli & Pereyra, 2007) shows convergence: ĝ(E) → g(E) × constant as f → 1.

#### Why This Solves Repetition

In Boltzmann sampling at low T, the chain is trapped because P(E_trap) >> P(E_other) for nearby energies. Wang-Landau **flattens** the energy distribution: it visits ALL energy levels equally, including those separated by barriers. The acceptance probability:

```
P(accept) = min(1, ĝ(E_old) / ĝ(E_new))
```

is independent of the Boltzmann weight. A transition from a low-energy trap to a higher-energy state is accepted with probability ĝ(E_low)/ĝ(E_high). Since g(E) is typically exponentially increasing with E (there are many more high-energy states), this ratio is often > 1, meaning **barrier crossings are enhanced, not suppressed**.

#### Integer-Only Implementation

The key challenge: ĝ(E) must be stored as integers, and the ratio ĝ(E_old)/ĝ(E_new) must be computed without FP.

**Solution: Log-space integer representation**

Instead of storing ĝ(E) directly, store log₂(ĝ(E)) as an integer (fixed-point):

```python
class WangLandauTable:
    """
    Integer-only Wang-Landau density of states estimator.
    
    Stores S(E) = round(log2(g(E)) × PRECISION) as integers.
    This avoids overflow (g(E) can be astronomically large)
    and allows ratio computation via integer subtraction.
    """
    
    def __init__(self, E_min, E_max, precision=1000):
        self.precision = precision  # Fixed-point scaling for log values
        self.E_min = E_min
        self.E_max = E_max
        n_bins = E_max - E_min + 1
        
        # S[E] = round(log2(g(E)) × precision)
        # Initialize: S[E] = 0 means g(E) = 1
        self.S = [0] * n_bins
        
        # Histogram: count visits to each energy bin
        self.histogram = [0] * n_bins
        
        # Log modification factor: f_bits = round(log2(f) × precision)
        # Start with f = e ≈ 2.718, so log2(f) ≈ 1.443
        self.f_bits = int(1.443 * precision)  # ≈ 1443 for precision=1000
        self.min_f_bits = 1  # Stop when f_bits < 1 (f < 2^(1/1000) ≈ 1.001)
        
        # Flatness criterion: all histogram bins > flatness_ratio × mean
        self.flatness_ratio = 80  # 80% of mean (standard: 80%)
    
    def energy_to_idx(self, E):
        return E - self.E_min
    
    def modify(self, E):
        """Visit energy E: g(E) *= f. In log-space: S(E) += f_bits."""
        idx = self.energy_to_idx(E)
        if 0 <= idx < len(self.S):
            self.S[idx] += self.f_bits
            self.histogram[idx] += 1
    
    def accept_ratio(self, E_old, E_new):
        """
        Compute acceptance ratio g(E_old)/g(E_new) as integer.
        
        In log-space: log2(g(E_old)/g(E_new)) = S(E_old) - S(E_new)
        
        Return: (numerator_bits, denominator_bits) representing
        2^numerator_bits / 2^denominator_bits = g(E_old)/g(E_new)
        
        Actually, we return: delta_S = S(E_old) - S(E_new)
        If delta_S >= 0: accept (ratio >= 1)
        If delta_S < 0: accept with probability 2^(delta_S / precision)
        """
        idx_old = self.energy_to_idx(E_old)
        idx_new = self.energy_to_idx(E_new)
        
        if not (0 <= idx_old < len(self.S) and 0 <= idx_new < len(self.S)):
            return True  # Out-of-range energies: always accept
        
        delta_S = self.S[idx_old] - self.S[idx_new]
        
        if delta_S >= 0:
            return True  # g(E_old) >= g(E_new): always accept
        
        # Accept with probability 2^(delta_S / precision)
        # This is: exp(delta_S / precision × ln(2))
        # = exp(delta_S_int × ln2 / precision)
        # = exp(delta_S_int / (precision / ln2))
        # = exp(delta_S_int / precision_fp)
        
        # Using precomputed threshold table (same technique as existing ProbabilityTable):
        # For each delta_S value, threshold[delta_S] = int(2^31 × 2^(delta_S/precision))
        # This is PRECOMPUTED once, then used as integer comparison
        
        # For integer-only: use the fact that 2^(-n/precision) ≈ 
        # precomputed_threshold[n] where n = -delta_S (positive)
        accept_threshold = self._get_threshold(-delta_S)
        rand_val = random.randint(0, 2**31 - 2)
        return rand_val < accept_threshold
    
    def _get_threshold(self, neg_delta_S):
        """
        Get precomputed threshold for 2^(-neg_delta_S / precision).
        
        Uses lookup table built at initialization.
        threshold[k] = int((2^31 - 1) × 2^(-k / precision))
        
        Since 2^(-k/precision) decreases with k, we only need a table
        up to some max_k where 2^(-max_k/precision) is negligible.
        """
        if neg_delta_S <= 0:
            return 2**31 - 1
        if neg_delta_S >= len(self.threshold_table):
            return 0
        return self.threshold_table[neg_delta_S]
    
    def is_flat(self):
        """Check if energy histogram is flat (integer arithmetic)."""
        total = sum(self.histogram)
        if total == 0:
            return False
        mean = total // len(self.histogram)
        if mean == 0:
            return False
        
        min_hist = min(h for h in self.histogram)
        # Flat if min_hist >= flatness_ratio% × mean
        return min_hist * 100 >= self.flatness_ratio * mean
    
    def reduce_f(self):
        """Reduce modification factor: f → sqrt(f). In log-space: f_bits //= 2."""
        self.f_bits = max(self.min_f_bits, self.f_bits // 2)
        self.histogram = [0] * len(self.histogram)  # Reset histogram
    
    def is_converged(self):
        return self.f_bits < self.min_f_bits
```

**Integration with existing sampler**:

```python
def wang_landau_sweep(state_words, state_types, wl_table, J_combined, h, I_emit):
    """One sweep of Wang-Landau sampling — PURE INTEGER."""
    L = len(state_words)
    
    for pos in range(L):
        current_word = state_words[pos]
        current_energy = compute_word_energy(pos, current_word, ...)
        
        # Propose new word (same proposal mechanism as current code)
        proposed_word = propose_word(pos, current_word, ...)
        proposed_energy = compute_word_energy(pos, proposed_word, ...)
        
        # Wang-Landau acceptance: min(1, g(E_old)/g(E_new))
        # NOT Boltzmann — no beta involved!
        if wl_table.accept_ratio(current_energy, proposed_energy):
            state_words[pos] = proposed_word
            new_energy = proposed_energy
        else:
            new_energy = current_energy
        
        # Update density of states estimator
        wl_table.modify(new_energy)
    
    # Check flatness and reduce f if needed
    if wl_table.is_flat():
        wl_table.reduce_f()
```

**Post-processing: Reweighting to Boltzmann**

After Wang-Landau converges, you have ĝ(E) for all energies. To generate samples at temperature T:

```
P_T(state) ∝ g(E(state)) × exp(-βE(state))
           = exp(S(E) × ln(2) / precision - βE)
```

In integer arithmetic:
```python
def boltzmann_weight(energy, wl_table, beta_int):
    """Compute integer Boltzmann weight from Wang-Landau g(E)."""
    idx = wl_table.energy_to_idx(energy)
    S_E = wl_table.S[idx]  # log2(g(E)) × precision
    
    # Weight = g(E) × exp(-beta × E)
    # log2(weight) = S_E/precision - beta × E / ln(2)
    # In fixed point: = S_E - E × beta_int × precision_ln2 / 1000
    
    # This gives relative weights for different energies
    # Use to construct a new probability table
    log2_weight = S_E - energy * beta_int * 693 // 1000  # ln(2) ≈ 0.693
    return log2_weight
```

#### Detailed Balance

**YES, in the limit f → 1** (Wang & Landau, 2001; Zhou & Bhatt, 2005). During the learning phase (f > 1), detailed balance is **approximately** satisfied — the error decreases as f decreases. This is the main theoretical weakness: the samples during learning are biased. The standard practice is to:
1. Run Wang-Landau until convergence (learn g(E))
2. Then sample using the converged g(E) with exact detailed balance

For your application, you would:
1. **Precompute** g(E) from training data (one-time, can use FP)
2. **Use** the precomputed g(E) for sampling with detailed balance

#### How It Addresses Repetition

1. **Flat histogram**: By construction, the chain visits all energy levels equally. Low-energy traps are no more likely to be visited than high-energy states.

2. **Enhanced barrier crossing**: The acceptance ratio g(E_old)/g(E_new) typically **favors** transitions to high-energy states (since g(E) grows exponentially with E), which are precisely the barrier-crossing transitions needed to escape local minima.

3. **Post-processing flexibility**: Once g(E) is known, you can sample at any temperature by reweighting. This means you don't need to choose β carefully — you can compute the optimal temperature after the fact.

#### Computational Cost

- **Precomputation** (learning g(E)): O(n_WL × L × V_proposal) where n_WL is the number of WL sweeps needed for convergence. For your problem size (L=20, V=3000), this is typically 10^4 - 10^5 sweeps. **This is a one-time cost.**

- **Generation** (using converged g(E)): O(L × V_proposal) — **same as current Metropolis**, with one extra table lookup per step for the g(E) ratio.

**Overhead**: ~1 integer table lookup + 1 integer addition per step, compared to current code. **Negligible.**

#### Key Challenge

The energy range must be binned appropriately. Your total energy can range over O(10^5) values, so you need ~10^5 bins in the histogram. This is ~400KB of memory — trivial.

---

### ═══════════════════════════════════════════════════════════
### SOLUTION 3: Proper Parallel Tempering with Optimal Temperature Ladder
### ═══════════════════════════════════════════════════════════

**Rank**: ⭐⭐⭐⭐ (High — you already have PT, but the implementation needs fixing)

#### Mathematical Principle

Parallel tempering (Geyer, 1991; Hukushima & Nemoto, 1996) runs M replicas at different temperatures β_1 < β_2 < ... < β_n and periodically proposes swaps between adjacent replicas:

```
P(swap) = min(1, exp((β_j - β_i) × (E_j - E_i)))
```

The mathematical result (Earl & Deem, 2005; Katzgraber et al., 2006) is that the **mixing time** of the cold chain is bounded by:

```
τ_cold ≤ τ_hot × (product over i of 1/α_i)
```

where α_i is the swap acceptance rate between replicas i and i+1. For optimal mixing, all α_i should be equal (the "optimal ladder" condition).

**Critical insight**: If swap rates are too low (< 20%), the hot replicas are effectively disconnected from the cold one, and PT provides no benefit. Your current implementation with 4 replicas and arithmetic beta spacing is almost certainly suboptimal.

#### Problems with Current Implementation

From your code (`_pt_swap`):
1. **The swap criterion uses `math.exp()` at runtime** — this violates the integer-only constraint!
2. **Arithmetic beta spacing** `[beta_hot, (beta_hot+beta_cold)//3, 2*(beta_hot+beta_cold)//3, beta_cold]` is wrong. Temperature ladders should be **geometrically spaced** in β (or more precisely, tuned to equalize swap rates).
3. **Only 4 replicas** — for a frustrated system with V=3000, you likely need 8-16 replicas.
4. **Replica initialization** perturbs only 1/3 of positions — should use independent hot-chain equilibriation.

#### Integer-Only Implementation of PT Swaps

```python
class ParallelTempering:
    """
    Proper integer-only parallel tempering with optimal ladder.
    """
    
    def __init__(self, betas, max_delta_e=5000, rand_max=2**31-1):
        self.betas = betas  # List of integer betas (×1000)
        self.n_replicas = len(betas)
        
        # Precompute swap acceptance tables
        # For each pair (beta_i, beta_j) and energy difference dE,
        # swap_threshold[(beta_i, beta_j)][dE] = int(rand_max × exp((beta_j-beta_i)*dE/1000))
        # This is precomputed ONCE — no FP at runtime
        
        self.swap_tables = {}
        for i in range(self.n_replicas - 1):
            beta_diff = self.betas[i+1] - self.betas[i]  # Integer
            # dE ranges from -max_dE to max_dE
            table = [0] * (2 * max_delta_e + 1)
            for dE in range(-max_delta_e, max_delta_e + 1):
                idx = dE + max_delta_e
                # exp(beta_diff * dE / 1000) — precomputed
                exponent = beta_diff * dE / 1000.0  # FP only in precomputation!
                if exponent >= 0:
                    table[idx] = rand_max  # Always accept
                else:
                    prob = math.exp(exponent)
                    table[idx] = max(0, min(rand_max, int(rand_max * prob)))
            self.swap_tables[i] = table
    
    def attempt_swap(self, i, E_i, E_j, rand_val):
        """
        Attempt swap between replica i and replica i+1.
        PURE INTEGER: one table lookup + one comparison.
        """
        table = self.swap_tables[i]
        max_delta_e = (len(table) - 1) // 2
        dE = E_j - E_i
        
        if dE <= -max_delta_e:
            return True
        if dE >= max_delta_e:
            return False
        
        return rand_val < table[dE + max_delta_e]
```

**Optimal Temperature Ladder** (Katzgraber et al., 2006):

For a spin glass with N sites and coupling variance σ_J, the optimal geometric spacing is:

```
β_k = β_1 × (β_n / β_1)^((k-1)/(n-1))
```

But this must be **adaptively tuned** to achieve ~23% swap rate (the optimal value for tunneling). The integer implementation:

```python
def build_adaptive_ladder(beta_min, beta_max, n_replicas, target_swap_rate=23):
    """
    Build temperature ladder using adaptive method.
    Start with geometric spacing, then adjust based on measured swap rates.
    """
    # Initial geometric ladder
    betas = []
    for k in range(n_replicas):
        frac = k / (n_replicas - 1)
        # In integer: beta = beta_min * (beta_max/beta_min)^frac
        # Use precomputed log table
        log_ratio = math.log(beta_max / beta_min)
        beta = int(beta_min * math.exp(frac * log_ratio))
        betas.append(beta)
    betas[0] = beta_min
    betas[-1] = beta_max
    
    return betas

def adapt_ladder(betas, swap_rates, target=0.23):
    """
    Adapt ladder based on measured swap rates.
    If swap rate between i and i+1 is too low, insert a new beta between them.
    If too high, remove the middle beta.
    
    All integer arithmetic for the adaptation.
    """
    new_betas = [betas[0]]
    for i in range(len(betas) - 1):
        rate = swap_rates[i]  # Integer: (swaps × 100) / attempts
        if rate < target * 80:  # Too low (< 18.4%)
            # Insert intermediate beta (integer interpolation)
            mid_beta = (betas[i] + betas[i+1]) // 2
            new_betas.append(mid_beta)
        new_betas.append(betas[i+1])
    return new_betas
```

#### How It Addresses Repetition

1. **Hot replicas explore freely**: At β_hot (e.g., 50), the chain can cross barriers easily. States discovered by hot replicas get transferred to cold replicas via swaps.

2. **Tunneling**: The mixing time for tunneling through a barrier of height Δ is O(β_hot/β_cold × exp(-β_hot × Δ)) instead of O(exp(-β_cold × Δ)). This is exponentially faster.

3. **Proper ergodicity**: Even if the cold chain is stuck, the hot chain's equilibrium distribution covers the entire state space, and swaps ensure the cold chain eventually visits all states.

#### Computational Cost

- **Per-sweep cost**: O(n_replicas × L × V_proposal) — n_replicas times the cost of one Metropolis sweep
- **Swap cost**: O(n_replicas) integer comparisons per swap attempt
- **Total**: O(n_replicas × base_cost)

For n_replicas=8: **8× the cost of a single-chain run**. However, this is **much more efficient** than running 8 independent chains, because the swap mechanism provides the tunneling capability.

**Compared to current code**: Your current 4-replica implementation already pays this cost but gets suboptimal results. The fix is not more computation — it's **correct** temperature ladder and **proper** swap acceptance (integer precomputed, not runtime FP).

---

### ═══════════════════════════════════════════════════════════
### SOLUTION 4: Entropy-Regularized Free Energy (F = E - T_ent × S_traj)
### ═══════════════════════════════════════════════════════════

**Rank**: ⭐⭐⭐⭐ (High — principled, novel, directly targets the cause)

#### Mathematical Principle

The repetition problem is fundamentally an **entropy deficit** in the trajectory distribution. Consider the chain's trajectory x₀, x₁, ..., x_T. The probability of repetition at position i is:

```
P(x_t^i = x_{t-1}^i = ... = x_{t-k}^i) → high when S(chain) is low
```

The principled solution from statistical mechanics is to replace the energy minimization with **free energy minimization**:

```
F(state) = E(state) - T_ent × S(state)
```

where S(state) is the **local entropy** — the number of accessible states nearby. This is the **maximum entropy** (MaxEnt) principle applied to the sampling trajectory.

**Key theorem** (from Jaynes, 1957; applied to MCMC by Habib et al., 2024):

The trajectory distribution that maximizes entropy subject to the constraint that the expected energy matches a target value is:

```
P(x₀,...,x_T) ∝ exp(-Σ_t [β × E(x_t) - γ × S(x_t)])
```

where γ = T_ent controls the entropy-regularization strength.

#### Local Entropy Computation (Integer-Only)

For a discrete system, the local entropy at state x is:

```
S(x) = log(|{y : d(x,y) ≤ r and E(y) ≤ E(x) + ΔE}|)
```

This counts the number of states within "distance" r of x that have energy within ΔE of x. For your problem:

- **Distance**: Hamming distance (number of positions with different words)
- **Radius**: r = 1 (single-word changes)
- **Energy window**: ΔE (tunable parameter)

**Integer computation**:
```python
class LocalEntropyEstimator:
    """
    Estimates local entropy S(x) = log2(number of accessible neighbors)
    using integer-only arithmetic.
    """
    
    def __init__(self, delta_E_window=500, precision=100):
        self.delta_E_window = delta_E_window
        self.precision = precision
    
    def compute_local_entropy(self, pos, current_word, current_energy, 
                               proposal_set, J_combined, h, prob_table):
        """
        Count accessible neighbors: proposals with |ΔE| ≤ delta_E_window.
        Returns S = log2(count) × precision (integer).
        """
        accessible_count = 0
        
        for proposed_word in proposal_set:
            if proposed_word == current_word:
                continue
            
            proposed_energy = compute_word_energy(pos, proposed_word, ...)
            delta_e = proposed_energy - current_energy
            
            # Count neighbors within energy window
            if abs(delta_e) <= self.delta_E_window:
                accessible_count += 1
        
        # S = log2(accessible_count) × precision
        # Using integer log2: find position of highest set bit
        if accessible_count <= 0:
            return 0
        
        # Integer log2: number of bits minus 1
        s_int = accessible_count.bit_length() - 1  # floor(log2(count))
        # Fractional part: use next bit for refinement
        if accessible_count > (1 << s_int):
            # log2(count) is between s_int and s_int + 1
            # Approximate: s_int + (count - 2^s_int) / 2^s_int
            # In fixed point: s_int * precision + (count - 2^s_int) * precision // 2^s_int
            frac = (accessible_count - (1 << s_int)) * self.precision // (1 << s_int)
            return s_int * self.precision + frac
        else:
            return s_int * self.precision
```

**Integration with sampler — the modified energy**:

```python
def entropy_regularized_energy(pos, word, base_energy, local_entropy, 
                                 T_ent_int=100):
    """
    Modified energy: F = E - T_ent × S
    
    In integer arithmetic:
    F = base_energy - (T_ent_int × local_entropy) // 1000
    
    T_ent_int is an integer temperature parameter (like beta_int).
    local_entropy is in fixed-point (× precision=100).
    
    The subtraction means: high-entropy states (many accessible neighbors)
    get LOWER free energy → preferred.
    Low-entropy states (few neighbors = trapped) get HIGHER free energy → avoided.
    """
    entropy_bonus = (T_ent_int * local_entropy) // 1000
    return base_energy - entropy_bonus
```

#### Why This Solves Repetition

1. **Trapped states have low entropy**: When a word is stuck, most alternatives have ΔE >> 0, so `accessible_count` is small, S is low, and F = E - T_ent × S is **higher** than E alone.

2. **Escape states have high entropy**: States near barriers (where many alternatives are accessible) get a free energy boost from the entropy term, making them **preferred** by the sampler.

3. **Self-regulating**: As T_ent → 0, you recover standard Boltzmann sampling. As T_ent → ∞, you get maximum-entropy sampling (uniform). The parameter T_ent smoothly interpolates.

4. **Principled**: This is exactly the free energy from thermodynamics. The temperature T_ent controls the trade-off between energy minimization (coherent text) and entropy maximization (diverse text).

#### Detailed Balance

**YES, if implemented correctly**. The key is that the modified energy F(x) = E(x) - T_ent × S(x) is a **state function** — it depends only on the current state, not on the trajectory. Therefore, Metropolis-within-Gibbs with energy F preserves detailed balance with respect to the modified Boltzmann distribution:

```
π(x) ∝ exp(-β × F(x)) = exp(-β × (E(x) - T_ent × S(x)))
                        = exp(-β × E(x)) × exp(β × T_ent × S(x))
```

This is a **well-defined** probability distribution. It's not the standard Boltzmann distribution — it's a maximum-entropy-penalized distribution — but it's a valid target.

**Important**: You are sampling from a **different distribution** than pure Boltzmann. This is a feature, not a bug: the Boltzmann distribution at low T concentrates on the ground state (repetitive text), while the entropy-regularized distribution spreads probability mass more broadly (diverse text).

#### Computational Cost

- **Local entropy computation**: O(|proposal_set|) energy evaluations per position per sweep. For proposal_set of size 30, this is 30 extra energy evaluations — **~2× the cost of current sampling**.

- **However**: The entropy computation can be **cached**. The accessible neighbor count only changes when the state changes, which happens on accepted moves only. For a typical acceptance rate of 10-30%, you can update incrementally.

**Optimization**: Instead of recomputing S(x) from scratch, maintain a counter of "accessible neighbors" that updates incrementally:
```python
# When word at position j changes:
# Recompute entropy only for positions within window of j
# This reduces cost to O(window × |proposal_set|) per accepted move
```

---

### ═══════════════════════════════════════════════════════════
### SOLUTION 5: Lifted/Non-Reversible MCMC (Diaconas-Neal Turitsyn Chain)
### ═══════════════════════════════════════════════════════════

**Rank**: ⭐⭐⭐ (Medium-High — elegant theory, but harder to apply to high-dimensional discrete spaces)

#### Mathematical Principle

The key result from the theory of non-reversible Markov chains (Diaconas, Holmes & Neal, 2000; Turitsyn, Chertkov & Vucelja, 2011; Vucelja, 2016) is:

**Lifting a reversible chain by adding a "momentum" variable can reduce mixing time by O(N) compared to the reversible chain, where N is the system size.**

The mathematical mechanism:
1. Double the state space: (x, σ) where x is the configuration and σ ∈ {+1, -1} is a "lift" direction
2. In state (x, +1), the chain proposes moves in the "positive" direction (increasing some coordinate)
3. In state (x, -1), the chain proposes moves in the "negative" direction
4. Direction flips occur with probability proportional to the "rejection" rate

The non-reversible chain satisfies **global balance** (but not detailed balance):
```
Σ_y T(x→y) π(x) = Σ_y T(y→x) π(y)
```

The mixing time improvement comes from the elimination of **diffusive behavior**. A reversible chain performs a random walk, which takes O(N²) steps to traverse a sequence of length N. A non-reversible (lifted) chain performs a **directed walk**, traversing in O(N) steps.

#### Application to Ising-Potts Language Model

For discrete word variables, "direction" must be defined carefully. The natural choice for your model:

**Direction = type transition direction**: Define σ_i ∈ {+1, -1} for each position i, indicating whether the word at position i is "transitioning toward" higher or lower type indices.

```python
class LiftedMCMC:
    """
    Lifted MCMC for the Ising-Potts language model.
    
    State: (words, types, directions)
    directions[i] ∈ {+1, -1} — momentum for position i
    
    The chain is non-reversible (no detailed balance),
    but satisfies global balance → correct stationary distribution.
    """
    
    def __init__(self, n_positions, ...):
        self.directions = [+1] * n_positions  # Initial directions
        # ... same as existing sampler
    
    def lifted_step(self, pos, state_words, state_types, prob_table):
        """
        One step of lifted MCMC at position pos.
        
        Key difference from Metropolis:
        - If the move is REJECTED, flip the direction (σ → -σ)
        - If the move is ACCEPTED, keep the direction
        """
        current_word = state_words[pos]
        current_type = state_types[pos]
        direction = self.directions[pos]
        
        # Propose word biased by direction
        if direction > 0:
            # Propose "next" words: types with higher index, or 
            # words with higher emission probability for next type
            proposed = self._propose_directional(pos, current_word, current_type, +1)
        else:
            proposed = self._propose_directional(pos, current_word, current_type, -1)
        
        if proposed == current_word:
            # No move possible: flip direction
            self.directions[pos] = -direction
            return current_word, current_type
        
        # Compute energy difference (INTEGER)
        current_energy = compute_word_energy(pos, current_word, ...)
        proposed_energy = compute_word_energy(pos, proposed, ...)
        delta_e = proposed_energy - current_energy
        
        # Accept/reject
        rand_val = random.randint(0, 2**31 - 2)
        if self._accept(delta_e, rand_val, prob_table):
            # ACCEPTED: keep direction, update state
            new_type = get_type_for_word(proposed)
            self.directions[pos] = direction  # Keep direction
            return proposed, new_type
        else:
            # REJECTED: flip direction (this is the LIFT)
            self.directions[pos] = -direction
            return current_word, current_type
```

**Direction-biased proposal**:
```python
def _propose_directional(self, pos, current_word, current_type, direction):
    """
    Propose a word biased by the direction.
    
    direction > 0: prefer words of "next" type (cyclic in type space)
    direction < 0: prefer words of "previous" type
    
    This creates a directed walk through type space,
    preventing the back-and-forth oscillation of Metropolis.
    """
    if direction > 0:
        # Next type (cyclic)
        next_type = (current_type + 1) % self.n_types
    else:
        prev_type = (current_type - 1) % self.n_types
    
    # Sample from emission distribution for the target type
    target_type = next_type if direction > 0 else prev_type
    
    # Use precomputed emission cumsum for this type (INTEGER)
    proposed = self._sample_from_cumsum(self.emit_cumsum_by_type[target_type])
    
    # With some probability, also try other proposals
    r = random.randint(0, 9)
    if r < 3:
        # 30%: emission-biased proposal (as above)
        pass  # already set
    elif r < 7:
        # 40%: coupling-neighbor proposal (same as current)
        neighbors = self.proposal_cache.get(current_word, [current_word])
        proposed = random.choice(neighbors)
    else:
        # 30%: field-weighted proposal
        proposed = self._sample_from_cumsum(self.field_weights[pos % seq_len])
    
    return proposed
```

#### Why This Solves Repetition

1. **No backtracking**: When the chain rejects a move from word A to word B, the direction flips. The next attempt goes in the opposite direction. This prevents the "bouncing" between A and B that causes apparent trapping.

2. **Directed exploration**: In a flat energy region (many words with similar energy), the directed walk systematically explores in one direction rather than diffusing randomly. This is like "sweeping" through the vocabulary rather than jittering.

3. **Faster barrier crossing**: Turitsyn et al. (2011) showed that for Ising models, the lifted chain crosses barriers O(N) times faster than the reversible chain, where N is the number of spins.

#### Detailed Balance

**NO — deliberately broken**. The lifted chain satisfies **global balance** instead:

```
Σ_{(y,σ')} T((x,σ) → (y,σ')) π(x) = Σ_{(y,σ')} T((y,σ') → (x,σ)) π(y)
```

This is **still a valid MCMC chain** — it has the correct stationary distribution π(x) (marginalized over σ). The lack of detailed balance is a feature, not a bug: it's what enables the O(N) speedup.

**Important caveat**: The stationary distribution of the marginal chain over x is still π(x) = exp(-βE(x))/Z. You are sampling from the **same** target distribution as Metropolis, just more efficiently.

#### Computational Cost

- **Per step**: Same as Metropolis + one direction flip operation (integer sign flip)
- **Total**: O(L × V_proposal) — **identical to current cost**

This is the most computationally efficient solution — zero additional cost per step.

#### Limitations

1. **Direction definition in high-dimensional discrete space**: The notion of "direction" in word-type space is less natural than in continuous space or Ising models (where σ = ±1 naturally corresponds to spin up/down). Your type space has 13 values, which is awkward for a binary direction.

2. **Marginal improvement for heterogeneous couplings**: The theoretical O(N) speedup is proven for homogeneous Ising models. For disordered systems (your case), the speedup is typically smaller.

3. **The "momentum" can get trapped in cycles**: If the directed walk encounters a closed loop in type space, it may cycle indefinitely. This is mitigated by the probabilistic direction flips.

---

## ADDITIONAL APPROACHES ANALYZED

### Glauber Dynamics vs Metropolis

**Glauber dynamics** (heat-bath algorithm) samples the new state directly from the conditional distribution:

```
P(w_new = w | rest) = exp(-β × E(pos, w, rest)) / Z_local
```

This is different from Metropolis, which proposes and accepts/rejects. Glauber has **faster mixing** for high-T and moderate coupling (Levin, Luczak & Peres, 2010), but **slower mixing** for low-T strong-coupling regimes.

**Integer implementation**: You already partially implement Glauber for the type layer (exact enumeration over 13 types). For the word layer, exact enumeration over V=3000 is too expensive per step.

**Verdict**: Not a solution to repetition. Glauber and Metropolis have the same ergodicity properties — both get trapped.

### Multicanonical Ensemble (Berg & Neuhaus, 1992)

Closely related to Wang-Landau but uses a **precomputed** weight function w(E) instead of the adaptive g(E) estimator. The target distribution is:

```
P(E) ∝ w(E) × g(E) × exp(-βE)
```

where w(E) is chosen so that P(E) is flat. The difference from Wang-Landau is that w(E) is computed from a preliminary simulation, then fixed.

**Verdict**: Equivalent to Wang-Landau in the limit, but requires a preliminary simulation. Wang-Landau is preferable because it's adaptive.

### Simulated Tempering

A single chain varies its temperature over time according to a schedule. The temperature is treated as an auxiliary variable:

```
P(β, x) ∝ exp(-β × E(x) + g(β))
```

where g(β) is a "weight function" for temperatures (analogous to multicanonical weights for energy).

**Verdict**: Simpler than parallel tempering (only one chain), but requires careful tuning of g(β). For your problem, PT is preferable because it doesn't require the g(β) weights.

### Fermionic/Pauli Exclusion Constraint

Idea: impose that no two positions can have the same word value, analogous to Pauli exclusion for fermions.

**Problems**:
1. **Violates the model**: Language DOES repeat words ("the the" is wrong, but "the ... the" is fine).
2. **Intractable for V=3000**: The constraint state space has size V!/(V-L)! which is enormous.
3. **No clear mapping to integer MCMC**: Fermionic constraints in quantum Monte Carlo require determinants, which need FP.

**Verdict**: Not applicable.

### Tabu Search as MCMC

Tabu search maintains a list of recently visited states and forbids revisiting them. It can be combined with MCMC by modifying the proposal distribution to exclude tabu states.

**Problems**:
1. **Breaks detailed balance**: The modified proposal doesn't satisfy detailed balance with respect to any known stationary distribution.
2. **Memory**: For V^L states, the tabu list grows quickly.
3. **Not principled**: There's no theorem guaranteeing convergence to the correct distribution.

**Verdict**: Unprincipled. The history-driven target in your current code is essentially tabu search, and it should be replaced.

### J-search / Look-ahead MCMC

The idea is to look ahead k steps and choose the move that leads to the best state after k steps.

**Problems**:
1. **Exponential lookahead**: k-step lookahead requires O(V^k) energy evaluations.
2. **Breaks detailed balance**: The move selection is no longer Markovian.
3. **Not standard MCMC**: No convergence guarantees.

**Verdict**: Not practical for V=3000.

---

## SPIN GLASS THEORY: WHAT YOUR ENERGY LANDSCAPE LOOKS LIKE

### The SK Model Analogy

Your model with random-like PMI couplings (both positive and negative J entries) is analogous to the **Sherrington-Kirkpatrick (SK) model** — a fully connected spin glass with random couplings. The SK model has:

1. **Exponentially many metastable states**: O(exp(0.199N)) for N spins (Bray & Moore, 1980)
2. **Ultrametric organization**: Metastable states form a hierarchical tree (Parisi, 1980)
3. **Diverging barriers**: Energy barriers between states grow as N^(1/2)
4. **Aging**: The system's response depends on its entire history

For your problem (N=20 positions, V=3000 values each), the effective state space is V^N ≈ 3000^20. The coupling matrix J is derived from PMI statistics, which have both positive and negative entries with high variance. This creates a **rough landscape** with many local minima.

### Implications for Sampler Design

1. **Single-site Metropolis WILL get trapped**: This is a mathematical certainty for SK-like models below the critical temperature. No amount of tuning will fix it.

2. **Cluster algorithms help but not completely**: In the SK model, clusters are typically small (size ~1) because the random couplings prevent large-scale correlations. However, your J is not fully random — it has structure from PMI co-occurrence, which creates **block structure** that cluster algorithms can exploit.

3. **The optimal approach is a COMBINATION**: Cluster moves (to exploit structure) + parallel tempering (to handle frustration) + entropy regularization (to prevent trapping).

---

## RECOMMENDED IMPLEMENTATION PRIORITY

### Phase 1 (Immediate): Fix Parallel Tempering
- Replace runtime `math.exp()` in swaps with precomputed integer tables
- Use geometric beta ladder with 8-16 replicas
- Implement adaptive ladder tuning
- **Effort**: ~2 hours of code changes
- **Impact**: Moderate (PT already partially works)

### Phase 2 (High Impact): Swendsen-Wang Cluster Moves
- Implement FK bond activation with precomputed integer thresholds
- Implement Union-Find cluster identification
- Replace single-site updates with cluster updates for same-type regions
- **Effort**: ~1 day
- **Impact**: High — fundamentally changes the move set

### Phase 3 (Novel): Entropy-Regularized Free Energy
- Implement local entropy estimator (accessible neighbor count)
- Add entropy bonus to energy: F = E - T_ent × S
- Tune T_ent to balance coherence vs diversity
- **Effort**: ~1 day
- **Impact**: High — directly targets the entropy deficit

### Phase 4 (Optional): Wang-Landau Density of States
- Precompute g(E) from training data (one-time)
- Use for sampling with modified acceptance ratios
- **Effort**: ~2 days
- **Impact**: High for generation quality, but requires storage of g(E)

### Phase 5 (Optional): Lifted MCMC
- Add direction variable to state
- Implement directional proposals with rejection-triggered direction flips
- **Effort**: ~4 hours
- **Impact**: Low-to-moderate for this problem

---

## COMPARISON TABLE

| Solution | Preserves Detailed Balance? | Addresses Root Cause? | Integer-Only? | Cost vs Metropolis | Impact |
|----------|---------------------------|----------------------|---------------|-------------------|--------|
| **SW Cluster** | YES | YES (barriers) | YES | Same O() | ⭐⭐⭐⭐⭐ |
| **Wang-Landau** | YES (post-convergence) | YES (flat histogram) | YES | Same + precompute | ⭐⭐⭐⭐ |
| **Proper PT** | YES | YES (tunneling) | YES (after fix) | n_replicas × | ⭐⭐⭐⭐ |
| **Entropy-regularized** | YES (new target) | YES (entropy deficit) | YES | ~2× | ⭐⭐⭐⭐ |
| **Lifted MCMC** | NO (global balance) | Partial (faster mixing) | YES | Same | ⭐⭐⭐ |
| History-driven | NO | Partial (symptom) | YES | Same + memory | ⭐⭐ |
| Momentum | NO | No | YES | Same | ⭐ |
| Self-repulsion | YES (wrong model) | No (symptom) | YES | Same | ⭐ |

---

## KEY REFERENCES

1. **Swendsen-Wang**: Swendsen, R.H. & Wang, J.S. (1987). "Nonuniversal critical dynamics in Monte Carlo simulations." *Phys. Rev. Lett.* 58, 86.
2. **Wolff**: Wolff, U. (1989). "Collective Monte Carlo updating for spin systems." *Phys. Rev. Lett.* 62, 361.
3. **FK Representation**: Fortuin, C.M. & Kasteleyn, P.W. (1972). "On the random-cluster model." *Physica* 57, 536.
4. **Wang-Landau**: Wang, F. & Landau, D.P. (2001). "Efficient, multiple-range random walk algorithm to calculate the density of states." *Phys. Rev. Lett.* 86, 2050.
5. **Multicanonical**: Berg, B.A. & Neuhaus, T. (1992). "Multicanonical ensemble: A new approach to simulate first-order phase transitions." *Phys. Rev. Lett.* 68, 9.
6. **Parallel Tempering**: Hukushima, K. & Nemoto, K. (1996). "Exchange Monte Carlo method and application to spin glass simulations." *J. Phys. Soc. Japan* 65, 1604.
7. **Optimal PT Ladder**: Katzgraber, H.G. et al. (2006). "How to hack the optimum Monte Carlo algorithm for a spin glass." *Phys. Rev. Lett.* 97, 129601.
8. **Lifted MCMC**: Diaconas, P., Holmes, S. & Neal, R.M. (2000). "Analysis of a non-reversible Markov chain sampler." *Ann. Stat.* 28, 40.
9. **Non-reversible MCMC**: Turitsyn, K.S., Chertkov, M. & Vucelja, M. (2011). "Irreversible Monte Carlo algorithms for efficient sampling." *Physica D* 240, 410.
10. **Locally-balanced proposals**: Zanella, G. (2020). "Informed proposals for local MCMC in discrete spaces." *JASA* 115, 1452.
11. **Spin Glass Theory**: Mezard, M., Parisi, G. & Virasoro, M.A. (1987). *Spin Glass Theory and Beyond.* World Scientific.
12. **SK Model**: Sherrington, D. & Kirkpatrick, S. (1975). "Solvable model of a spin-glass." *Phys. Rev. Lett.* 35, 1792.
13. **Edwards-Sokal**: Edwards, R.G. & Sokal, A.D. (1988). "Generalization of the Fortuin-Kasteleyn-Swendsen-Wang representation and Monte Carlo algorithm." *Phys. Rev. D* 38, 2009.
14. **WL Convergence**: Zhou, T. & Bhatt, R.N. (2005). "Understanding the Wang-Landau algorithm." *Phys. Rev. E* 72, 025701.
15. **Entropy-regularized MCMC**: Habib, S. et al. (2024). "Optimizing Monte Carlo sampling with variational autoencoders." arXiv preprint.

---

## THE DEEP INSIGHT

The repetition problem in your Ising-Potts language model is not a bug — it's a **phase transition**. At low temperature, the system is in an ordered phase where individual positions "freeze" into low-energy word choices. This is the same physics as the low-temperature phase of the Potts model, where all spins align.

The principled solutions all work by **changing the dynamics**, not the energy landscape:
- **Cluster algorithms**: Make global moves that can unfreeze entire correlated regions
- **Wang-Landau / Multicanonical**: Sample from a modified distribution that doesn't concentrate on the frozen phase
- **Parallel tempering**: Use high-temperature chains that are above the freezing transition
- **Entropy regularization**: Add a thermodynamic potential that penalizes frozen states

The ad-hoc workarounds (history, momentum, repulsion) try to fix the symptoms by modifying the energy function or breaking detailed balance. The principled solutions fix the cause by using dynamics that are ergodic at all temperatures.
