# Research Report: Hierarchical Typed Ising Spin Models for Text Generation

## Executive Summary

This report investigates how to extend the current flat Ising spin language model into a **hierarchical, typed** system where: (1) couplings operate at multiple scales (local → clause → sentence → discourse), (2) each position carries a TYPE (POS/semantic category) and a VALUE (specific word), and (3) ALL operations remain integer-only. We identify concrete architectures, theoretical foundations, and specific papers for each of six research areas, and map each to an extension of the current codebase.

---

## 1. Multi-Scale / Hierarchical Ising Models

### Key Idea

Language exhibits coupling at multiple scales simultaneously:
- **Local (d=1–3)**: Adjacent words — bigram/trigram collocations ("the dog", "has been")
- **Mid-range (d=4–8)**: Within-clause — subject-verb agreement, modifier attachment
- **Long-range (d=8+)**: Within-sentence — center embedding, coreference; across sentences — discourse coherence

The current `IsingCouplings` already has `J_by_dist` which separates couplings by distance, but treats them all at the same level. A hierarchical model would have **structurally different coupling matrices at each scale**, not just distance-weighted versions of the same matrix.

### Specific Architecture

```
E(x) = E_local(x) + α·E_mid(x) + β·E_long(x)

E_local  = -Σ_{|i-j|≤3}  J^L[w_i, w_j]        (bigram/trigram scale)
E_mid    = -Σ_{|i-j|≤8}  J^M[type_i, type_j]   (clause-level grammar)
E_long   = -Σ_{edges}    J^G[head_i, dep_j]     (dependency tree edges)
```

Each `J^L`, `J^M`, `J^G` is a distinct integer coupling matrix computed from different statistics:
- `J^L` = co-occurrence counts within distance 3 (current approach)
- `J^M` = POS-tag pair co-occurrence within clauses (from parsed corpus)
- `J^G` = head-dependent pair counts along dependency edges

### Key References

| Paper | arXiv ID | Relevance |
|-------|----------|-----------|
| "A Multi-Scale Spin-Glass Mean-Field Model" | 1804.00629 | Introduces multi-scale SK model with couplings at different scales; pressure per particle is derived for each scale level. Directly applicable: replace "scales" with linguistic levels. |
| "Neural Network Renormalization Group" | PRL 121.260601 (2018) | Variational RG approach with hierarchical architecture. Shows how coarse-graining creates a natural hierarchy of coupling matrices. The RG flow maps directly to: fine-grained lexical → coarse-grained grammatical. |
| "RG-Flow: A hierarchical and explainable flow model based on renormalization group and sparse prior" | ResearchGate/2020 | Incorporates RG and sparse prior into hierarchical generative model. The sparse prior is key: at coarse scales, only a few "block spins" (grammatical types) are active. |
| "Multi-Scale Probabilistic Generation Theory (MSPGT)" | 2505.18244 | **Most directly relevant.** Models LLMs as Hierarchical Variational Information Bottleneck with three latent scales (Local, Intermediate, Global). Proves that language models spontaneously develop phase-transition boundaries between these scales. Shows the three scales have distinct information-theoretic properties. |
| "Ising model with long range correlated disorder on hierarchical lattices" | PhysRevB.81.014204 | Studies next-neighbor Ising model with disordered but long-range correlated couplings on hierarchical lattices. The correlated disorder models linguistic long-range dependencies. |

### Integer-Only Implementation

```
# Current: single J_global + J_by_dist (distance-weighted versions of same thing)
# Proposed: three separate integer coupling matrices

class HierarchicalCouplings:
    J_local:    np.ndarray  # (V, V) int64 — co-occurrence within d≤3
    J_mid:      np.ndarray  # (T, T) int64 — POS-pair within d≤8  (T=num types)
    J_long:     dict        # sparse (head, dep) -> int — dependency edges
    
    alpha: int  # scaling for mid-range (integer multiplier)
    beta:  int  # scaling for long-range (integer multiplier)
```

Energy computation remains pure integer addition:
```python
def get_local_energy(self, state, types, pos, word):
    e = 0
    # Local: word-word within window
    for j in window:
        e += self.J_local[word, state[j]]
    # Mid: type-type within clause
    for j in clause_range:
        e += self.alpha * self.J_mid[types[pos], types[j]]
    # Long: dependency edge
    if pos in self.dep_edges:
        head_pos = self.dep_edges[pos]
        e += self.beta * self.J_long.get((state[head_pos], word), 0)
    return e
```

### How This Extends Current Model

1. Add `J_mid` (type × type) and `J_long` (sparse dependency dict) to `IsingCouplings`
2. Add `alpha`, `beta` integer scaling parameters
3. Modify `get_local_energy()` to sum contributions from all three scales
4. Compute `J_mid` from POS-tagged corpus statistics (integer counting)
5. Compute `J_long` from dependency-parsed corpus statistics (integer counting)

---

## 2. Typed Spin Variables

### Key Idea

Currently each position `i` holds a single spin value `x_i ∈ {0, ..., V-1}` from the vocabulary. In a **typed** model, each position holds:
- `t_i ∈ {0, ..., T-1}` — a TYPE (POS tag, semantic category)
- `v_i ∈ AllowedValues(t_i)` — a VALUE (specific word consistent with the type)

This creates a two-layer spin system: the type layer constrains the value layer.

### Specific Architecture

The coupled Ising-Potts model is the natural mathematical framework:

```
E(types, values) = E_type(types) + E_emit(values | types) + E_value(values)

E_type(types)  = -Σ_{i,j} J^T[t_i, t_j] - Σ_i h^T[i][t_i]          # Type interactions
E_emit(values|types) = -Σ_i I[i][v_i, t_i]                            # Type→Value coupling
E_value(values) = -Σ_{i,j} J^V[v_i, v_j]                              # Value interactions
```

Where:
- `J^T` is a T×T integer matrix (e.g., 17×17 for universal POS tags)
- `I[i]` is a V×T indicator matrix: `I[v, t] = 1` if word v has type t, else 0
- `J^V` is the standard V×V coupling matrix (the current `J_global`)
- `AllowedValues(t)` is the set of words with `I[v, t] > 0`

### Key References

| Paper | arXiv ID | Relevance |
|-------|----------|-----------|
| "Coupled Ising-Potts Model: Rich Sets of Critical Temperatures and Translation-Invariant Gibbs Measures" | 2502.12014 | **Exact model needed.** Considers coupled Ising-Potts on Cayley trees with spin vectors (s, σ). The Ising spin (binary) corresponds to a coarse type distinction; the Potts spin (q-state) corresponds to the fine-grained value. Derives multiple Gibbs measures — different "phases" correspond to different grammatical/semantic regimes. |
| "Ising Models with Hidden Markov Structure" | 2504.13927 | **Directly applicable.** Couples two Ising layers: hidden spins s(x) ∈ {±1} and observed spins σ(x) ∈ {±1} on a Cayley tree. The Hamiltonian has Ising interactions within each layer AND site-wise emission couplings between layers. This extends hidden Markov models to a bilayer MRF. The hidden layer = types; the observed layer = values. |
| "Spin Glass Models of Syntax and Language Evolution" | 1508.00504 | Marcolli et al. treat binary syntactic parameters as Ising spins, then couple Ising and Potts (q=3) at vertices to handle entailment relations. This is exactly the type-value structure: the Ising layer captures binary features (+/-V2, +/- wh-movement), while the Potts layer captures multi-valued parameters. |
| "Potts model with invisible states" | 2211.14048 | Introduces "invisible states" that don't interact — models words that are present in the vocabulary but don't contribute to the energy. Useful for modeling rare words within a type. |
| "A constrained Potts antiferromagnet model" | cond-mat/9708171 | Potts model where neighboring spins must have DIFFERENT values — models the constraint that adjacent words should not be the same type (anti-adjacency constraint for function words). |

### Integer-Only Implementation

```python
class TypedSpinState:
    """Each position has a type (POS) and value (word)."""
    types: List[int]   # t_i ∈ {0,...,T-1}, T = number of POS tags
    values: List[int]  # v_i ∈ {0,...,V-1}, constrained by type
    
class TypedCouplings:
    # Type-level couplings (small: T×T, e.g., 17×17)
    J_type: np.ndarray     # int64, shape (T, T)
    h_type: np.ndarray     # int64, shape (seq_len, T)
    
    # Emission coupling: which values are allowed for each type
    # I[v, t] = integer weight (0 = forbidden, >0 = allowed)
    I_emit: np.ndarray     # int64, shape (V, T)
    
    # Value-level couplings (large: V×V, sparse)
    J_value: np.ndarray    # int64, shape (V, V) — same as current J_global
    
    # Allowed values per type (precomputed for fast proposal)
    allowed_values: Dict[int, List[int]]  # type -> list of word ids
    
    def get_local_energy(self, state: TypedSpinState, pos: int, 
                         new_type: int, new_value: int) -> int:
        e = 0
        # Type-type interaction (small matrix, fast)
        for j in neighbors(pos):
            e += self.J_type[new_type, state.types[j]]
        # Emission: type-value consistency
        e += self.I_emit[new_value, new_type]
        # Value-value interaction (same as current)
        for j in window(pos):
            e += self.J_value[new_value, state.values[j]]
        return e
```

### How This Extends Current Model

1. Add `types` array alongside `values` in the state
2. Add `J_type` (T×T) and `I_emit` (V×T) coupling matrices
3. Modify `compute_from_sequences()` to also count POS co-occurrences → `J_type`
4. Build `I_emit` from vocabulary → POS mapping (integer indicator matrix)
5. Modify Gibbs sampler to alternate between type updates and value updates
6. Constrain value proposals to `allowed_values[t_i]` — dramatically reduces proposal space

**Key advantage**: The type layer (T≈17 POS tags) is small enough for exact enumeration in Gibbs sampling, while the value layer (V≈5000) uses the current sparse approach. Type updates are essentially free computationally.

---

## 3. Factorized Energy Functions for Grammar + Semantics

### Key Idea

Decompose the energy into three additive terms, each computed from different coupling structures:

```
E(x) = E_grammar(types) + E_lexical(words | types) + E_semantic(words)
```

This is a **factor graph** decomposition: each factor (grammar, lexical, semantic) operates on a different subset of variables and has its own coupling topology.

### Specific Architecture

```
                    ┌─────────────────┐
                    │  E_grammar(t)    │  ← operates on type variables only
                    │  J^T: T×T       │  ← POS tag transition matrix
                    │  Topology: chain │  ← linear sequence of POS tags
                    └────────┬────────┘
                             │ I_emit[v,t]  ← type→value coupling
                    ┌────────┴────────┐
                    │  E_lexical(v|t)  │  ← values conditioned on types
                    │  I: V×T sparse   │  ← emission weights
                    │  Topology: star  │  ← each value connected to its type
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │  E_semantic(v)   │  ← word-word semantic coherence
                    │  J^V: V×V sparse │  ← co-occurrence coupling
                    │  Topology: tree  │  ← dependency parse edges
                    └─────────────────┘
```

Each factor contributes independently to the total energy (all integer):

| Factor | Variables | Coupling | Topology | Integer Source |
|--------|-----------|----------|----------|---------------|
| E_grammar | types only | J^T (T×T) | Linear chain | POS n-gram counts |
| E_lexical | values given types | I (V×T) | Bipartite star | Word-POS counts |
| E_semantic | values only | J^V (V×V) | Dependency tree | Co-occurrence counts |

### Key References

| Paper | arXiv ID | Relevance |
|-------|----------|-----------|
| "Energy-Based Models with Applications to Speech and Language" | 2403.10961 | **Comprehensive monograph** on EBMs for speech/language. Chapter on factor graph decompositions: shows how complex language models can be built by combining simple energy factors. Explicitly discusses CRFs, Boltzmann machines, and factor graphs as special cases of EBMs. |
| "Autoregressive Language Models are Secretly Energy-Based Models" | 2512.15605 | Establishes explicit bijection between ARMs and EBMs. Key insight: an ARM's conditional probabilities P(x_i|x_{<i}) can be reinterpreted as energy differences in an EBM. This means our factorized EBM can simulate any ARM, but with more structural control. |
| "LEO: Learning Energy-based Models in Factor Graph Optimization" | 2108.02274 | Learns observation models end-to-end with graph optimizers. Shows how to combine learned energy factors with combinatorial optimization. The factor graph structure is key: each factor is a local energy function. |
| "An Introduction to Conditional Random Fields" | 1011.4088 | Sutton & McCallum tutorial. CRFs are exactly factorized energy models: P(y|x) ∝ exp(Σ_k θ_k f_k(y,x)). When features f_k are integer-valued and parameters θ_k are integers, the entire model is integer-only. Linear-chain CRFs correspond to E_grammar; skip-chain CRFs correspond to E_semantic. |
| "Graphical Models with Structured Factors, Neural Parameters" | CMU thesis (Gormley) | Introduces general framework for modeling with four ingredients: factors, parameters, sufficient statistics, and potential functions. When sufficient statistics are integer (e.g., indicator functions) and parameters are integer, all energy computations are integer. |

### Integer-Only Implementation

```python
class FactorizedEnergy:
    """E = E_grammar + E_lexical + E_semantic, all integer."""
    
    def compute_grammar_energy(self, types: List[int]) -> int:
        """Linear chain CRF on types. O(n * T^2) but T is small."""
        e = 0
        for i in range(len(types)):
            e += self.h_type[i, types[i]]
            if i > 0:
                e += self.J_type[types[i-1], types[i]]
        return e
    
    def compute_lexical_energy(self, values: List[int], types: List[int]) -> int:
        """Emission: value-type consistency. O(n)."""
        e = 0
        for i in range(len(values)):
            e += self.I_emit[values[i], types[i]]
        return e
    
    def compute_semantic_energy(self, values: List[int], 
                                 dep_edges: List[Tuple[int,int]]) -> int:
        """Word-word coupling along dependency tree. O(n * avg_degree)."""
        e = 0
        for i in range(len(values)):
            e += self.h_value[i, values[i]]
        for (head, dep) in dep_edges:
            e += self.J_value[values[head], values[dep]]
        return e
    
    def total_energy(self, state: TypedSpinState) -> int:
        return (self.compute_grammar_energy(state.types) +
                self.compute_lexical_energy(state.values, state.types) +
                self.compute_semantic_energy(state.values, state.dep_edges))
```

### How This Extends Current Model

The current model already has a form of `E_semantic` (J_global) and `E_lexical` (h fields). The extension adds:
1. Explicit type variables and `E_grammar` factor
2. `I_emit` coupling between types and values
3. Separate Gibbs sweeps for each factor (type sweep, value sweep, edge sweep)
4. Each factor can have its own temperature schedule (see Section 6)

---

## 4. Markov Random Fields on Parse Trees

### Key Idea

Instead of placing the Ising model on a linear chain (positions 1, 2, ..., n), place it on the **dependency parse tree**. The topology becomes a tree, not a chain. This captures the fundamental linguistic structure: words interact primarily through syntactic head-dependent relations, not just linear adjacency.

### Specific Architecture

```
Linear chain (current):     Parse tree (proposed):
1─2─3─4─5─6─7─8               1 (root)
                                ├─ 2
                                │   ├─ 3
                                │   └─ 4
                                ├─ 5
                                │   └─ 6
                                └─ 7
                                    └─ 8
```

In the tree topology:
- Each word is coupled to its **syntactic head** (parent) and **dependents** (children)
- Long-range dependencies become short-range in tree space
- Subject-verb agreement, which is long-range on the chain, is a direct edge on the tree

The energy function on the tree:

```
E_tree(x) = -Σ_{(i,j)∈edges(T)} J_tree[x_i, x_j] - Σ_i h[i][x_i]
```

where `edges(T)` are the edges of the dependency tree. This is a **tree-structured Ising model**, which has remarkable computational properties.

### Key References

| Paper | arXiv ID | Relevance |
|-------|----------|-----------|
| "Tree-structured Ising models under mean parameterization" | 2507.18749 | **Most directly relevant.** Studies tree-structured Ising models, showing that mean parameterization (instead of canonical) has significant advantages for inference. On trees, exact inference (marginals, MAP) is O(n) via belief propagation. The mean parameters are just the spin expectations ⟨s_i⟩ and ⟨s_i s_j⟩. |
| "Predictive Learning on Hidden Tree-Structured Ising Models" | JMLR vol 22 (2021) | Provides sample complexity guarantees for exact structure recovery and predictive learning on tree-structured Ising models with noise. Shows that trees can be learned efficiently from data. |
| "Tree-structured Ising models can be learned efficiently" | 2010.14864 | First polynomial-sample, polynomial-time algorithm for learning tree-structured Ising models. The Chow-Liu algorithm recovers the tree structure from mutual information — which can be computed from integer co-occurrence counts. |
| "Easy-First Dependency Parsing with Hierarchical Tree LSTMs" | 1603.00375 | Though neural, the "easy-first" parsing strategy (attach most confident edges first) maps directly to a simulated annealing schedule on a tree-structured MRF: high-T → broad structure, low-T → specific words. |
| "Tree-structured Markov random fields with Poisson marginal" | 2408.13649 | New family of tree-structured MRFs for discrete counting variables. The Poisson marginal is useful for modeling word frequencies (which follow approximately Poisson distributions). |
| "Dependency Parsing" | Jurafsky & Martin, SLP3 Ch. 19 | Standard reference. Graph-based dependency parsing IS already an energy maximization: Y* = argmax score(X, Y) where Y is a dependency tree. The Eisner algorithm finds the optimal tree in O(n³). |

### Integer-Only Implementation

```python
class TreeStructuredCouplings:
    """Ising model on a dependency parse tree."""
    
    # Coupling along tree edges: J_tree[(i,j)][w_i, w_j] = int
    J_tree: Dict[Tuple[int,int], Dict[Tuple[int,int], int]]
    
    # Local fields: same as current
    h: np.ndarray  # (seq_len, V) int64
    
    # Dependency tree structure (precomputed from corpus or generated)
    dep_edges: List[Tuple[int, int]]  # (head, dependent) pairs
    dep_parents: Dict[int, int]       # child -> parent mapping
    
    def get_local_energy(self, state, pos, word):
        """Energy contribution from position pos on the tree."""
        e = int(self.h[pos, word])
        
        # Parent edge
        if pos in self.dep_parents:
            parent = self.dep_parents[pos]
            key = (state[parent], word)  # (parent_word, this_word)
            edge_key = (parent, pos)
            if edge_key in self.J_tree and key in self.J_tree[edge_key]:
                e += self.J_tree[edge_key][key]
        
        # Children edges
        for child in self.dep_children.get(pos, []):
            key = (word, state[child])  # (this_word, child_word)
            edge_key = (pos, child)
            if edge_key in self.J_tree and key in self.J_tree[edge_key]:
                e += self.J_tree[edge_key][key]
        
        # Optional: also include linear-chain neighbors (hybrid)
        for j in linear_neighbors(pos, window=2):
            e += int(self.J_global[word, state[j]])
        
        return e
    
    def compute_from_parsed_corpus(self, parsed_sequences):
        """Build J_tree from dependency-parsed corpus."""
        edge_counts = Counter()  # (head_pos_type, dep_pos_type, w_head, w_dep) -> count
        for tree in parsed_sequences:
            for (head, dep) in tree.edges:
                edge_counts[(tree.pos[head], tree.pos[dep], 
                            tree.words[head], tree.words[dep])] += 1
        # Convert to integer couplings
        for key, count in edge_counts.items():
            ...
```

### Remarkable Property: Exact Inference on Trees

Tree-structured Ising models admit **exact** marginal computation via belief propagation in O(n·V²) — no MCMC needed for the tree part. However, when combined with long-range semantic couplings that break the tree structure, we need hybrid inference:

1. **Tree part**: Belief propagation (exact, integer arithmetic if messages are integer-scaled)
2. **Non-tree part**: Gibbs sampling (as currently implemented)

### How This Extends Current Model

1. Add `dep_edges`, `dep_parents`, `dep_children` to the state
2. Add `J_tree` sparse coupling along tree edges
3. Modify `get_local_energy()` to include tree-edge contributions
4. Optionally: run belief propagation on the tree component (integer messages)
5. For generation without a pre-specified tree: sample the tree structure alongside word values (joint tree-word MCMC)

---

## 5. Integer-Valued Energy Composition

### Key Idea

When E = E₁ + E₂ + E₃ with each Eₖ computed from different coupling matrices, we need efficient composition. The key insight: **integer addition is free** — the challenge is computing each Eₖ efficiently.

### Specific Architecture

The current energy computation iterates over neighbors:
```python
# Current: single window-based coupling
energy = h[pos, word]
for j in window(pos):
    energy += J_global[word, state[j]]
```

With multiple energy terms:
```python
# Proposed: sum of coupling contributions
energy = 0
# Term 1: local coupling (current J_global with small window)
energy += h_local[pos, word]
for j in window(pos, radius=3):
    energy += J_local[word, state[j]]
# Term 2: type coupling (small matrix, fast)
energy += alpha * J_type[type_i, type_j for j in neighbors]
# Term 3: tree coupling (sparse, few edges)
energy += beta * J_tree[(pos, parent)][word, state[parent]]
```

### Key References

| Paper | arXiv ID | Relevance |
|-------|----------|-----------|
| "Efficient Sampling for Ising and Potts Models using Auxiliary Variables" | 2110.10801 | Block Gibbs sampler using auxiliary Gaussian variables. For additive energy decompositions, each term can be "split" into an auxiliary variable, enabling parallel sampling of different terms. The key: when E = E₁ + E₂, introduce auxiliary u₁, u₂ such that p(u₁|E₁) and p(u₂|E₂) are tractable, then alternate between sampling u and sampling x. |
| "On Sampling from Ising Models with Spectral Constraints" | 2407.07645 | Considers sampling when the coupling matrix has eigenvalues in a constrained interval. For additive decompositions, the combined coupling matrix J = J₁ + J₂ + J₃ has a structured spectrum. If each Jₖ is low-rank (which they are: J_type is T×T embedded in V×V, hence rank ≤ T), the spectral structure is exploitable. |
| "A new class of Markov Random Fields enabling lightweight sampling" | 2511.02373 | Introduces MRFs with a special structure that enables O(1) sampling per variable. The key: if each variable's energy can be decomposed into a "local" part (fast to compute) and a "global" part (precomputed), sampling is efficient. This maps to our decomposition: E_local from J_local (fast), E_global from J_type/J_tree (precomputed summaries). |
| "Submodular relaxation for inference in Markov random fields" | 1501.03771 | When exact MAP inference is hard (non-tree structure), submodular relaxations provide integer solutions. The energy decomposition E₁ + E₂ + E₃ can be relaxed separately, and the individual relaxations combined. |

### Integer-Only Implementation: The Key Tricks

**Trick 1: Precompute partial sums.** Since Gibbs sampling changes one variable at a time, we can maintain running sums:

```python
class ComposedEnergy:
    # Running sum of each energy term (maintained incrementally)
    E_local_cache: int   # sum of all local contributions
    E_type_cache: int    # sum of all type contributions
    E_tree_cache: int    # sum of all tree contributions
    
    def delta_energy(self, pos, old_word, new_word, old_type, new_type):
        """Compute change in total energy when flipping position pos."""
        delta = 0
        
        # Local term: recompute only neighbors of pos
        for j in window(pos):
            delta -= J_local[old_word, state[j]]
            delta += J_local[new_word, state[j]]
        delta -= h_local[pos, old_word]
        delta += h_local[pos, new_word]
        
        # Type term: recompute only type neighbors
        for j in type_neighbors(pos):
            delta -= alpha * J_type[old_type, types[j]]
            delta += alpha * J_type[new_type, types[j]]
        
        # Tree term: recompute only tree neighbors
        if pos in dep_parents:
            parent = dep_parents[pos]
            delta -= beta * J_tree_edge(pos, parent, old_word, state[parent])
            delta += beta * J_tree_edge(pos, parent, new_word, state[parent])
        
        return delta  # pure integer
```

**Trick 2: Block Gibbs for type layer.** Since the type layer has only T≈17 states, we can do **exact** Gibbs updates by enumerating all T types:

```python
def gibbs_update_type(self, pos):
    """Exact type update: enumerate all T types."""
    energies = [0] * self.T
    for t in range(self.T):
        energies[t] = self.compute_type_energy(pos, t)
    # Convert to integer probabilities via threshold table
    total = sum(max(0, e) for e in energies)
    r = random.randint(0, total)
    cumsum = 0
    for t in range(self.T):
        cumsum += max(0, energies[t])
        if r < cumsum:
            return t
    return self.T - 1
```

**Trick 3: Separate temperature tables for each factor.**

```python
class MultiTemperature:
    """Each factor can have its own integer temperature."""
    beta_local: int   # β for E_local (fine-grained)
    beta_type: int    # β for E_type (grammatical)
    beta_tree: int    # β for E_tree (structural)
    
    prob_tables: Dict[str, ProbabilityTable]  # one per factor
    
    def accept(self, factor_name, delta_e, rand_val):
        return self.prob_tables[factor_name].accept(delta_e, rand_val)
```

### How This Extends Current Model

1. Replace single `ProbabilityTable` with `MultiTemperature` system
2. Maintain running energy caches for each factor
3. Implement `delta_energy()` for incremental updates (O(neighbors) per flip, not O(n))
4. Add block Gibbs for the type layer (exact, cheap)
5. Energy decomposition is purely additive — integer sum is trivial

---

## 6. Simulated Annealing for Parsing + Generation

### Key Idea

Use a temperature schedule that resolves linguistic structure **level by level**, from coarse (grammar) to fine (words):

```
High T  (β small)  →  Type variables equilibrate: POS tag sequence emerges
Mid T   (β medium) →  Value variables constrain: word types narrow down  
Low T   (β large)  →  Specific words selected: exact lexical choices made
```

This mirrors the physical process of crystallization: first the crystal structure (lattice type) forms, then atoms settle into specific sites.

### Specific Architecture: Staged Annealing Schedule

```
Phase 1: Structural (sweeps 1-30)
    β_type = 500   (medium temperature for types)
    β_value = 0    (values are free — random walk)
    → Type sequence converges to grammatical POS pattern
    
Phase 2: Constrained (sweeps 31-70)
    β_type = 1000  (types are locked)
    β_value = 500  (medium temperature for values)
    → Values narrow to type-consistent words
    
Phase 3: Refinement (sweeps 71-100)
    β_type = 2000  (types frozen)
    β_value = 1500 (low temperature for values)
    → Specific word selection, semantic coherence
```

At each phase, we sample from:
- Phase 1: `P(types) ∝ exp(-β₁ · E_grammar(types))` — values are marginalized
- Phase 2: `P(values | types) ∝ exp(-β₂ · E_lexical(values | types))` — types are clamped
- Phase 3: `P(values | types) ∝ exp(-β₃ · (E_lexical + E_semantic))` — full energy

### Key References

| Paper | arXiv ID | Relevance |
|-------|----------|-----------|
| "Simulated Annealing for Optimization of Graphs and Sequences (SAGS)" | 2110.01384 | Directly applies simulated annealing to sequence optimization (paraphrase generation). Uses temperature scheduling to first resolve global structure, then local details. Shows that staged annealing outperforms single-temperature sampling for text generation. |
| "Optimal schedules for annealing algorithms" | 2402.14717 | Derives formalism for optimal annealing schedules in multidimensional parameter spaces using non-equilibrium statistical mechanics. Key result: the optimal schedule spends more time near phase transitions. For language, this means lingering near the grammar→lexical transition temperature. |
| "Collective Annealing by Switching Temperatures" | 2512.13522 | Proposes switching between temperatures (rather than monotonic cooling) to escape local minima. For language: alternate between high-T (explore grammatical alternatives) and low-T (refine word choices) — a "pumping" strategy. |
| "Score-Based Diffusion meets Annealed Importance Sampling" | 2208.07698 | Leverages score-based generative modeling for AIS. The key insight: annealing from noise to data IS the generation process. Each temperature level corresponds to a different "coarseness" of the data. Maps directly to: high-T = coarse grammatical skeleton, low-T = fine-grained text. |
| "Energy-Inspired Models: Learning with Sampler-Induced Distributions" | NeurIPS 2019 | Describes models where the training objective matches the sampler's stationary distribution. If we design the annealing schedule to match linguistic structure resolution, we get a model where sampling IS structure-aware generation. |
| "MSPGT: Multi-Scale Probabilistic Generation Theory" | 2505.18244 | (Also referenced in §1.) Proves that language models develop **phase-transition boundaries** between local, intermediate, and global processing. These boundaries define the natural temperature thresholds for staged annealing: cross boundary 1 → grammar resolved; cross boundary 2 → semantics resolved. |

### Integer-Only Implementation

```python
class StagedAnnealing:
    """Temperature schedule that resolves grammar before words."""
    
    stages = [
        # (sweep_range, beta_type, beta_value, description)
        (range(0, 30),   500,    0,      "Grammar resolution"),
        (range(30, 70),  1000,   500,    "Lexical narrowing"),
        (range(70, 100), 2000,   1500,   "Semantic refinement"),
    ]
    
    def __init__(self, couplings, vocab):
        self.couplings = couplings
        self.vocab = vocab
        # Precompute probability tables for each stage
        self.prob_tables = {}
        for stage_idx, (sweeps, bt, bv, desc) in enumerate(self.stages):
            self.prob_tables[f"type_{stage_idx}"] = ProbabilityTable(
                max_delta_e=2000, beta_int=bt)
            self.prob_tables[f"value_{stage_idx}"] = ProbabilityTable(
                max_delta_e=2000, beta_int=bv)
    
    def generate(self, length=20, prompt=None):
        state = self._init_state(length, prompt)
        
        for stage_idx, (sweep_range, bt, bv, desc) in enumerate(self.stages):
            for sweep in sweep_range:
                # Phase 1: Update types (grammar)
                if bt > 0:
                    for pos in range(length):
                        self._gibbs_update_type(
                            state, pos, 
                            self.prob_tables[f"type_{stage_idx}"])
                
                # Phase 2: Update values (words)
                if bv > 0:
                    for pos in range(length):
                        self._gibbs_update_value(
                            state, pos,
                            self.prob_tables[f"value_{stage_idx}"])
        
        return self.vocab.decode(state.values)
```

### Advanced: Adaptive Temperature from Energy Gap

The MSPGT paper shows that phase transitions in language models are detectable from the **energy gap** between scales. We can detect these automatically:

```python
def detect_phase_transition(self, state, beta_range):
    """Detect the temperature where type variables 'freeze'."""
    for beta in beta_range:
        # Run a few Gibbs sweeps at this beta
        for _ in range(5):
            self._gibbs_update_type(state, beta)
        # Measure type entropy (integer: count distinct types)
        type_diversity = len(set(state.types))
        if type_diversity < len(state.types) * 0.3:
            # Types have collapsed → this is the critical temperature
            return beta
    return beta_range[-1]
```

### How This Extends Current Model

1. Replace single `temperature` parameter with `StagedAnnealing` schedule
2. Replace single `ProbabilityTable` with per-stage, per-factor tables
3. Modify the `generate()` loop to have explicit phases
4. Add `_gibbs_update_type()` method (exact enumeration over T types)
5. Add `_gibbs_update_value()` method (current approach, but constrained by types)

---

## Integrated Architecture: The Full Hierarchical Typed Ising Model

Combining all six areas, the complete architecture is:

```
┌─────────────────────────────────────────────────────────────────┐
│                    HIERARCHICAL TYPED ISING MODEL               │
│                                                                 │
│  State: (types[0..n], values[0..n], dep_edges[])               │
│                                                                 │
│  Energy:                                                        │
│    E = E_grammar(types)                                         │
│      + E_lexical(values | types)                                │
│      + E_semantic(values)                                       │
│      + E_tree(values, dep_edges)                                │
│                                                                 │
│  Couplings (all integer):                                       │
│    J_type:   T×T       — POS tag transitions (local scale)      │
│    J_mid:    T×T       — POS tag within-clause (mid scale)      │
│    I_emit:   V×T       — type→value emission                    │
│    J_local:  V×V       — word co-occurrence (local scale)       │
│    J_tree:   sparse    — head-dependent pairs (tree scale)      │
│    h_type:   n×T       — position-specific type fields          │
│    h_value:  n×V       — position-specific word fields          │
│                                                                 │
│  Inference: Staged annealing                                    │
│    Phase 1: β_type=500, β_value=0     → resolve grammar         │
│    Phase 2: β_type=1000, β_value=500  → narrow words            │
│    Phase 3: β_type=2000, β_value=1500 → refine semantics        │
│                                                                 │
│  ALL operations: integer addition, integer comparison,           │
│                  table lookup (precomputed thresholds)           │
│  ZERO floating-point in generation loop                         │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Roadmap

| Phase | Extension | Files Modified | Effort |
|-------|-----------|---------------|--------|
| 1 | Add type layer (POS tags) | `couplings.py`, `sampler.py` | Medium |
| 2 | Add multi-scale couplings (J_type, J_mid) | `couplings.py` | Medium |
| 3 | Add dependency tree couplings (J_tree) | New: `tree_couplings.py` | Large |
| 4 | Add factorized energy with type-value emission (I_emit) | `couplings.py` | Medium |
| 5 | Add staged annealing schedule | `sampler.py` | Small |
| 6 | Add data pipeline for POS-tagged and parsed corpus | `data_loader.py` | Medium |
| 7 | Add block Gibbs for type layer | `sampler.py` | Small |
| 8 | Add belief propagation for tree component | New: `belief_prop.py` | Large |

### Expected Benefits

1. **Grammar coherence**: Type layer ensures POS sequences are valid
2. **Long-range dependencies**: Tree couplings capture subject-verb agreement at any distance
3. **Efficient sampling**: Type layer (T≈17) is cheap to sample exactly; constrains value proposals
4. **Controllable generation**: Temperature schedule gives explicit control over grammar vs. semantics
5. **Interpretability**: Each energy term has a clear linguistic interpretation
6. **Integer-only**: All operations remain integer addition, comparison, and lookup

---

## Complete Reference List

| # | Paper | arXiv/DOI | Area |
|---|-------|-----------|------|
| 1 | Multi-Scale Spin-Glass Mean-Field Model | 1804.00629 | Multi-scale |
| 2 | Neural Network Renormalization Group | PRL 121.260601 | Multi-scale |
| 3 | RG-Flow: Hierarchical flow model based on RG | ResearchGate 2020 | Multi-scale |
| 4 | MSPGT: Multi-Scale Probabilistic Generation Theory | 2505.18244 | Multi-scale, Annealing |
| 5 | Ising model with long-range correlated disorder | PhysRevB.81.014204 | Multi-scale |
| 6 | Coupled Ising-Potts Model | 2502.12014 | Typed spins |
| 7 | Ising Models with Hidden Markov Structure | 2504.13927 | Typed spins |
| 8 | Spin Glass Models of Syntax and Language Evolution | 1508.00504 | Typed spins, Grammar |
| 9 | Potts model with invisible states | 2211.14048 | Typed spins |
| 10 | Constrained Potts antiferromagnet | cond-mat/9708171 | Typed spins |
| 11 | Energy-Based Models for Speech and Language | 2403.10961 | Factorized energy |
| 12 | Autoregressive LMs are Secretly EBMs | 2512.15605 | Factorized energy |
| 13 | LEO: Learning Energy-based Models in Factor Graphs | 2108.02274 | Factorized energy |
| 14 | Introduction to Conditional Random Fields | 1011.4088 | Factorized energy |
| 15 | Graphical Models with Structured Factors | CMU thesis | Factorized energy |
| 16 | Tree-structured Ising models under mean parameterization | 2507.18749 | Parse tree MRF |
| 17 | Predictive Learning on Hidden Tree-Structured Ising | JMLR 22 (2021) | Parse tree MRF |
| 18 | Tree-structured Ising models can be learned efficiently | 2010.14864 | Parse tree MRF |
| 19 | Easy-First Dependency Parsing with Hierarchical Tree LSTMs | 1603.00375 | Parse tree MRF |
| 20 | Tree-structured MRFs with Poisson marginal | 2408.13649 | Parse tree MRF |
| 21 | Efficient Sampling for Ising and Potts using Auxiliary Variables | 2110.10801 | Energy composition |
| 22 | Sampling from Ising Models with Spectral Constraints | 2407.07645 | Energy composition |
| 23 | New class of MRFs enabling lightweight sampling | 2511.02373 | Energy composition |
| 24 | Submodular relaxation for MRF inference | 1501.03771 | Energy composition |
| 25 | SAGS: Simulated Annealing for Graph/Sequence Optimization | 2110.01384 | Annealing |
| 26 | Optimal schedules for annealing algorithms | 2402.14717 | Annealing |
| 27 | Collective Annealing by Switching Temperatures | 2512.13522 | Annealing |
| 28 | Score-Based Diffusion meets Annealed Importance Sampling | 2208.07698 | Annealing |
| 29 | Energy-Inspired Models (NeurIPS 2019) | NeurIPS 2019 | Annealing |
| 30 | Boltzmann-GPT: Bridging Energy-Based World Models | 2601.17094 | Hierarchical EBM |
| 31 | Spin glass model of in-context learning | 2408.02288 | Spin glass + language |
| 32 | The grammar of the Ising model: A new complexity hierarchy | 2208.08301 | Ising complexity |
| 33 | Learning Feature Hierarchies with Centered DBMs | 1203.3783 | Hierarchical BM |
| 34 | Restricted BM for Classification with Hierarchical | 1406.3407 | Hierarchical BM |
| 35 | Boltzmann Machines as Spin Glasses | LinkedIn/Michael Erlihson | BM-SG connection |

---

## Key Theoretical Insights

### 1. The Ising-Potts Coupling is Natural for Language

The coupled Ising-Potts model (arXiv:2502.12014) provides the exact mathematical structure for typed spins. The Ising layer (binary) captures coarse distinctions (noun vs. verb, content vs. function word), while the Potts layer (q-state) captures fine-grained distinctions (specific noun, specific verb). The coupling between layers ensures consistency.

### 2. Hidden Markov Structure Extends to MRF

The Ising model with hidden Markov structure (arXiv:2504.13927) shows that the standard HMM (which is a tree-structured Bayesian network) naturally extends to a bilayer MRF (which is an undirected graphical model). This means we can add long-range dependencies that HMMs cannot capture, while maintaining the hidden-observed structure.

### 3. Phase Transitions in Language Models are Real

The MSPGT paper (arXiv:2505.18244) proves that language models spontaneously develop phase-transition boundaries between local, intermediate, and global processing. This validates the staged annealing approach: the natural temperature schedule for language generation IS coarse-to-fine.

### 4. Tree Structure Enables Exact Inference

Tree-structured Ising models (arXiv:2507.18749) admit O(n) exact inference via belief propagation. This means the grammatical component (type layer on a chain or tree) can be solved exactly, and only the semantic component (value layer with non-tree couplings) requires MCMC.

### 5. Integer Arithmetic is Sufficient

All the referenced models use energy functions that are sums of integer coupling terms. The Boltzmann distribution P(x) ∝ exp(-E(x)/T) requires floating point in principle, but:
- **At inference time**: precomputed threshold tables (current approach) avoid exp()
- **At training time**: integer counting (co-occurrence, n-gram) replaces gradient descent
- **The probability table approach**: converts the continuous Boltzmann distribution into a discrete acceptance rule — purely integer comparison

This is not an approximation: for integer energies and integer temperatures, the exact acceptance probability is a rational number, and the threshold table computes it to any desired precision at initialization time.
