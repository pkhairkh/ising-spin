"""
Dependency Tree Couplings (J_tree) for the Ising Spin Language Model.

Implements long-range couplings extracted from dependency parse trees.
These enable subject-verb agreement and other syntactic dependencies
that span beyond the local window.

The key insight from Reinhart & De las Coves (arXiv:2208.08301):
  - 1D Ising chains can only generate context-free languages
  - Long-range couplings (like dependency edges) enable context-sensitive
    generative capacity

J_tree is computed as:
  J_tree[head_w, dep_w] = sum over dep_labels of:
    sign(dep_label) * (count(head_w, dep_w, dep_label).bit_length() - 1)

Where sign is:
  +1 for subject/object dependencies (agreement couplings)
  -1 for modifier dependencies (anti-correlation / competition)
  0 for uninformative dependencies

All integer arithmetic. The J_tree matrix is sparse — only non-zero
entries for pairs actually observed in dependency relations.
"""

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple
import json
import numpy as np

from .spacy_tagger import (
    DEP_LABELS_FOR_TREE, N_DEP_LABELS, DEP_IDX2LABEL,
)
from .type_system import POS2IDX, N_POS


# Dependency label categories:
# AGREEMENT: subject-verb, verb-object — these pairs should COUPLE (positive J)
# MODIFICATION: adj-noun, det-noun — these also couple
# COMPETITION: multiple subjects of same verb, multiple determiners of same noun
AGREEMENT_DEPS = {"nsubj", "nsubjpass", "csubj", "dobj", "iobj", "aux", "cop"}
MODIFICATION_DEPS = {"amod", "det", "compound", "nmod"}
COMPLEMENT_DEPS = {"ccomp", "xcomp", "acl", "advcl"}
OTHER_DEPS = {"mark"}  # weak coupling

# Map to integer indices and signs
DEP_SIGN = {}
for label, idx in DEP_LABELS_FOR_TREE.items():
    if label in AGREEMENT_DEPS:
        DEP_SIGN[idx] = 1
    elif label in MODIFICATION_DEPS:
        DEP_SIGN[idx] = 1
    elif label in COMPLEMENT_DEPS:
        DEP_SIGN[idx] = 1
    elif label in OTHER_DEPS:
        DEP_SIGN[idx] = 0
    else:
        DEP_SIGN[idx] = 0


class DependencyCouplings:
    """
    Long-range dependency tree couplings for the Ising model.

    J_tree[w_head, w_dep] = integer coupling from dependency statistics.
    This enables subject-verb agreement and other long-range syntactic
    constraints beyond the local window.

    Also stores:
      - J_tree_type[t_head, t_dep, dep_label]: type-level coupling
      - dep_bias[pos, dep_label]: bias for position pos to be in a dep relation
    """

    def __init__(self, vocab_size: int, n_pos: int = N_POS, n_dep: int = N_DEP_LABELS):
        self.vocab_size = vocab_size
        self.n_pos = n_pos
        self.n_dep = n_dep

        # Word-level dependency coupling: J_tree[w_head, w_dep] = integer
        # Sparse: only stores observed dependency pairs
        self.J_tree = np.zeros((vocab_size, vocab_size), dtype=np.int64)

        # Type-level dependency coupling: J_tree_type[dep_label, t_head, t_dep] = integer
        self.J_tree_type = np.zeros((n_dep, n_pos, n_pos), dtype=np.int64)

        # Dependency-specific pair counts: (head_w, dep_w, dep_label) -> count
        self.dep_pair_counts: Dict[Tuple[int, int, int], int] = defaultdict(int)

        # Position-level dependency bias: dep_bias[dep_label, t_head, t_dep]
        # How often does POS t_head govern POS t_dep via this dependency?
        self.dep_pos_counts: Dict[Tuple[int, int, int], int] = defaultdict(int)

        # Grammar agreement penalties derived from dep tree
        # E.g.: nsubj(VERB, NOUN) should have matching number
        self.agreement_rules: List[Dict] = []

    def build_from_spacy_tagger(
        self,
        spacy_tagger,
        idx2word: Dict[int, str],
        min_count: int = 1,
        coupling_strength: int = 3,
    ) -> "DependencyCouplings":
        """
        Build J_tree from spaCy tagger's dependency edge data.

        For each (head_w, dep_w, dep_label, dist) in the spaCy output:
          1. Accumulate counts per (head, dep, label) triple
          2. Compute log-floor coupling: sign * bit_length(count)
          3. Accumulate type-level coupling from POS of head/dep
          4. Build agreement rules from high-count patterns

        All integer arithmetic.
        """
        dep_counts: Dict[Tuple[int, int, int], int] = defaultdict(int)

        # Accumulate from spaCy tagger's dep_edges
        for head_w, dep_w, dep_label_idx, dist in spacy_tagger.dep_edges:
            if head_w >= self.vocab_size or dep_w >= self.vocab_size:
                continue
            key = (head_w, dep_w, dep_label_idx)
            dep_counts[key] += 1
            self.dep_pair_counts[key] += 1

        # Also accumulate from dep_pair_counts (already computed by tagger)
        for (head_w, dep_w, dep_label_idx), count in spacy_tagger.dep_pair_counts.items():
            if head_w >= self.vocab_size or dep_w >= self.vocab_size:
                continue
            key = (head_w, dep_w, dep_label_idx)
            dep_counts[key] = dep_counts.get(key, 0) + count

        # Build J_tree using log-floor coupling
        for (head_w, dep_w, dep_label_idx), count in dep_counts.items():
            if count < min_count:
                continue

            sign = DEP_SIGN.get(dep_label_idx, 0)
            if sign == 0:
                continue

            # Log-floor coupling: bit_length(count) - 1
            # This gives floor(log2(count)) — pure integer operation
            log_count = count.bit_length() - 1
            j_val = sign * log_count * coupling_strength

            # Add to J_tree (symmetric: both directions get the coupling)
            self.J_tree[head_w, dep_w] += j_val
            self.J_tree[dep_w, head_w] += j_val

        # Build type-level coupling from dep_label_pos_counts
        for (dep_label_idx, head_pos, dep_pos), count in spacy_tagger.dep_label_pos_counts.items():
            if dep_label_idx < self.n_dep and head_pos < self.n_pos and dep_pos < self.n_pos:
                self.J_tree_type[dep_label_idx, head_pos, dep_pos] = count * coupling_strength
                self.dep_pos_counts[(dep_label_idx, head_pos, dep_pos)] = count

        # Build agreement rules from observed patterns
        self._build_agreement_rules()

        return self

    def _build_agreement_rules(self):
        """
        Derive agreement rules from dependency statistics.

        Rules are of the form:
          - nsubj: head POS should be VERB, dep POS should be NOUN/PRON
          - dobj: head POS should be VERB, dep POS should be NOUN/PRON
          - det: head POS should be NOUN, dep POS should be DET
          - aux: head POS should be VERB, dep POS should be AUX

        These are used as integer grammar penalties in the sampler.
        """
        # For each dependency label, find the most common (head_pos, dep_pos) pair
        for dep_label_idx in range(self.n_dep):
            dep_label = DEP_IDX2LABEL.get(dep_label_idx, "?")

            # Collect all (head_pos, dep_pos) counts for this label
            pos_counts = {}
            for (dl, hp, dp), count in self.dep_pos_counts.items():
                if dl == dep_label_idx:
                    pos_counts[(hp, dp)] = count

            if not pos_counts:
                continue

            # Find the most common pair
            best_pair = max(pos_counts, key=pos_counts.get)
            head_pos, dep_pos = best_pair
            count = pos_counts[best_pair]

            # Only add rule if we have enough evidence
            if count < 5:
                continue

            # Determine penalty for violating this pattern
            sign = DEP_SIGN.get(dep_label_idx, 0)
            if sign == 0:
                continue

            self.agreement_rules.append({
                "dep_label": dep_label,
                "dep_label_idx": dep_label_idx,
                "expected_head_pos": int(head_pos),
                "expected_dep_pos": int(dep_pos),
                "count": int(count),
                "penalty": min(50, count.bit_length() * 5),  # integer penalty
            })

    def compute_dep_energy(
        self,
        state_words: List[int],
        state_types: List[int],
        pos: int,
        word: int,
        word_type: int,
    ) -> int:
        """
        Compute dependency coupling energy for a word at position pos.

        E_dep = sum over all (pos', word') where J_tree[word, word'] != 0:
            J_tree[word, word'] * dep_type_bonus(type, type')

        The dep_type_bonus implements Marcolli's implicational coupling:
        the dependency coupling is only active when the POS types match
        the expected pattern for that dependency.

        Pure integer addition.
        """
        energy = 0

        # Check J_tree coupling with all positions in the state
        # This is the LONG-RANGE part — no distance limit!
        row = self.J_tree[word]
        for j in range(len(state_words)):
            if j == pos:
                continue
            j_word = state_words[j]
            if j_word >= self.vocab_size:
                continue
            j_val = int(row[j_word])
            if j_val == 0:
                continue

            # Type-based gating: check if the (type[pos], type[j]) pair
            # matches any known dependency pattern
            type_bonus = 0
            for dep_label_idx in range(self.n_dep):
                j_type = state_types[j]
                if dep_label_idx < self.n_dep:
                    # Forward: pos is head, j is dep
                    if self.J_tree_type[dep_label_idx, word_type, j_type] > 0:
                        type_bonus += int(self.J_tree_type[dep_label_idx, word_type, j_type])
                    # Reverse: j is head, pos is dep
                    if self.J_tree_type[dep_label_idx, j_type, word_type] > 0:
                        type_bonus += int(self.J_tree_type[dep_label_idx, j_type, word_type])

            if type_bonus > 0:
                # Activate dependency coupling with type bonus
                energy += j_val + type_bonus // (self.n_dep * 2)
            else:
                # No type match: still couple but weaker
                # Use distance decay (integer: 1 / max(1, |pos-j|//5))
                dist = abs(pos - j)
                decay = max(1, 5 // max(1, dist // 5))
                energy += j_val // decay

        return energy

    def compute_agreement_penalty(
        self,
        state_types: List[int],
        pos: int,
        proposed_type: int,
    ) -> int:
        """
        Compute agreement penalty for type assignments based on
        dependency patterns.

        If a word at position pos is in a subject-verb dependency with
        a word at position j, and their types don't match the expected
        pattern (e.g., NOUN-VERB for nsubj), add a penalty.

        Pure integer comparison and addition.
        """
        penalty = 0

        for rule in self.agreement_rules:
            dep_label_idx = rule["dep_label_idx"]
            exp_head = rule["expected_head_pos"]
            exp_dep = rule["expected_dep_pos"]
            p = rule["penalty"]

            # Check if proposed_type at pos could be head or dep
            # in this dependency pattern
            if proposed_type == exp_head:
                # Look for a dep of the expected type nearby
                for d in range(1, len(state_types)):
                    j = pos + d
                    if j < len(state_types) and state_types[j] == exp_dep:
                        # Agreement satisfied — no penalty
                        break
                    j = pos - d
                    if j >= 0 and state_types[j] == exp_dep:
                        break
                else:
                    # No matching dep found — soft penalty
                    penalty += p // 3

            if proposed_type == exp_dep:
                # Look for a head of the expected type nearby
                for d in range(1, len(state_types)):
                    j = pos + d
                    if j < len(state_types) and state_types[j] == exp_head:
                        break
                    j = pos - d
                    if j >= 0 and state_types[j] == exp_head:
                        break
                else:
                    penalty += p // 3

        return penalty

    def get_tree_neighbors(self, word: int, top_k: int = 20) -> List[Tuple[int, int]]:
        """
        Get top-k words coupled to this word via J_tree.
        Returns list of (word_idx, coupling_value).
        """
        row = self.J_tree[word]
        pairs = [(int(i), int(row[i])) for i in range(self.vocab_size) if row[i] != 0]
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        return pairs[:top_k]

    def get_dep_stats(self) -> Dict:
        """Get dependency coupling statistics."""
        nnz = int(np.count_nonzero(self.J_tree))
        total_edges = len(self.dep_pair_counts)
        n_rules = len(self.agreement_rules)

        pos_range = (int(self.J_tree.min()), int(self.J_tree.max()))

        return {
            "J_tree_nnz": nnz,
            "total_dep_edges": total_edges,
            "agreement_rules": n_rules,
            "J_tree_range": pos_range,
            "dep_label_counts": {
                DEP_IDX2LABEL.get(i, "?"): int(self.J_tree_type[i].sum())
                for i in range(self.n_dep)
            },
        }

    def save(self, path: str):
        """Save dependency couplings to disk."""
        np.save(f"{path}_J_tree.npy", self.J_tree)
        np.save(f"{path}_J_tree_type.npy", self.J_tree_type)

        # Save dep pair counts (sparse)
        dep_pair_ser = {f"{h},{d},{l}": int(c) for (h, d, l), c in self.dep_pair_counts.items()}
        with open(f"{path}_dep_pairs.json", "w") as f:
            json.dump(dep_pair_ser, f)

        # Save agreement rules
        with open(f"{path}_agreement_rules.json", "w") as f:
            json.dump(self.agreement_rules, f, indent=2)

        meta = {
            "vocab_size": self.vocab_size,
            "n_pos": self.n_pos,
            "n_dep": self.n_dep,
        }
        with open(f"{path}_meta.json", "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> "DependencyCouplings":
        """Load dependency couplings from disk."""
        with open(f"{path}_meta.json") as f:
            meta = json.load(f)

        dc = cls(
            vocab_size=meta["vocab_size"],
            n_pos=meta["n_pos"],
            n_dep=meta["n_dep"],
        )
        dc.J_tree = np.load(f"{path}_J_tree.npy")
        dc.J_tree_type = np.load(f"{path}_J_tree_type.npy")

        try:
            with open(f"{path}_dep_pairs.json") as f:
                dep_pair_ser = json.load(f)
            dc.dep_pair_counts = {
                tuple(map(int, k.split(","))): v for k, v in dep_pair_ser.items()
            }
        except FileNotFoundError:
            pass

        try:
            with open(f"{path}_agreement_rules.json") as f:
                dc.agreement_rules = json.load(f)
        except FileNotFoundError:
            pass

        return dc
