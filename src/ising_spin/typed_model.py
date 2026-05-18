"""
Unified Typed Ising-Potts Language Model.

Orchestrates the full architecture:
  1. Data loading from fineweb-edu
  2. Vocabulary building (integer counting)
  3. PMI coupling computation (log-floor, integer-only)
  4. POS type system (rule-based, integer-only)
  5. Semantic type system (rule-based + co-occurrence, integer-only)
  6. Grammar penalties (integer quadratic constraints)
  7. Staged annealing generation (zero FP in generation loop)

Energy function:
  E(types, words) = E_type(types) + E_emit(words|types)
                  + E_lexical(words) + E_semantic(words)
                  + E_grammar(types, words)

All generation-path computation is integer arithmetic only.
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .vocabulary import Vocabulary
from .pmi_couplings import PMICouplings
from .type_system import POSTypeSystem, N_POS, IDX2POS, POS2IDX
from .semantic_types import SemanticTypeSystem, N_SEM, SEMANTIC_SUPERTYPES
from .typed_sampler import StagedAnnealingSampler
from .data_loader import load_fineweb_edu, tokenize_texts, truncate_sequences


class TypedIsingModel:
    """
    End-to-end Typed Ising-Potts Language Model.

    Training phase (one-time, may use FP for data loading/prob tables):
        1. Load corpus from fineweb-edu
        2. Build integer vocabulary
        3. Compute log-floor PMI couplings
        4. Build POS type system with grammar penalties
        5. Build semantic type system with compatibility matrix
        6. Compute Hebbian memory term

    Generation phase (ZERO FP operations):
        1. Initialize types and words from prompt or distributions
        2. Run staged annealing: types → types+words → words
        3. Decode state to text with POS annotations
    """

    def __init__(
        self,
        # Vocabulary
        vocab_min_freq: int = 5,
        vocab_max_size: Optional[int] = 5000,
        # Sequence
        seq_len: int = 30,
        # Coupling
        window: int = 8,
        pmi_cap: int = 15,
        min_cooc: int = 2,
        # Weights
        pmi_weight: int = 3,
        hebbian_weight: int = 1,
        semantic_weight: int = 1,
        # Grammar
        grammar_penalty: int = 50,
        # Annealing
        phase1_beta: int = 200,
        phase2_beta: int = 500,
        phase3_beta: int = 1000,
        total_sweeps: int = 150,
        # Emission
        emission_weight: int = 10,
    ):
        self.vocab_min_freq = vocab_min_freq
        self.vocab_max_size = vocab_max_size
        self.seq_len = seq_len
        self.window = window
        self.pmi_cap = pmi_cap
        self.min_cooc = min_cooc
        self.pmi_weight = pmi_weight
        self.hebbian_weight = hebbian_weight
        self.semantic_weight = semantic_weight
        self.grammar_penalty = grammar_penalty
        self.phase1_beta = phase1_beta
        self.phase2_beta = phase2_beta
        self.phase3_beta = phase3_beta
        self.total_sweeps = total_sweeps
        self.emission_weight = emission_weight

        # Components (populated during training)
        self.vocab: Optional[Vocabulary] = None
        self.pmi: Optional[PMICouplings] = None
        self.types: Optional[POSTypeSystem] = None
        self.semantics: Optional[SemanticTypeSystem] = None
        self.sampler: Optional[StagedAnnealingSampler] = None

    def train(
        self,
        n_samples: int = 50000,
        verbose: bool = True,
    ) -> "TypedIsingModel":
        """
        Train the full model: vocabulary, PMI, types, semantics.

        This is the only phase that may involve FP (for data loading
        and probability table precomputation). All resulting parameters
        are pure integer.
        """
        print("=" * 70)
        print("TYPED ISING-POTTS LANGUAGE MODEL — TRAINING")
        print("=" * 70)

        # Step 1: Load data
        t0 = time.time()
        texts = load_fineweb_edu(n_samples=n_samples)
        print(f"[1/6] Data loading: {len(texts)} texts ({time.time()-t0:.1f}s)")

        # Step 2: Build vocabulary
        t0 = time.time()
        self.vocab = Vocabulary(
            min_freq=self.vocab_min_freq,
            max_size=self.vocab_max_size,
        )
        self.vocab.build(texts)
        V = len(self.vocab)
        print(f"[2/6] Vocabulary: {V} words ({time.time()-t0:.1f}s)")

        # Step 3: Tokenize and build sequences
        t0 = time.time()
        sequences = tokenize_texts(texts, self.vocab)
        sequences = truncate_sequences(sequences, max_len=self.seq_len)
        print(f"[3/6] Tokenization: {len(sequences)} sequences ({time.time()-t0:.1f}s)")

        # Step 4: Compute PMI couplings
        t0 = time.time()
        self.pmi = PMICouplings(
            vocab_size=V,
            seq_len=self.seq_len,
            window=self.window,
        )
        self.pmi.compute_from_sequences(
            sequences,
            min_count=self.min_cooc,
            pmi_cap=self.pmi_cap,
            use_hebbian=True,
            hebbian_weight=self.hebbian_weight,
        )
        pmi_nnz = int(np.count_nonzero(self.pmi.J_PMI))
        pmi_max = int(self.pmi.J_PMI.max())
        pmi_min = int(self.pmi.J_PMI.min())
        hebb_nnz = int(np.count_nonzero(self.pmi.J_Hebb))
        print(f"[4/6] PMI couplings: {pmi_nnz} non-zeros, range [{pmi_min}, {pmi_max}] "
              f"Hebbian: {hebb_nnz} non-zeros ({time.time()-t0:.1f}s)")

        # Step 5: Build POS type system
        t0 = time.time()
        self.types = POSTypeSystem(
            vocab_size=V,
            n_types=N_POS,
            window=self.window,
        )
        self.types.build_from_vocabulary(self.vocab.word2idx, self.vocab.idx2word)
        self.types.compute_type_couplings(sequences, self.vocab.idx2word, scaling=10)
        self.types.build_grammar_penalties(penalty_strength=self.grammar_penalty)
        self.types.precompute_type_distribution()
        n_typed = sum(1 for w in range(V) if len(self.types.allowed_types.get(w, set())) > 0)
        n_penalties = len(self.types.grammar_penalties)
        print(f"[5/6] POS type system: {n_typed}/{V} words typed, "
              f"{n_penalties} grammar penalties ({time.time()-t0:.1f}s)")

        # Step 6: Build semantic type system
        t0 = time.time()
        self.semantics = SemanticTypeSystem(
            vocab_size=V,
            n_sem_types=N_SEM,
            compatibility_strength=3,
        )
        self.semantics.build_from_vocabulary(self.vocab.word2idx, self.vocab.idx2word)
        self.semantics.compute_compatibility_matrix(sequences, min_cooc=2)
        self.semantics.compute_hebbian_coupling(sequences, hebbian_weight=1)
        sem_nnz = int(np.count_nonzero(self.semantics.S))
        print(f"[6/6] Semantic types: {N_SEM} types, "
              f"S matrix {sem_nnz} non-zeros ({time.time()-t0:.1f}s)")

        # Step 7: Build sampler
        print("\nBuilding staged annealing sampler...")
        t0 = time.time()
        self.sampler = StagedAnnealingSampler(
            pmi_couplings=self.pmi,
            type_system=self.types,
            semantic_system=self.semantics,
            phase1_beta=self.phase1_beta,
            phase2_beta=self.phase2_beta,
            phase3_beta=self.phase3_beta,
            total_sweeps=self.total_sweeps,
            pmi_weight=self.pmi_weight,
            hebbian_weight=self.hebbian_weight,
            semantic_weight=self.semantic_weight,
        )
        print(f"Sampler ready ({time.time()-t0:.1f}s)")

        # Print summary
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        self._print_summary()

        return self

    def _print_summary(self):
        """Print model summary statistics."""
        print(f"\nModel Architecture:")
        print(f"  Vocabulary size: {len(self.vocab)}")
        print(f"  POS types: {N_POS}")
        print(f"  Semantic types: {N_SEM}")
        print(f"  PMI coupling range: [{int(self.pmi.J_PMI.min())}, {int(self.pmi.J_PMI.max())}]")
        print(f"  PMI non-zeros: {int(np.count_nonzero(self.pmi.J_PMI))}")
        print(f"  Hebbian non-zeros: {int(np.count_nonzero(self.pmi.J_Hebb))}")
        print(f"  Semantic S non-zeros: {int(np.count_nonzero(self.semantics.S))}")
        print(f"  Grammar penalties: {len(self.types.grammar_penalties)}")
        print(f"  Annealing phases: {self.sampler.sweeps_p1}/{self.sampler.sweeps_p2}/{self.sampler.sweeps_p3} sweeps")
        print(f"  Zero FP in generation loop: YES")

    def generate(
        self,
        prompt: Optional[str] = None,
        length: int = 20,
        verbose: bool = False,
    ) -> str:
        """
        Generate text using staged annealing. ZERO FP in generation loop.

        Returns generated text string.
        """
        if self.sampler is None:
            raise RuntimeError("Model not trained. Call train() first.")

        words, types = self.sampler.generate(
            length=length,
            prompt=prompt,
            vocab=self.vocab,
            verbose=verbose,
        )

        return self._decode_with_annotations(words, types)

    def generate_raw(
        self,
        prompt: Optional[str] = None,
        length: int = 20,
        verbose: bool = False,
    ) -> Tuple[List[int], List[int]]:
        """Generate and return raw (words, types) state."""
        if self.sampler is None:
            raise RuntimeError("Model not trained. Call train() first.")

        return self.sampler.generate(
            length=length,
            prompt=prompt,
            vocab=self.vocab,
            verbose=verbose,
        )

    def generate_batch(
        self,
        n_samples: int = 5,
        prompt: Optional[str] = None,
        length: int = 20,
    ) -> List[str]:
        """Generate multiple samples."""
        results = []
        for _ in range(n_samples):
            words, types = self.sampler.generate(
                length=length,
                prompt=prompt,
                vocab=self.vocab,
            )
            results.append(self._decode_with_annotations(words, types))
        return results

    def generate_with_trace(
        self,
        prompt: Optional[str] = None,
        length: int = 20,
    ) -> Dict:
        """Generate with full trace for analysis."""
        words, types = self.sampler.generate(
            length=length,
            prompt=prompt,
            vocab=self.vocab,
            verbose=True,
        )

        # Compute final energy
        energy = 0
        for i in range(length):
            energy += int(self.pmi.h[i % self.pmi.seq_len, words[i]])
        for i in range(length):
            for j_offset in range(1, self.pmi.window + 1):
                j = i + j_offset
                if j < length:
                    energy += int(self.pmi.J_PMI[words[i], words[j]])

        # Type distribution
        type_counts = {}
        for t in types:
            name = IDX2POS.get(t, "UNK")
            type_counts[name] = type_counts.get(name, 0) + 1

        # Semantic distribution
        sem_counts = {}
        for w in words:
            if w < len(self.semantics.word_to_sem):
                s_idx = int(self.semantics.word_to_sem[w])
                s_name = SEMANTIC_SUPERTYPES[s_idx] if s_idx < len(SEMANTIC_SUPERTYPES) else "UNK"
                sem_counts[s_name] = sem_counts.get(s_name, 0) + 1

        return {
            "text": self.vocab.decode(words),
            "types": [IDX2POS.get(t, "UNK") for t in types],
            "energy": energy,
            "type_counts": type_counts,
            "sem_counts": sem_counts,
            "words": words,
        }

    def _decode_with_annotations(self, words: List[int], types: List[int]) -> str:
        """Decode words with POS type annotations."""
        parts = []
        for i, (w, t) in enumerate(zip(words, types)):
            word = self.vocab.idx2word.get(w, "<UNK>")
            pos = IDX2POS.get(t, "X")
            if word.startswith("<") and word.endswith(">"):
                continue  # skip special tokens
            parts.append(f"{word}/{pos}")
        return " ".join(parts)

    def evaluate_grammar(self, words: List[int], types: List[int]) -> Dict:
        """
        Evaluate grammatical coherence of a generated sequence.

        Returns counts of various grammatical patterns.
        """
        metrics = {
            "det_noun": 0,      # DET followed by NOUN-like within 2
            "det_non_noun": 0,  # DET not followed by NOUN-like within 2
            "aux_verb": 0,      # AUX followed by VERB-like within 2
            "adj_noun": 0,      # ADJ followed by NOUN within 2
            "prep_noun": 0,     # PREP followed by NOUN-like within 3
            "double_det": 0,    # Two DETs in a row
            "double_prep": 0,   # Two PREPs in a row
            "noun_verb": 0,     # NOUN-VERB pattern
        }

        NOUN_LIKE = {POS2IDX[t] for t in ["NOUN", "PRON", "NUM"]}
        VERB_LIKE = {POS2IDX[t] for t in ["VERB", "AUX"]}

        for i, t in enumerate(types):
            if t == POS2IDX["DET"]:
                # Check next 2 positions for noun
                found_noun = False
                for d in range(1, 3):
                    if i + d < len(types) and types[i+d] in NOUN_LIKE:
                        found_noun = True
                        break
                if found_noun:
                    metrics["det_noun"] += 1
                else:
                    metrics["det_non_noun"] += 1

                # Check for double DET
                if i + 1 < len(types) and types[i+1] == POS2IDX["DET"]:
                    metrics["double_det"] += 1

            if t == POS2IDX["AUX"]:
                found_verb = False
                for d in range(1, 3):
                    if i + d < len(types) and types[i+d] in VERB_LIKE:
                        found_verb = True
                        break
                if found_verb:
                    metrics["aux_verb"] += 1

            if t == POS2IDX["ADJ"]:
                for d in range(1, 3):
                    if i + d < len(types) and types[i+d] == POS2IDX["NOUN"]:
                        metrics["adj_noun"] += 1
                        break

            if t == POS2IDX["PREP"]:
                found_noun = False
                for d in range(1, 4):
                    if i + d < len(types) and types[i+d] in NOUN_LIKE | {POS2IDX["DET"]}:
                        found_noun = True
                        break
                if found_noun:
                    metrics["prep_noun"] += 1

                if i + 1 < len(types) and types[i+1] == POS2IDX["PREP"]:
                    metrics["double_prep"] += 1

            if t == POS2IDX["NOUN"]:
                for d in range(1, 3):
                    if i + d < len(types) and types[i+d] in VERB_LIKE:
                        metrics["noun_verb"] += 1
                        break

        return metrics

    def save(self, directory: str):
        """Save all model components to directory."""
        os.makedirs(directory, exist_ok=True)
        self.vocab.save(os.path.join(directory, "vocab.json"))
        self.pmi.save(os.path.join(directory, "pmi"))
        self.types.save(os.path.join(directory, "types"))
        self.semantics.save(os.path.join(directory, "semantics"))

        # Save model config
        import json
        config = {
            "vocab_min_freq": self.vocab_min_freq,
            "vocab_max_size": self.vocab_max_size,
            "seq_len": self.seq_len,
            "window": self.window,
            "pmi_cap": self.pmi_cap,
            "min_cooc": self.min_cooc,
            "pmi_weight": self.pmi_weight,
            "hebbian_weight": self.hebbian_weight,
            "semantic_weight": self.semantic_weight,
            "grammar_penalty": self.grammar_penalty,
            "phase1_beta": self.phase1_beta,
            "phase2_beta": self.phase2_beta,
            "phase3_beta": self.phase3_beta,
            "total_sweeps": self.total_sweeps,
        }
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, directory: str) -> "TypedIsingModel":
        """Load model from directory."""
        import json
        with open(os.path.join(directory, "config.json")) as f:
            config = json.load(f)

        model = cls(**config)
        model.vocab = Vocabulary.load(os.path.join(directory, "vocab.json"))
        model.pmi = PMICouplings.load(os.path.join(directory, "pmi"))
        model.types = POSTypeSystem.load(os.path.join(directory, "types"))
        model.semantics = SemanticTypeSystem.load(os.path.join(directory, "semantics"))

        model.sampler = StagedAnnealingSampler(
            pmi_couplings=model.pmi,
            type_system=model.types,
            semantic_system=model.semantics,
            phase1_beta=model.phase1_beta,
            phase2_beta=model.phase2_beta,
            phase3_beta=model.phase3_beta,
            total_sweeps=model.total_sweeps,
            pmi_weight=model.pmi_weight,
            hebbian_weight=model.hebbian_weight,
            semantic_weight=model.semantic_weight,
        )

        return model
