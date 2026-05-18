"""
SpaCy-based POS tagger and dependency parser for the Ising Spin Language Model.

Replaces the rule-based POS assignment in type_system.py with accurate
spaCy tagging. Also extracts dependency tree edges for long-range
subject-verb agreement couplings (J_tree).

All extracted data is stored as integer indices and counts — no FP
in the resulting data structures.

Usage during training (one-time, FP allowed):
    tagger = SpaCyTagger(vocab)
    tagger.tag_corpus(texts, sequences)
    # tagger.word_pos[w] = {POS_IDX: count, ...}  (integer)
    # tagger.dep_edges = [(head_w, dep_w, dep_label_idx, dist), ...]

Usage during generation (zero FP):
    # Use precomputed word_pos and dep_edges as integer lookup tables
"""

import spacy
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple
import json
import numpy as np

from .type_system import POS2IDX, IDX2POS, N_POS, COARSE_POS_TAGS


# Mapping from spaCy fine-grained POS to our coarse POS tags
SPACY_TO_COARSE = {
    # Nouns
    "NOUN": "NOUN", "PROPN": "NOUN",
    # Verbs
    "VERB": "VERB",
    # Adjectives
    "ADJ": "ADJ",
    # Adverbs
    "ADV": "ADV",
    # Determiners
    "DET": "DET",
    # Prepositions (spaCy lumps IN as ADP)
    "ADP": "PREP",
    # Pronouns
    "PRON": "PRON",
    # Auxiliaries / Modals
    "AUX": "AUX",
    # Conjunctions
    "CCONJ": "CONJ", "SCONJ": "CONJ",
    # Particles
    "PART": "PART",
    # Numbers
    "NUM": "NUM",
    # Punctuation
    "PUNCT": "PUNCT",
    # Other
    "SYM": "X", "X": "X", "SPACE": "X", "INTJ": "X",
}

# Dependency labels relevant for long-range agreement
# These are the edges we extract for J_tree coupling
DEP_LABELS_FOR_TREE = {
    "nsubj": 0,      # nominal subject (subject-verb)
    "nsubjpass": 1,  # passive nominal subject
    "csubj": 2,      # clausal subject
    "dobj": 3,       # direct object (verb-object)
    "iobj": 4,       # indirect object
    "nmod": 5,       # nominal modifier (noun-noun)
    "amod": 6,       # adjectival modifier (adj-noun)
    "det": 7,        # determiner (det-noun)
    "aux": 8,        # auxiliary (aux-verb)
    "cop": 9,        # copula (cop-pred)
    "compound": 10,  # compound (noun-noun)
    "acl": 11,       # clausal modifier of noun
    "advcl": 12,     # adverbial clause modifier
    "ccomp": 13,     # clausal complement
    "xcomp": 14,     # open clausal complement
    "mark": 15,      # marker
}

N_DEP_LABELS = len(DEP_LABELS_FOR_TREE)
DEP_IDX2LABEL = {v: k for k, v in DEP_LABELS_FOR_TREE.items()}


class SpaCyTagger:
    """
    SpaCy-based POS tagger and dependency parser.

    Provides:
      1. Accurate POS tags per word (word_pos[w] = {pos_idx: count})
      2. Dependency tree edges for J_tree long-range couplings
      3. POS bigram counts for J_type couplings (more accurate than rule-based)
    """

    def __init__(self, vocab_size: int, n_pos: int = N_POS):
        self.vocab_size = vocab_size
        self.n_pos = n_pos

        # word -> {pos_idx: count} — integer counts of POS assignments
        self.word_pos: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        # POS bigram counts (t1, t2) -> count
        self.pos_bigram_counts: Dict[Tuple[int, int], int] = defaultdict(int)

        # Dependency tree edges extracted from parsed corpus
        # Each edge: (head_word_idx, dep_word_idx, dep_label_idx, distance)
        self.dep_edges: List[Tuple[int, int, int, int]] = []

        # Dependency label coupling counts: (dep_label, head_pos, dep_pos) -> count
        self.dep_label_pos_counts: Dict[Tuple[int, int, int], int] = defaultdict(int)

        # Sentence-level dependency pairs (head, dep) for Hebbian-style coupling
        self.dep_pair_counts: Dict[Tuple[int, int, int], int] = defaultdict(int)
        # (head_w, dep_w, dep_label_idx) -> count

        # SpaCy pipeline (lazy init)
        self._nlp = None

    def _get_nlp(self):
        """Lazy-load spaCy pipeline."""
        if self._nlp is None:
            self._nlp = spacy.load("en_core_web_sm")
            # Disable unnecessary components for speed
            # NOTE: Keep tagger + attribute_ruler + parser — they're needed for .pos_ and .dep_
            available = self._nlp.pipe_names
            disable = [c for c in ["ner", "lemmatizer", "senter"] if c in available]
            self._nlp.select_pipes(disable=disable)
        return self._nlp

    def _coarse_pos(self, spacy_pos: str) -> int:
        """Map spaCy POS to coarse POS index."""
        coarse = SPACY_TO_COARSE.get(spacy_pos, "X")
        return POS2IDX.get(coarse, POS2IDX["X"])

    def _dep_label_idx(self, dep: str) -> int:
        """Get index for dependency label, or -1 if not relevant."""
        return DEP_LABELS_FOR_TREE.get(dep, -1)

    def tag_corpus(
        self,
        texts: List[str],
        sequences: List[List[int]],
        word2idx: Dict[str, int],
        idx2word: Dict[int, str],
        max_texts: Optional[int] = None,
        batch_size: int = 1000,
    ) -> "SpaCyTagger":
        """
        Tag a corpus using spaCy for accurate POS and dependency parsing.

        This is a one-time training operation. All results are stored as
        integer counts.

        Args:
            texts: Raw text strings from the corpus
            sequences: Tokenized integer sequences (aligned with texts)
            word2idx: Word-to-index mapping
            idx2word: Index-to-word mapping
            max_texts: Maximum number of texts to process (None = all)
            batch_size: spaCy batch size for processing

        Returns:
            self (for chaining)
        """
        nlp = self._get_nlp()
        n_process = max(1, min(4, len(texts) // 500))

        texts_to_process = texts[:max_texts] if max_texts else texts
        sequences_to_process = sequences[:max_texts] if max_texts else sequences

        print(f"  SpaCy tagging {len(texts_to_process)} texts...")
        total_edges = 0
        total_tokens_tagged = 0

        # Process in batches for efficiency
        for batch_start in range(0, len(texts_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(texts_to_process))
            batch_texts = texts_to_process[batch_start:batch_end]
            batch_seqs = sequences_to_process[batch_start:batch_end]

            docs = list(nlp.pipe(batch_texts, batch_size=min(50, len(batch_texts))))

            for doc, seq in zip(docs, batch_seqs):
                # Build token -> word_idx mapping from sequence
                # We need to align spaCy tokens with our vocabulary indices
                spacy_tokens = list(doc)
                token_idx_map = self._align_tokens(spacy_tokens, seq, word2idx)

                # Extract POS tags
                prev_pos = None
                for spacy_idx, spacy_tok in enumerate(spacy_tokens):
                    word_idx = token_idx_map.get(spacy_idx)
                    if word_idx is None:
                        continue

                    pos_idx = self._coarse_pos(spacy_tok.pos_)
                    self.word_pos[word_idx][pos_idx] += 1
                    total_tokens_tagged += 1

                    # POS bigram
                    if prev_pos is not None:
                        self.pos_bigram_counts[(prev_pos, pos_idx)] += 1
                    prev_pos = pos_idx

                # Extract dependency edges
                for spacy_idx, spacy_tok in enumerate(spacy_tokens):
                    dep_idx = self._dep_label_idx(spacy_tok.dep_)
                    if dep_idx < 0:
                        continue

                    head_tok = spacy_tok.head
                    if head_tok == spacy_tok:
                        continue  # skip ROOT

                    # Map to our word indices
                    w_dep = token_idx_map.get(spacy_idx)
                    w_head = token_idx_map.get(head_tok.i)

                    if w_dep is None or w_head is None:
                        continue
                    if w_dep >= self.vocab_size or w_head >= self.vocab_size:
                        continue

                    # Distance (could be long-range!)
                    dist = abs(spacy_idx - head_tok.i)

                    # Store edge
                    self.dep_edges.append((w_head, w_dep, dep_idx, dist))
                    total_edges += 1

                    # Store dependency label + POS counts
                    dep_pos = self._coarse_pos(spacy_tok.pos_)
                    head_pos = self._coarse_pos(head_tok.pos_)
                    self.dep_label_pos_counts[(dep_idx, head_pos, dep_pos)] += 1

                    # Store pair counts for J_tree
                    self.dep_pair_counts[(w_head, w_dep, dep_idx)] += 1

            if (batch_start + batch_size) % 5000 < batch_size:
                print(f"    Processed {batch_end}/{len(texts_to_process)} texts, "
                      f"{total_edges} dep edges, {total_tokens_tagged} tokens tagged")

        print(f"  SpaCy tagging complete: {total_tokens_tagged} tokens, "
              f"{total_edges} dependency edges")

        return self

    def _align_tokens(
        self,
        spacy_tokens,
        seq: List[int],
        word2idx: Dict[str, int],
    ) -> Dict[int, int]:
        """
        Align spaCy tokens with vocabulary indices.

        Returns: {spacy_token_index: vocab_word_index}
        """
        token_map = {}
        for i, tok in enumerate(spacy_tokens):
            text = tok.text.lower().strip()
            if text in word2idx:
                token_map[i] = word2idx[text]
        return token_map

    def build_emission_weights(self) -> np.ndarray:
        """
        Build I_emit matrix from accumulated POS counts.

        I_emit[w, t] = count(word w tagged as POS t)
        Pure integer matrix.
        """
        I_emit = np.zeros((self.vocab_size, self.n_pos), dtype=np.int64)
        for word_idx, pos_counts in self.word_pos.items():
            if word_idx < self.vocab_size:
                for pos_idx, count in pos_counts.items():
                    if pos_idx < self.n_pos:
                        I_emit[word_idx, pos_idx] = count
        return I_emit

    def build_allowed_types(self) -> Dict[int, Set[int]]:
        """
        Build allowed_types mapping from accumulated POS counts.

        A word is allowed to have POS tag t if it was observed with
        that tag at least once in the corpus.
        """
        allowed = {}
        for word_idx, pos_counts in self.word_pos.items():
            if word_idx < self.vocab_size:
                allowed[word_idx] = set(pos_counts.keys())
        return allowed

    def build_type_couplings(self, scaling: int = 10) -> np.ndarray:
        """
        Build J_type matrix from POS bigram counts.

        J_type[t1, t2] = count(t1 followed by t2) * scaling
        Pure integer matrix.
        """
        J_type = np.zeros((self.n_pos, self.n_pos), dtype=np.int64)
        for (t1, t2), count in self.pos_bigram_counts.items():
            if t1 < self.n_pos and t2 < self.n_pos:
                J_type[t1, t2] = count * scaling
        return J_type

    def save(self, path: str):
        """Save SpaCy tagger data to disk."""
        # Save word_pos as sparse data
        word_pos_ser = {}
        for w, pos_counts in self.word_pos.items():
            word_pos_ser[str(w)] = {str(t): int(c) for t, c in pos_counts.items()}
        with open(f"{path}_word_pos.json", "w") as f:
            json.dump(word_pos_ser, f)

        # Save POS bigrams
        pos_bigram_ser = {f"{t1},{t2}": int(c) for (t1, t2), c in self.pos_bigram_counts.items()}
        with open(f"{path}_pos_bigram.json", "w") as f:
            json.dump(pos_bigram_ser, f)

        # Save dep edges as numpy (more efficient than JSON for large lists)
        if self.dep_edges:
            dep_arr = np.array(self.dep_edges, dtype=np.int64)
            np.save(f"{path}_dep_edges.npy", dep_arr)

        # Save dep pair counts
        dep_pair_ser = {f"{h},{d},{l}": int(c) for (h, d, l), c in self.dep_pair_counts.items()}
        with open(f"{path}_dep_pairs.json", "w") as f:
            json.dump(dep_pair_ser, f)

        # Save dep label + POS counts
        dep_pos_ser = {f"{l},{hp},{dp}": int(c) for (l, hp, dp), c in self.dep_label_pos_counts.items()}
        with open(f"{path}_dep_label_pos.json", "w") as f:
            json.dump(dep_pos_ser, f)

        meta = {
            "vocab_size": self.vocab_size,
            "n_pos": self.n_pos,
            "n_dep_edges": len(self.dep_edges),
            "n_dep_pairs": len(self.dep_pair_counts),
        }
        with open(f"{path}_meta.json", "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> "SpaCyTagger":
        """Load SpaCy tagger data from disk."""
        with open(f"{path}_meta.json") as f:
            meta = json.load(f)

        tagger = cls(vocab_size=meta["vocab_size"], n_pos=meta["n_pos"])

        with open(f"{path}_word_pos.json") as f:
            word_pos_ser = json.load(f)
        tagger.word_pos = {
            int(w): {int(t): int(c) for t, c in pos_counts.items()}
            for w, pos_counts in word_pos_ser.items()
        }

        try:
            with open(f"{path}_pos_bigram.json") as f:
                pos_bigram_ser = json.load(f)
            tagger.pos_bigram_counts = {
                tuple(map(int, k.split(","))): v for k, v in pos_bigram_ser.items()
            }
        except FileNotFoundError:
            pass

        try:
            dep_arr = np.load(f"{path}_dep_edges.npy")
            tagger.dep_edges = [tuple(row) for row in dep_arr]
        except FileNotFoundError:
            pass

        try:
            with open(f"{path}_dep_pairs.json") as f:
                dep_pair_ser = json.load(f)
            tagger.dep_pair_counts = {
                tuple(map(int, k.split(","))): v for k, v in dep_pair_ser.items()
            }
        except FileNotFoundError:
            pass

        try:
            with open(f"{path}_dep_label_pos.json") as f:
                dep_pos_ser = json.load(f)
            tagger.dep_label_pos_counts = {
                tuple(map(int, k.split(","))): v for k, v in dep_pos_ser.items()
            }
        except FileNotFoundError:
            pass

        return tagger
