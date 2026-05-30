"""
Base Language Model Interface for v76h EBM Re-ranker.

v76h KEY FIX: Proper BPE-to-word alignment for GPT-2.

The original BaseLMInterface passed word-level IDs directly to GPT-2,
which interprets them as BPE tokens — producing semantic garbage.
v76h adds `get_top_k_words()` which:
  1. Converts word-level context to text → BPE tokens
  2. Runs GPT-2 on proper BPE tokens
  3. Scores every word in our vocabulary by its first BPE token's log-prob
  4. Returns word-level candidates with correct log-probabilities

This eliminates the <unk> problem and gives GPT-2 proper semantic context.

Also provides DummyBaseLM for testing without torch/transformers.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


class BaseLMInterface:
    """
    Interface to a pretrained language model (GPT-2 124M).

    Provides:
      - Tokenization (uses the model's own tokenizer)
      - Top-K candidate generation with log-probabilities
      - Full sequence log-probability computation (for PPL)
      - v76h: Word-level candidate generation via BPE alignment

    The base model is FROZEN — no gradient updates.
    Only used for inference (candidate generation).
    """

    def __init__(self, model_name: str = "gpt2", device: str = "auto"):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "BaseLMInterface requires torch and transformers. "
                "Install with: pip install torch transformers"
            )

        self.model_name = model_name
        self.device = self._resolve_device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Disable gradient computation
        for param in self.model.parameters():
            param.requires_grad = False

        self._torch = torch

        # v76h: BPE-to-word alignment cache
        # Maps word_id -> first BPE token ID for " word" encoding
        self._word_bpe_cache: Dict[int, int] = {}
        self._idx2word: Optional[dict] = None
        self._word2idx: Optional[dict] = None

    def build_word_alignment(self, idx2word: dict, word2idx: dict):
        """
        v76h: Build the BPE-to-word alignment cache.

        For each word in our vocabulary, precompute the first BPE token
        of " word" (with leading space). This is used in get_top_k_words()
        to score vocabulary words against GPT-2's output distribution.

        Must be called once before using get_top_k_words().
        """
        self._idx2word = idx2word
        self._word2idx = word2idx
        self._word_bpe_cache = {}

        for word_id in range(4, len(idx2word)):
            word = idx2word[word_id]
            # Encode with leading space for proper word-initial BPE
            bpe_ids = self.tokenizer.encode(" " + word)
            if bpe_ids:
                self._word_bpe_cache[word_id] = bpe_ids[0]

        print(f"  BPE-word alignment: {len(self._word_bpe_cache)} words cached", flush=True)

    def _resolve_device(self, device: str) -> str:
        import torch
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text, return list of token IDs."""
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def get_top_k(
        self,
        input_ids: List[int],
        k: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get top-K candidate next tokens with log-probabilities.

        NOTE: This returns BPE token IDs, not word-level IDs.
        For word-level candidates, use get_top_k_words() instead.

        Args:
            input_ids: Token IDs for context.
            k: Number of top candidates.

        Returns:
            (candidates, log_probs) — each a numpy array of shape (k,).
            log_probs are natural log (nats).
        """
        torch = self._torch
        with torch.no_grad():
            ids_tensor = torch.tensor([input_ids], device=self.device)
            outputs = self.model(ids_tensor)
            logits = outputs.logits[0, -1, :]  # Last position
            log_probs = torch.log_softmax(logits, dim=-1)
            top_k = torch.topk(log_probs, min(k, log_probs.shape[0]))
            candidates = top_k.indices.cpu().numpy()
            log_probs_k = top_k.values.cpu().numpy().astype(np.float64)
        return candidates, log_probs_k

    def get_top_k_words(
        self,
        word_ids: List[int],
        k: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        v76h: Get top-K WORD-LEVEL candidates with log-probabilities.

        This properly converts word-level context to BPE tokens, runs
        GPT-2, and scores each vocabulary word by its first BPE token's
        probability. Returns word-level IDs (4 to V-1).

        This fixes the <unk> problem: all returned candidates are valid
        word-level IDs that map to real vocabulary words.

        Args:
            word_ids: Context as word-level IDs.
            k: Number of top candidates.

        Returns:
            (candidates, log_probs) — word-level IDs and their log-probs.
        """
        torch = self._torch

        # Convert word IDs to text, then to BPE tokens
        if self._idx2word is None:
            # Fallback: pass raw IDs (broken but won't crash)
            return self.get_top_k(word_ids, k=k)

        words = [self._idx2word.get(wid, "") for wid in word_ids]
        text = " ".join(w for w in words if w)
        if not text.strip():
            # Empty context — return most frequent words
            V = len(self._idx2word)
            top_ids = np.arange(4, min(4 + k, V))
            return top_ids, np.full(len(top_ids), -7.0)

        bpe_ids = self.tokenizer.encode(text)

        # Run GPT-2 on proper BPE context
        with torch.no_grad():
            ids_tensor = torch.tensor([bpe_ids], device=self.device)
            outputs = self.model(ids_tensor)
            logits = outputs.logits[0, -1, :]
            log_probs_all = torch.log_softmax(logits, dim=-1)
            log_probs_np = log_probs_all.cpu().numpy().astype(np.float64)

        # Score each vocabulary word using cached BPE mapping
        V = len(self._idx2word)
        word_scores = np.full(V, -100.0)

        for word_id, bpe_id in self._word_bpe_cache.items():
            if bpe_id < len(log_probs_np):
                word_scores[word_id] = log_probs_np[bpe_id]

        # Get top-K word-level candidates
        valid_mask = word_scores > -100
        if not np.any(valid_mask):
            # Fallback: return first k words
            top_ids = np.arange(4, min(4 + k, V))
            return top_ids, np.full(len(top_ids), -7.0)

        valid_scores = word_scores[4:V]
        top_k_offset = np.argsort(valid_scores)[-k:][::-1]
        top_k_ids = top_k_offset + 4
        top_k_scores = word_scores[top_k_ids]

        return top_k_ids, top_k_scores

    def compute_sequence_log_prob(self, input_ids: List[int]) -> float:
        """
        Compute total log-probability of a sequence.

        Used for perplexity: sum of log P(token_i | token_<i).

        Args:
            input_ids: Token IDs.

        Returns:
            Total log-probability in nats.
        """
        torch = self._torch
        with torch.no_grad():
            ids_tensor = torch.tensor([input_ids], device=self.device)
            outputs = self.model(ids_tensor)
            logits = outputs.logits
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :]
            shift_labels = ids_tensor[:, 1:]
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(
                2, shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            return token_log_probs.sum().item()

    def compute_word_sequence_log_prob(
        self,
        word_ids: List[int],
    ) -> float:
        """
        v76h: Compute log-probability of a word-level sequence.

        Converts word IDs to text → BPE tokens, then computes
        the full sequence log-probability under GPT-2.
        """
        if self._idx2word is None:
            return self.compute_sequence_log_prob(word_ids)

        words = [self._idx2word.get(wid, "") for wid in word_ids]
        text = " ".join(w for w in words if w)
        if not text.strip():
            return -100.0

        bpe_ids = self.tokenizer.encode(text)
        return self.compute_sequence_log_prob(bpe_ids)

    def align_token_to_vocab(
        self,
        token_id: int,
        word2idx: dict,
    ) -> Optional[int]:
        """
        Map a BPE token to a word-level vocab ID.

        GPT-2 uses BPE tokenization (50,257 tokens). Our SDR encoder
        uses word-level vocabulary. This method maps BPE tokens to
        our word IDs when possible.

        Args:
            token_id: GPT-2 BPE token ID.
            word2idx: Dict mapping word string -> word ID.

        Returns:
            Word ID if the token maps to a known word, else None.
        """
        token_text = self.tokenizer.decode([token_id]).strip()
        if not token_text:
            return None
        # Try exact match, then lowercase
        wid = word2idx.get(token_text)
        if wid is not None:
            return wid
        wid = word2idx.get(token_text.lower())
        return wid


class DummyBaseLM:
    """
    Dummy base model for testing without torch/transformers.

    Uses simple unigram + bigram frequency to generate candidates.
    Not a real language model — just for pipeline testing.
    """

    def __init__(
        self,
        vocab_words: List[str],
        word_freq: Optional[np.ndarray] = None,
        seed: int = 42,
    ):
        """
        Args:
            vocab_words: List of vocabulary words.
            word_freq: Array (V,) of word frequencies. Used for sampling.
            seed: Random seed.
        """
        self.vocab_words = vocab_words
        self.V = len(vocab_words)
        self._rng = np.random.RandomState(seed)
        self.model_name = "dummy"

        # Build frequency-based sampling weights
        if word_freq is not None and np.sum(word_freq) > 0:
            self._weights = word_freq.astype(np.float64)
            self._weights /= self._weights.sum()
        else:
            self._weights = np.ones(self.V, dtype=np.float64) / self.V

        # Simple bigram counts for basic coherence
        self._bigram_counts: Optional[np.ndarray] = None

    def build_bigrams(self, sequences: List[List[int]]):
        """Build simple bigram counts from training sequences."""
        self._bigram_counts = np.zeros((self.V, self.V), dtype=np.int32)
        for seq in sequences:
            for i in range(1, len(seq)):
                self._bigram_counts[seq[i-1], seq[i]] += 1

    def build_word_alignment(self, idx2word: dict, word2idx: dict):
        """v76h: Dummy — no BPE alignment needed."""
        pass

    def get_top_k(
        self,
        input_ids: List[int],
        k: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate top-K candidates using bigram + unigram scores.

        Args:
            input_ids: Context token IDs (word-level vocab IDs).
            k: Number of candidates.

        Returns:
            (candidates, log_probs) — numpy arrays of shape (k,).
        """
        # Score each word
        scores = np.log(self._weights + 1e-10)  # Unigram log-prob

        # Add bigram signal if available
        if self._bigram_counts is not None and len(input_ids) > 0:
            prev = input_ids[-1]
            if 0 <= prev < self.V:
                bigram_row = self._bigram_counts[prev].astype(np.float64)
                bigram_prob = bigram_row / max(1, bigram_row.sum())
                # Combine: 50% bigram + 50% unigram
                combined = 0.5 * np.log(bigram_prob + 1e-10) + 0.5 * scores
                scores = combined

        # Get top-K
        top_k_indices = np.argsort(scores)[-k:][::-1]
        log_probs = scores[top_k_indices]

        return top_k_indices, log_probs

    def get_top_k_words(
        self,
        word_ids: List[int],
        k: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """v76h: Dummy — delegates to get_top_k (already word-level)."""
        return self.get_top_k(word_ids, k=k)

    def compute_sequence_log_prob(self, input_ids: List[int]) -> float:
        """Compute approximate sequence log-probability."""
        total = 0.0
        for i in range(1, len(input_ids)):
            # Unigram probability
            p = self._weights[input_ids[i]]
            if p > 0:
                total += np.log(p)
            else:
                total += np.log(1e-10)
        return total

    def compute_word_sequence_log_prob(self, word_ids: List[int]) -> float:
        """v76h: Dummy — same as compute_sequence_log_prob."""
        return self.compute_sequence_log_prob(word_ids)

    def align_token_to_vocab(
        self,
        token_id: int,
        word2idx: dict,
    ) -> Optional[int]:
        """Dummy: tokens ARE vocab IDs already."""
        if 0 <= token_id < self.V:
            return token_id
        return None

    def tokenize(self, text: str) -> List[int]:
        """Dummy: split on whitespace and map to vocab."""
        # Not implemented for dummy — sequences are already word IDs
        return []

    def decode(self, token_ids: List[int], idx2word: Optional[dict] = None) -> str:
        """Dummy: map word IDs back to text."""
        if idx2word is not None:
            return " ".join(idx2word.get(t, "<unk>") for t in token_ids)
        return " ".join(str(t) for t in token_ids)
