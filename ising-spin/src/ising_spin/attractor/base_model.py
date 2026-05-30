"""
Base Language Model Interface for v76 EBM Re-ranker.

Provides a thin wrapper around a frozen pretrained LM (GPT-2 124M)
that produces top-K candidates with log-probabilities.

The base model is NEVER trained — only used for inference.
The DAM discriminator re-ranks its outputs.

Also provides DummyBaseLM for testing without torch/transformers.
"""

import numpy as np
from typing import List, Tuple, Optional


class BaseLMInterface:
    """
    Interface to a pretrained language model (GPT-2 124M).

    Provides:
      - Tokenization (uses the model's own tokenizer)
      - Top-K candidate generation with log-probabilities
      - Full sequence log-probability computation (for PPL)

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

    Uses simple unigram frequency to generate candidates.
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
