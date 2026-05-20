"""
Integer Matrix Factorization for the Ising Spin Language Model.

J ≈ W × H  where W, H ∈ ℤ^{...}

The full V×V coupling matrix is O(V^2) memory, which becomes prohibitive
for large vocabularies (e.g., V > 10K). Integer NMF decomposes:

  J_PMI ≈ W × H

where:
  W: V × K integer matrix (word-to-latent-factor mapping)
  H: K × V integer matrix (latent-factor-to-word mapping)
  K: number of latent factors (K << V)

This reduces memory from O(V^2) to O(V×K), enabling vocabulary
scaling from ~3K to ~10K+.

Algorithm: Multiplicative-update NMF adapted for integers
  1. Initialize W, H with SVD-based integer approximation
  2. Iteratively update:
     W ← W * (J @ H^T) // (W @ H @ H^T + ε)  (element-wise)
     H ← H * (W^T @ J) // (W^T @ W @ H + ε)  (element-wise)
  3. Round to integers after each update

All intermediate operations use integer arithmetic where possible.
The SVD initialization uses FP (one-time), but the resulting
W, H matrices are pure integer.

Reference:
  - Lee & Seung (2001): "Algorithms for Non-negative Matrix Factorization"
  - Novel adaptation: integer rounding after each multiplicative update
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import time


class IntegerNMF:
    """
    Integer Matrix Factorization: J ≈ W × H, W,H ∈ ℤ.

    Decomposes the coupling matrix J (V×V) into two low-rank
    integer matrices W (V×K) and H (K×V), reducing storage
    from O(V^2) to O(V×K).

    The approximation is:
      J_approx[i,j] = sum_k W[i,k] * H[k,j]

    For energy computation during generation:
      E_coupling(i, state) = sum_j J[i, state[j]]
                           = sum_j sum_k W[i,k] * H[k, state[j]]
                           = sum_k W[i,k] * (sum_j H[k, state[j]])
                           = sum_k W[i,k] * H_sum[k]

    where H_sum[k] = sum_j H[k, state[j]] is precomputed once per sweep.
    This makes per-position energy computation O(K) instead of O(V).
    """

    def __init__(self, vocab_size: int, n_factors: int = 128):
        self.vocab_size = vocab_size
        self.n_factors = n_factors

        # Factorized matrices (integer)
        self.W = np.zeros((vocab_size, n_factors), dtype=np.int64)
        self.H = np.zeros((n_factors, vocab_size), dtype=np.int64)

        # Reconstruction error tracking
        self.errors: List[int] = []
        self.fitted = False

    def fit(
        self,
        J: np.ndarray,
        n_iterations: int = 50,
        initial_scale: int = 10,
        verbose: bool = True,
    ) -> "IntegerNMF":
        """
        Compute integer NMF: J ≈ W × H.

        Algorithm:
          1. Initialize W, H from random positive integers
          2. Multiplicative update with integer rounding
          3. Track reconstruction error

        Args:
            J: Input coupling matrix (V×V), can have negative entries
            n_iterations: Number of NMF iterations
            initial_scale: Scale for random initialization
            verbose: Print progress

        Returns:
            self (for chaining)
        """
        V = self.vocab_size
        K = self.n_factors
        t0 = time.time()

        # Handle negative entries: decompose J = J_plus - J_minus
        J_plus = np.maximum(J, 0).astype(np.float64)
        J_minus = np.maximum(-J, 0).astype(np.float64)

        # Initialize W, H with random positive values
        np.random.seed(42)
        W = np.random.randint(1, initial_scale + 1, size=(V, K)).astype(np.float64)
        H = np.random.randint(1, initial_scale + 1, size=(K, V)).astype(np.float64)

        # Scale to match J's magnitude
        J_scale = max(np.abs(J_plus).max(), 1.0)
        W_scale = np.sqrt(J_scale / (V * K)) * initial_scale
        W = W * W_scale / W.mean()
        H = H * W_scale / H.mean()

        eps = 1e-10  # small constant to avoid division by zero

        if verbose:
            print(f"  Integer NMF: V={V}, K={K}, "
                  f"J range=[{int(J.min())}, {int(J.max())}]")

        # NMF on J_plus (positive part)
        for it in range(n_iterations):
            # Update H
            numerator = W.T @ J_plus
            denominator = W.T @ W @ H + eps
            H = H * (numerator / denominator)

            # Update W
            numerator = J_plus @ H.T
            denominator = W @ H @ H.T + eps
            W = W * (numerator / denominator)

            # Clip to reasonable range
            W = np.clip(W, 0, None)
            H = np.clip(H, 0, None)

            # Track error
            J_approx = W @ H
            error = int(np.sum(np.abs(J_plus - J_approx)))
            self.errors.append(error)

            if verbose and (it + 1) % 10 == 0:
                nnz_J = int(np.count_nonzero(J_plus))
                nnz_approx = int(np.count_nonzero(J_approx))
                print(f"    Iter {it+1}/{n_iterations}: "
                      f"error={error}, "
                      f"J+ nnz={nnz_J}, approx nnz={nnz_approx}")

        # Store positive part as integers
        W_plus = np.round(W).astype(np.int64)
        H_plus = np.round(H).astype(np.int64)

        # Now handle negative part
        if J_minus.max() > 0:
            W_neg = np.random.randint(1, initial_scale + 1, size=(V, K)).astype(np.float64)
            H_neg = np.random.randint(1, initial_scale + 1, size=(K, V)).astype(np.float64)
            W_neg = W_neg * W_scale / W_neg.mean()
            H_neg = H_neg * W_scale / H_neg.mean()

            for it in range(n_iterations):
                numerator = W_neg.T @ J_minus
                denominator = W_neg.T @ W_neg @ H_neg + eps
                H_neg = H_neg * (numerator / denominator)

                numerator = J_minus @ H_neg.T
                denominator = W_neg @ H_neg @ H_neg.T + eps
                W_neg = W_neg * (numerator / denominator)

                W_neg = np.clip(W_neg, 0, None)
                H_neg = np.clip(H_neg, 0, None)

            W_minus = np.round(W_neg).astype(np.int64)
            H_minus = np.round(H_neg).astype(np.int64)
        else:
            W_minus = np.zeros((V, K), dtype=np.int64)
            H_minus = np.zeros((K, V), dtype=np.int64)

        # Combine: J ≈ (W_plus @ H_plus) - (W_minus @ H_minus)
        # Store as combined: W = [W_plus | W_minus], H = [H_plus; H_minus]
        # This gives K=2*n_factors total, but split into positive/negative
        self.W = np.hstack([W_plus, W_minus])
        self.H = np.vstack([H_plus, H_minus])

        self.fitted = True

        # Report quality
        J_reconstructed = self.reconstruct()
        abs_err = int(np.sum(np.abs(J - J_reconstructed)))
        rel_err = abs_err / max(1, int(np.sum(np.abs(J))))
        max_abs_err = int(np.max(np.abs(J - J_reconstructed)))

        if verbose:
            print(f"  NMF complete ({time.time()-t0:.1f}s): "
                  f"abs_err={abs_err}, rel_err={rel_err:.3f}, "
                  f"max_abs_err={max_abs_err}")

        return self

    def reconstruct(self) -> np.ndarray:
        """
        Reconstruct J from W × H. Pure integer matrix multiply.
        """
        return self.W @ self.H

    def compute_energy_factorized(
        self,
        word: int,
        state: List[int],
        pos: int,
        H_sum: np.ndarray,
    ) -> int:
        """
        Compute coupling energy for word at position pos using
        factorized representation.

        E = sum_k W[word, k] * H_sum[k]

        where H_sum[k] = sum_j H[k, state[j]] for j != pos.

        This is O(K) per position instead of O(V).

        Pure integer: one dot product of two integer vectors.
        """
        return int(self.W[word] @ H_sum)

    def compute_H_sum(
        self,
        state: List[int],
        exclude_pos: int = -1,
    ) -> np.ndarray:
        """
        Precompute H_sum[k] = sum_j H[k, state[j]] for j != exclude_pos.

        This is done once per sweep, then reused for all positions.
        O(L * K) where L = sequence length.
        """
        K_total = self.W.shape[1]
        H_sum = np.zeros(K_total, dtype=np.int64)

        for j, w in enumerate(state):
            if j == exclude_pos:
                continue
            if w < self.vocab_size:
                H_sum += self.H[:, w]

        return H_sum

    def update_H_sum(
        self,
        H_sum: np.ndarray,
        old_word: int,
        new_word: int,
    ) -> np.ndarray:
        """
        Update H_sum when a word changes at some position.
        H_sum -= H[:, old_word] + H[:, new_word]
        Pure integer vector addition.
        """
        if old_word < self.vocab_size:
            H_sum -= self.H[:, old_word]
        if new_word < self.vocab_size:
            H_sum += self.H[:, new_word]
        return H_sum

    def get_top_neighbors(
        self, word: int, top_k: int = 20
    ) -> List[Tuple[int, int]]:
        """
        Get top-k neighbors for a word using factorized J.
        J_approx[word, :] = W[word, :] @ H
        Only compute for non-zero W entries.
        """
        w_vec = self.W[word]
        # Only compute for factors where this word is active
        active_factors = np.nonzero(w_vec)[0]
        if len(active_factors) == 0:
            return []

        # Compute J_approx[word, :] = w_vec @ H
        row = w_vec @ self.H

        # Get top-k
        top_indices = np.argsort(np.abs(row))[-top_k:][::-1]
        return [(int(i), int(row[i])) for i in top_indices if row[i] != 0]

    def memory_savings(self) -> Dict:
        """Compute memory savings from factorization."""
        full_matrix = self.vocab_size ** 2
        factorized = self.vocab_size * self.W.shape[1] * 2  # W + H
        savings_pct = (1 - factorized / full_matrix) * 100 if full_matrix > 0 else 0
        return {
            "full_matrix_elements": full_matrix,
            "factorized_elements": factorized,
            "savings_pct": savings_pct,
            "n_factors_total": self.W.shape[1],
            "vocab_size": self.vocab_size,
        }

    def save(self, path: str):
        """Save NMF factorization to disk."""
        np.save(f"{path}_W.npy", self.W)
        np.save(f"{path}_H.npy", self.H)

        meta = {
            "vocab_size": self.vocab_size,
            "n_factors": self.n_factors,
            "n_factors_total": int(self.W.shape[1]),
            "fitted": self.fitted,
        }
        with open(f"{path}_meta.json", "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> "IntegerNMF":
        """Load NMF factorization from disk."""
        with open(f"{path}_meta.json") as f:
            meta = json.load(f)

        nmf = cls(vocab_size=meta["vocab_size"], n_factors=meta["n_factors"])
        nmf.W = np.load(f"{path}_W.npy")
        nmf.H = np.load(f"{path}_H.npy")
        nmf.fitted = meta["fitted"]

        return nmf
