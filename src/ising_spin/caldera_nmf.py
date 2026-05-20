"""
CALDERA-style Integer Matrix Factorization for the Ising Spin Language Model.

J ≈ Q + L1 × L2

where:
  Q: sparse integer matrix (top-k strongest couplings stored exactly)
  L1, L2: low-rank integer matrices (correct the approximation error)

This approach (inspired by Saha et al. 2024, arXiv:2405.18886) dramatically
reduces approximation error compared to pure NMF because:
  1. The sparse backbone Q captures the most important couplings EXACTLY
  2. The low-rank residual only needs to approximate weaker couplings,
     which are much more compressible
  3. Memory savings remain high: sparse Q + low-rank L1,L2 << full J

Expected improvement: 66% relative error → <15% relative error.

Energy computation during generation:
  E_coupling(i, state) = Q[word, neighbors]  (sparse lookup, O(nnz_per_row))
                       + L1[word, :] @ H_sum  (dot product, O(K))
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import time
from scipy import sparse


class CalderaNMF:
    """
    CALDERA-style factorization: J = Q + L1 @ L2
    
    Q: sparse matrix storing top-k couplings per row (exact)
    L1, L2: low-rank integer matrices approximating the residual
    """

    def __init__(self, vocab_size: int, n_factors: int = 128, n_top: int = 15):
        self.vocab_size = vocab_size
        self.n_factors = n_factors
        self.n_top = n_top  # number of top couplings to store exactly per row

        # Sparse backbone
        self.Q = None  # scipy sparse matrix (CSR format for fast row access)
        
        # Low-rank residual factors
        self.L1 = np.zeros((vocab_size, n_factors), dtype=np.int64)
        self.L2 = np.zeros((n_factors, vocab_size), dtype=np.int64)
        
        # For fast row access during generation
        self.Q_rows = {}  # word_idx -> [(col_idx, value), ...]
        
        self.fitted = False
        self.errors = []

    def fit(
        self,
        J: np.ndarray,
        n_iterations: int = 50,
        verbose: bool = True,
    ) -> "CalderaNMF":
        """
        Compute CALDERA factorization: J = Q + L1 @ L2.
        
        Step 1: Extract top-n_top couplings per row into sparse Q
        Step 2: Compute residual R = J - Q
        Step 3: Factorize |R| ≈ L1 @ L2 using NMF with SVD init, then round
        Step 4: Handle negative residual entries
        """
        V = self.vocab_size
        K = self.n_factors
        t0 = time.time()

        if verbose:
            print(f"  CALDERA NMF: V={V}, K={K}, n_top={self.n_top}, "
                  f"J range=[{int(J.min())}, {int(J.max())}]")

        # ===== Step 1: Extract sparse backbone Q =====
        Q_data = np.zeros_like(J)
        for i in range(V):
            row = np.abs(J[i])
            if row.sum() == 0:
                continue
            # Find top-n_top absolute values
            n_select = min(self.n_top, int(np.count_nonzero(row)))
            if n_select == 0:
                continue
            top_indices = np.argpartition(row, -n_select)[-n_select:]
            for idx in top_indices:
                if J[i, idx] != 0:
                    Q_data[i, idx] = J[i, idx]
        
        # Make symmetric: Q[i,j] = Q[j,i] = max(|J[i,j]|, |J[j,i]|) with sign
        Q_sym = (Q_data + Q_data.T)
        # For overlapping entries, take the average (integer)
        overlap = (Q_data != 0).astype(int) + (Q_data.T != 0).astype(int)
        overlap = np.maximum(overlap, 1)  # avoid div by 0
        Q_sym = Q_sym // overlap
        
        self.Q = sparse.csr_matrix(Q_sym.astype(np.int64))
        
        # Build fast row lookup
        for i in range(V):
            row_start = self.Q.indptr[i]
            row_end = self.Q.indptr[i + 1]
            if row_end > row_start:
                cols = self.Q.indices[row_start:row_end]
                vals = self.Q.data[row_start:row_end]
                self.Q_rows[i] = [(int(c), int(v)) for c, v in zip(cols, vals)]
            else:
                self.Q_rows[i] = []

        # ===== Step 2: Compute residual =====
        R = J - Q_sym
        R_pos = np.maximum(R, 0).astype(np.float64)
        R_neg = np.maximum(-R, 0).astype(np.float64)

        # ===== Step 3: NMF on positive residual with SVD initialization =====
        L1_pos, L2_pos = self._nmf_with_svd_init(R_pos, K, n_iterations, verbose, "R+")
        
        # ===== Step 4: NMF on negative residual =====
        if R_neg.max() > 0:
            L1_neg, L2_neg = self._nmf_with_svd_init(R_neg, K, n_iterations, verbose, "R-")
        else:
            L1_neg = np.zeros((V, K), dtype=np.int64)
            L2_neg = np.zeros((K, V), dtype=np.int64)

        # Combine: J ≈ Q + (L1_pos @ L2_pos) - (L1_neg @ L2_neg)
        # Store as: L1 = [L1_pos | L1_neg], L2 = [L2_pos; L2_neg]
        self.L1 = np.hstack([L1_pos, L1_neg])
        self.L2 = np.vstack([L2_pos, L2_neg])

        self.fitted = True

        # Report quality
        J_recon = self.reconstruct()
        abs_err = int(np.sum(np.abs(J - J_recon)))
        rel_err = abs_err / max(1, int(np.sum(np.abs(J))))
        max_abs_err = int(np.max(np.abs(J - J_recon)))

        if verbose:
            q_nnz = int(self.Q.nnz)
            print(f"  CALDERA complete ({time.time()-t0:.1f}s): "
                  f"Q_nnz={q_nnz}, "
                  f"abs_err={abs_err:,}, rel_err={rel_err:.3f}, "
                  f"max_abs_err={max_abs_err}")

        return self

    def _nmf_with_svd_init(self, M, K, n_iterations, verbose, label):
        """NMF with improved SVD-based initialization (Syed et al. 2018)."""
        V = M.shape[0]
        eps = 1e-10
        
        # SVD-based initialization (much better than random)
        try:
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            U_k = U[:, :K]
            S_k = S[:K]
            Vt_k = Vt[:K, :]
            
            # Ensure non-negativity via split: neg part → + 
            W_init = np.abs(U_k) * np.sqrt(np.abs(S_k)).reshape(1, -1)
            H_init = np.sqrt(np.abs(S_k)).reshape(-1, 1) * np.abs(Vt_k)
            
            # Remove zeros
            W_init = np.maximum(W_init, eps)
            H_init = np.maximum(H_init, eps)
        except Exception:
            # Fallback to random
            W_init = np.random.rand(V, K) + eps
            H_init = np.random.rand(K, V) + eps

        W = W_init.copy()
        H = H_init.copy()

        for it in range(n_iterations):
            # Update H
            numerator = W.T @ M
            denominator = W.T @ W @ H + eps
            H = H * (numerator / denominator)

            # Update W
            numerator = M @ H.T
            denominator = W @ H @ H.T + eps
            W = W * (numerator / denominator)

            # Clip
            W = np.clip(W, eps, None)
            H = np.clip(H, eps, None)

            if verbose and (it + 1) % 20 == 0:
                error = int(np.sum(np.abs(M - W @ H)))
                print(f"    {label} NMF iter {it+1}/{n_iterations}: error={error}")

        # Round to integers with scale
        scale = 10
        W_int = np.round(W * scale).astype(np.int64)
        H_int = np.round(H * scale).astype(np.int64)

        return W_int, H_int

    def reconstruct(self) -> np.ndarray:
        """Reconstruct J from Q + L1 @ L2."""
        if self.Q is not None:
            return np.array(self.Q.todense()) + self.L1 @ self.L2
        return self.L1 @ self.L2

    def compute_H_sum(self, state: List[int], exclude_pos: int = -1) -> np.ndarray:
        """Precompute H_sum[k] = sum_j L2[k, state[j]] for j != exclude_pos."""
        K_total = self.L1.shape[1]
        H_sum = np.zeros(K_total, dtype=np.int64)
        for j, w in enumerate(state):
            if j == exclude_pos:
                continue
            if w < self.vocab_size:
                H_sum += self.L2[:, w]
        return H_sum

    def update_H_sum(self, H_sum, old_word, new_word):
        """Update H_sum when a word changes."""
        if old_word < self.vocab_size:
            H_sum -= self.L2[:, old_word]
        if new_word < self.vocab_size:
            H_sum += self.L2[:, new_word]
        return H_sum

    def get_sparse_energy(self, word: int, neighbor_words: List[int]) -> int:
        """
        Compute Q contribution to energy: sum_j Q[word, neighbor_words[j]].
        Uses sparse row lookup — O(nnz_per_row).
        """
        energy = 0
        row_entries = self.Q_rows.get(word, [])
        # Build set for fast lookup
        neighbor_set = set(neighbor_words)
        for col, val in row_entries:
            if col in neighbor_set:
                energy += val
        return energy

    def get_factorized_energy(self, word: int, H_sum: np.ndarray) -> int:
        """Compute L1[word,:] @ H_sum — O(K) dot product."""
        return int(self.L1[word] @ H_sum)

    def memory_savings(self) -> Dict:
        """Compute memory savings."""
        full_matrix = self.vocab_size ** 2
        q_nnz = int(self.Q.nnz) if self.Q is not None else 0
        factorized = q_nnz + self.vocab_size * self.L1.shape[1] * 2
        savings_pct = (1 - factorized / full_matrix) * 100 if full_matrix > 0 else 0
        return {
            "full_matrix_elements": full_matrix,
            "factorized_elements": factorized,
            "savings_pct": savings_pct,
            "q_nnz": q_nnz,
            "n_factors_total": self.L1.shape[1],
            "vocab_size": self.vocab_size,
        }

    def save(self, path: str):
        """Save CALDERA factorization."""
        if self.Q is not None:
            sparse.save_npz(f"{path}_Q.npz", self.Q)
        np.save(f"{path}_L1.npy", self.L1)
        np.save(f"{path}_L2.npy", self.L2)

        meta = {
            "vocab_size": self.vocab_size,
            "n_factors": self.n_factors,
            "n_top": self.n_top,
            "n_factors_total": int(self.L1.shape[1]),
            "fitted": self.fitted,
        }
        with open(f"{path}_meta.json", "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> "CalderaNMF":
        """Load CALDERA factorization."""
        with open(f"{path}_meta.json") as f:
            meta = json.load(f)

        nmf = cls(
            vocab_size=meta["vocab_size"],
            n_factors=meta["n_factors"],
            n_top=meta["n_top"],
        )
        
        try:
            nmf.Q = sparse.load_npz(f"{path}_Q.npz")
        except Exception:
            nmf.Q = None
        
        nmf.L1 = np.load(f"{path}_L1.npy")
        nmf.L2 = np.load(f"{path}_L2.npy")
        nmf.fitted = meta["fitted"]
        
        # Rebuild fast row lookup
        if nmf.Q is not None:
            for i in range(nmf.vocab_size):
                row_start = nmf.Q.indptr[i]
                row_end = nmf.Q.indptr[i + 1]
                if row_end > row_start:
                    cols = nmf.Q.indices[row_start:row_end]
                    vals = nmf.Q.data[row_start:row_end]
                    nmf.Q_rows[i] = [(int(c), int(v)) for c, v in zip(cols, vals)]
                else:
                    nmf.Q_rows[i] = []

        return nmf
