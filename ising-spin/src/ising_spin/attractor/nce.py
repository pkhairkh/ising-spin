"""
Noise Contrastive Estimation (NCE) for DAM Discriminator Training.

v76e: Fix word_swap + re-ranking calibration.

Bug 12: word_swap accuracy = 0.371 (below chance).
  Only 1 word_swap negative per positive, drowned out by 3 other types.
  Word_swap is the HARDEST corruption (smallest context change), so it
  needs MORE training signal, not less.
  FIX: Generate n_word_swap extras (default 3) per positive, and apply
  per-type weighting (word_swap gets 2x weight in the NCE update).

Bug 13: Re-ranking calibration gives rerank_acc = 0.364.
  Calibration searched beta in [0.005..0.1] but used greedy argmin
  (which IGNORES beta entirely). Also didn't search dam_weight_shift.
  FIX: Calibration now searches dam_weight_shift AND uses proper
  Boltzmann-weighted selection. Also searches wider beta range.

v76d: Balanced NCE ratio + asymmetric J (fixed energy inversion).
"""

import numpy as np
import time
from typing import List, Tuple, Optional, Dict

from .dam import DAMLayer
from .sdr import SDREncoder
from .corruptions import Corruptor, CORRUPTION_NAMES


class NCETrainer:
    """
    NCE Hebbian trainer for the DAM discriminator.

    v76e: Extra word_swap negatives + per-type weighting.
    v76d: Balanced NCE ratio for both J and h, asymmetric J,
    multi-type disc_acc check.
    """

    def __init__(
        self,
        dam: DAMLayer,
        sdr_encoder: SDREncoder,
        corruptor: Corruptor,
        context_window: int = 10,
        batch_size: int = 200,
        eta: int = 10,
        j_clip: int = 32000,
        uv_regularize: bool = True,
        uv_lambda: int = 10,
        n_word_swap: int = 3,
        word_swap_weight: int = 2,
    ):
        self.dam = dam
        self.sdr_encoder = sdr_encoder
        self.corruptor = corruptor
        self.context_window = context_window
        self.batch_size = min(batch_size, 200)
        self.eta = eta
        self.j_clip = j_clip
        self.uv_regularize = uv_regularize
        self.uv_lambda = uv_lambda
        self.n_word_swap = n_word_swap      # Extra word_swap negatives per positive
        self.word_swap_weight = word_swap_weight  # Weight multiplier for word_swap in NCE update

    def train_epoch(
        self,
        sequences: List[List[int]],
        epoch: int = 0,
        n_negatives: int = 4,
        callback=None,
    ) -> Dict:
        """One epoch of NCE training, streaming pairs from sequences.

        v76e: Total negatives per positive = n_negatives (base types)
        + n_word_swap (extra word_swap). Default: 4 + 3 = 7 total.
        """
        # Count total pairs
        n_pairs = sum(max(0, len(s) - 1) for s in sequences if len(s) >= 2)
        n_batches = max(1, (n_pairs + self.batch_size - 1) // self.batch_size)

        disc_correct = 0
        disc_total = 0
        total_pos_energy = 0
        total_neg_energy = 0

        t_start = time.time()
        batch_pairs = []
        pairs_processed = 0
        batch_count = 0

        for seq in sequences:
            if len(seq) < 2:
                continue
            for pos in range(1, len(seq)):
                batch_pairs.append((seq[:pos], seq[pos]))

                if len(batch_pairs) >= self.batch_size:
                    result = self._process_batch(
                        batch_pairs, n_negatives,
                    )
                    pairs_processed += len(batch_pairs)
                    batch_count += 1
                    disc_correct += result['disc_correct']
                    disc_total += result['disc_total']
                    total_pos_energy += result['total_pos_energy']
                    total_neg_energy += result['total_neg_energy']
                    batch_pairs = []

                    # Progress logging
                    if batch_count % 10 == 0:
                        elapsed = time.time() - t_start
                        rate = pairs_processed / max(1, elapsed)
                        eta_s = (n_pairs - pairs_processed) / max(1, rate)
                        print(f"      Batch {batch_count}/{n_batches}: "
                              f"{pairs_processed}/{n_pairs} pairs, "
                              f"{rate:.0f} pairs/s, "
                              f"ETA {eta_s:.0f}s", flush=True)

        # Process remaining
        if batch_pairs:
            result = self._process_batch(batch_pairs, n_negatives)
            pairs_processed += len(batch_pairs)
            batch_count += 1
            disc_correct += result['disc_correct']
            disc_total += result['disc_total']
            total_pos_energy += result['total_pos_energy']
            total_neg_energy += result['total_neg_energy']

        t_elapsed = time.time() - t_start
        disc_acc = disc_correct / max(1, disc_total)
        avg_pos_e = total_pos_energy / max(1, disc_total)
        avg_neg_e = total_neg_energy / max(1, disc_total)

        # Apply UV regularization ONCE at end of epoch (not per-batch!)
        if self.uv_regularize:
            nnz_before = int(np.count_nonzero(self.dam.J))
            self.dam._uv_regularize()
            nnz_after = int(np.count_nonzero(self.dam.J))
            print(f"    UV regularization: J_nnz {nnz_before} → {nnz_after}")

        # v76d diagnostic: h statistics
        h_nnz = int(np.count_nonzero(self.dam.h))
        h_max = int(np.max(np.abs(self.dam.h))) if h_nnz > 0 else 0
        h_mean = float(np.mean(self.dam.h.astype(np.float32)))
        print(f"    h stats: nnz={h_nnz}, max={h_max}, mean={h_mean:.1f}")

        print(f"    Epoch {epoch+1}: disc_acc={disc_acc:.3f}, "
              f"energy_gap={avg_neg_e - avg_pos_e:.1f}, "
              f"J_nnz={int(np.count_nonzero(self.dam.J))}, "
              f"J_max={int(np.max(np.abs(self.dam.J)))}, "
              f"time={t_elapsed:.1f}s", flush=True)

        return {
            'epoch': epoch,
            'n_pairs': n_pairs,
            'time_s': t_elapsed,
            'disc_accuracy': disc_acc,
            'avg_pos_energy': avg_pos_e,
            'avg_neg_energy': avg_neg_e,
            'energy_gap': avg_neg_e - avg_pos_e,
            'J_nnz': int(np.count_nonzero(self.dam.J)),
            'J_max': int(np.max(np.abs(self.dam.J))),
            'h_nnz': h_nnz,
            'h_max': h_max,
            'h_mean': h_mean,
        }

    def _process_batch(
        self,
        batch_pairs: List[Tuple[List[int], int]],
        n_negatives: int,
    ) -> Dict:
        """Process one batch of (context, target) pairs.

        v76d: Balanced NCE update for both J and h.
        The key insight: with n_neg negatives, the positive signal must
        be weighted n_neg× to balance. Otherwise common features (present
        in both pos and neg) get driven negative, inverting the energy
        landscape so the DAM assigns LOWER energy to corrupted text.
        """
        B = len(batch_pairs)
        D = self.dam.D

        # Encode positive pairs
        pos_ctx_sdrs = np.zeros((B, D), dtype=np.uint8)
        pos_tgt_sdrs = np.zeros((B, D), dtype=np.uint8)

        ctx_cache = {}
        for i, (ctx, nxt) in enumerate(batch_pairs):
            pos_tgt_sdrs[i] = self.sdr_encoder.encode(nxt)
            ctx_key = tuple(ctx[-self.context_window:])
            if ctx_key not in ctx_cache:
                ctx_cache[ctx_key] = self.sdr_encoder.encode_context_positional(
                    list(ctx_key), context_window=self.context_window
                )
            pos_ctx_sdrs[i] = ctx_cache[ctx_key]

        # v76e: Generate and encode negative pairs with extra word_swap
        total_neg = n_negatives + self.n_word_swap
        neg_ctx_sdrs = [np.zeros((B, D), dtype=np.uint8) for _ in range(total_neg)]
        neg_tgt_sdrs = [np.zeros((B, D), dtype=np.uint8) for _ in range(total_neg)]
        neg_type_weights = np.ones(total_neg, dtype=np.float32)  # per-negative weight

        for i, (ctx, nxt) in enumerate(batch_pairs):
            # Standard 4 negatives (one per type)
            negatives = self.corruptor.generate_negatives(ctx, nxt, n_negatives)
            for k, (neg_ctx, neg_cand, ctype) in enumerate(negatives[:n_negatives]):
                if ctype != 3:  # Not WORD_SWAP: reuse positive context
                    neg_ctx_sdrs[k][i] = pos_ctx_sdrs[i]
                else:
                    neg_ctx_key = tuple(neg_ctx[-self.context_window:])
                    if neg_ctx_key not in ctx_cache:
                        ctx_cache[neg_ctx_key] = self.sdr_encoder.encode_context_positional(
                            list(neg_ctx_key), context_window=self.context_window
                        )
                    neg_ctx_sdrs[k][i] = ctx_cache[neg_ctx_key]
                neg_tgt_sdrs[k][i] = self.sdr_encoder.encode(neg_cand)

            # v76e: Extra word_swap negatives (the hardest type needs more signal)
            for extra_k in range(self.n_word_swap):
                idx = n_negatives + extra_k
                neg_type_weights[idx] = float(self.word_swap_weight)
                swap_neg = self.corruptor._corrupt(ctx, nxt, 3)  # WORD_SWAP=3
                if swap_neg is not None:
                    neg_ctx_list, neg_cand, _ = swap_neg
                    neg_ctx_key = tuple(neg_ctx_list[-self.context_window:])
                    if neg_ctx_key not in ctx_cache:
                        ctx_cache[neg_ctx_key] = self.sdr_encoder.encode_context_positional(
                            list(neg_ctx_key), context_window=self.context_window
                        )
                    neg_ctx_sdrs[idx][i] = ctx_cache[neg_ctx_key]
                    neg_tgt_sdrs[idx][i] = self.sdr_encoder.encode(neg_cand)
                else:
                    # Fallback: reuse positive (zero gradient contribution)
                    neg_ctx_sdrs[idx][i] = pos_ctx_sdrs[i]
                    neg_tgt_sdrs[idx][i] = self.sdr_encoder.encode(nxt)

        # === BALANCED NCE UPDATE for J ===
        # v76e: Per-type weighted NCE update.
        # word_swap negatives get word_swap_weight (default 2x) because they
        # provide the weakest signal (smallest context change) and need more
        # gradient to compete with the stronger corruption types.
        #
        # The balanced ratio still applies: total_neg_weight * pos - weighted_neg_sum
        pos_ctx_f = pos_ctx_sdrs.astype(np.float32)
        pos_tgt_f = pos_tgt_sdrs.astype(np.float32)
        J_pos_mean = (pos_tgt_f.T @ pos_ctx_f) / B  # entries in [0, 1]

        J_neg_weighted = np.zeros_like(J_pos_mean)
        total_neg_weight = 0.0
        for k in range(total_neg):
            w = float(neg_type_weights[k])
            J_neg_weighted += w * (neg_tgt_sdrs[k].astype(np.float32).T
                          @ neg_ctx_sdrs[k].astype(np.float32)) / B
            total_neg_weight += w

        # Balanced NCE update: total_weight * pos - weighted_neg
        J_update_f = float(self.eta) * (total_neg_weight * J_pos_mean - J_neg_weighted)
        J_update = np.round(J_update_f).astype(np.int32)
        del pos_ctx_f, pos_tgt_f, J_pos_mean, J_neg_weighted, J_update_f

        # BUG 10 FIX: Do NOT symmetrize J for the discriminative model.
        # The energy is E = -s_tgt^T (J @ s_ctx + h) — a BIPARTITE bilinear form.
        # J[i,j] = coupling FROM context bit j TO target bit i.
        # Symmetrizing mixes this with the reverse signal (ctx bit i → tgt bit j),
        # which is a different signal that dilutes discriminative power.
        # Also: no need to zero diagonal — self-coupling is valid in bipartite energy.
        # (Removed: np.fill_diagonal(J_update, 0) and J_update symmetrization)

        J_new = self.dam.J.astype(np.int32) + J_update
        np.clip(J_new, -self.j_clip, self.j_clip, out=J_new)
        self.dam.J = J_new.astype(np.int16)

        # === BALANCED NCE UPDATE for h ===
        # v76e: Same per-type weighting as J.
        h_pos_mean = np.mean(pos_tgt_sdrs.astype(np.float32), axis=0)  # [0, 1]
        h_neg_weighted = np.zeros(D, dtype=np.float32)
        for k in range(total_neg):
            w = float(neg_type_weights[k])
            h_neg_weighted += w * np.mean(neg_tgt_sdrs[k].astype(np.float32), axis=0)
        h_update = np.round(
            self.eta * (total_neg_weight * h_pos_mean - h_neg_weighted)
        ).astype(np.int32)
        h_new = self.dam.h.astype(np.int32) + h_update
        np.clip(h_new, -self.j_clip, self.j_clip, out=h_new)
        self.dam.h = h_new.astype(np.int16)

        # NOTE: UV regularization is NOT applied per-batch.
        # Applied once at end of epoch in train_epoch().

        # === Discriminative accuracy check — ALL neg types ===
        # v76e: Check all negatives including extra word_swap.
        disc_correct = 0
        disc_total = 0
        total_pos_energy = 0
        total_neg_energy = 0
        n_check = min(20, B)
        check_indices = np.random.choice(B, n_check, replace=False)
        for idx in check_indices:
            ctx_sdr = pos_ctx_sdrs[idx]
            tgt_sdr = pos_tgt_sdrs[idx]
            field = self.dam.compute_field(ctx_sdr)
            active = np.where(tgt_sdr > 0)[0]
            pos_e = -int(np.sum(field[active]))

            # Check ALL negative types (including extra word_swap)
            for k in range(total_neg):
                neg_field = self.dam.compute_field(neg_ctx_sdrs[k][idx])
                neg_active = np.where(neg_tgt_sdrs[k][idx] > 0)[0]
                neg_e = -int(np.sum(neg_field[neg_active]))

                if pos_e < neg_e:  # Lower energy = more likely
                    disc_correct += 1
                disc_total += 1
                total_pos_energy += pos_e
                total_neg_energy += neg_e

        return {
            'disc_correct': disc_correct,
            'disc_total': disc_total,
            'total_pos_energy': total_pos_energy,
            'total_neg_energy': total_neg_energy,
        }
