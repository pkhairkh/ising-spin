"""
Noise Contrastive Estimation (NCE) for DAM Discriminator Training.

v76g: Remove word_swap from training.

  WHY: The DAM with SDR encoding CANNOT detect word swaps. The positional
  VSA encoding (np.roll by position) creates such small changes in the
  context SDR after a swap that the DAM energy barely moves. After 3
  versions (v76d/e/f) trying to fix this (balanced updates, extra
  negatives, per-type weighting), word_swap accuracy remains stuck at
  0.411 — below random chance (0.5).

  Continuing to train on word_swap WASTES J-matrix capacity on an
  impossible task, which degrades performance on the tasks the DAM
  CAN learn (random_sub, pos_violate, topic_violate). Word order is
  now handled by an explicit bigram energy table in the reranker.

  CHANGES from v76e:
  - Only 3 corruption types: RANDOM_SUB, POS_VIOLATE, TOPIC_VIOLATE
  - Removed n_word_swap and word_swap_weight parameters
  - Simpler code, faster training, no wasted capacity

v76d: Balanced NCE ratio + asymmetric J (fixed energy inversion).
"""

import numpy as np
import time
from typing import List, Tuple, Optional, Dict

from .dam import DAMLayer
from .sdr import SDREncoder
from .corruptions import Corruptor, CORRUPTION_NAMES, RANDOM_SUB, POS_VIOLATE, TOPIC_VIOLATE


class NCETrainer:
    """
    NCE Hebbian trainer for the DAM discriminator.

    v76g: 3 corruption types only (no word_swap).
    v76d: Balanced NCE ratio for both J and h, asymmetric J.
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

    def train_epoch(
        self,
        sequences: List[List[int]],
        epoch: int = 0,
        n_negatives: int = 3,
        callback=None,
    ) -> Dict:
        """
        One epoch of NCE training, streaming pairs from sequences.

        v76g: n_negatives defaults to 3 (one per corruption type).
        No word_swap negatives.
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

        # Apply UV regularization ONCE at end of epoch
        if self.uv_regularize:
            nnz_before = int(np.count_nonzero(self.dam.J))
            self.dam._uv_regularize()
            nnz_after = int(np.count_nonzero(self.dam.J))
            print(f"    UV regularization: J_nnz {nnz_before} → {nnz_after}")

        # h statistics
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
        """
        Process one batch of (context, target) pairs.

        v76g: Balanced NCE with 3 corruption types only.
        No word_swap — it was below chance and wasted capacity.
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

        # Generate and encode negative pairs
        # v76g: Only 3 types — RANDOM_SUB, POS_VIOLATE, TOPIC_VIOLATE
        neg_ctx_sdrs = [np.zeros((B, D), dtype=np.uint8) for _ in range(n_negatives)]
        neg_tgt_sdrs = [np.zeros((B, D), dtype=np.uint8) for _ in range(n_negatives)]

        for i, (ctx, nxt) in enumerate(batch_pairs):
            # Generate negatives with only 3 types (no WORD_SWAP)
            negatives = self.corruptor.generate_negatives(
                ctx, nxt, n_negatives=n_negatives
            )
            for k, (neg_ctx, neg_cand, ctype) in enumerate(negatives[:n_negatives]):
                neg_tgt_sdrs[k][i] = self.sdr_encoder.encode(neg_cand)
                # All 3 corruption types keep the same context
                neg_ctx_sdrs[k][i] = pos_ctx_sdrs[i]

        # === BALANCED NCE UPDATE for J ===
        pos_ctx_f = pos_ctx_sdrs.astype(np.float32)
        pos_tgt_f = pos_tgt_sdrs.astype(np.float32)
        J_pos_mean = (pos_tgt_f.T @ pos_ctx_f) / B

        J_neg_mean = np.zeros_like(J_pos_mean)
        for k in range(n_negatives):
            J_neg_mean += (neg_tgt_sdrs[k].astype(np.float32).T
                          @ neg_ctx_sdrs[k].astype(np.float32)) / B

        # Balanced NCE: n_neg * pos - neg_sum
        J_update_f = float(self.eta) * (float(n_negatives) * J_pos_mean - J_neg_mean)
        J_update = np.round(J_update_f).astype(np.int32)
        del pos_ctx_f, pos_tgt_f, J_pos_mean, J_neg_mean, J_update_f

        # Asymmetric J (bipartite: no symmetrization)
        J_new = self.dam.J.astype(np.int32) + J_update
        np.clip(J_new, -self.j_clip, self.j_clip, out=J_new)
        self.dam.J = J_new.astype(np.int16)

        # === BALANCED NCE UPDATE for h ===
        h_pos_mean = np.mean(pos_tgt_sdrs.astype(np.float32), axis=0)
        h_neg_mean = np.zeros(D, dtype=np.float32)
        for k in range(n_negatives):
            h_neg_mean += np.mean(neg_tgt_sdrs[k].astype(np.float32), axis=0)
        h_update = np.round(
            self.eta * (float(n_negatives) * h_pos_mean - h_neg_mean)
        ).astype(np.int32)
        h_new = self.dam.h.astype(np.int32) + h_update
        np.clip(h_new, -self.j_clip, self.j_clip, out=h_new)
        self.dam.h = h_new.astype(np.int16)

        # === Discriminative accuracy check ===
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

            for k in range(n_negatives):
                neg_field = self.dam.compute_field(neg_ctx_sdrs[k][idx])
                neg_active = np.where(neg_tgt_sdrs[k][idx] > 0)[0]
                neg_e = -int(np.sum(neg_field[neg_active]))

                if pos_e < neg_e:
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
