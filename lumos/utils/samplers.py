import math
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler, Sampler


class EquiSampler(Sampler):
    """Equidistant batch sampler.

    Yields n (where n==batch_size) equidistant indices, steps through the dataset by adding the sequence length to each index and yielding the new set of indices.
    """

    def __init__(self, data_size, seq_len, batch_size, init_idx=None):
        self.data_size = data_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.init_idx = init_idx
        self.chunk_size = math.ceil(self.data_size / self.batch_size)
        self.n_steps = math.ceil(self.chunk_size / self.seq_len)
        print("Chunk size:", self.chunk_size)
        print("n steps:", self.n_steps)

    def __iter__(self):
        if self.init_idx is None:
            init_idx = np.random.randint(self.data_size)
        else:
            init_idx = self.init_idx
        for i in range(self.n_steps):
            iters = []
            for j in range(self.batch_size):
                start_idx = (init_idx + i * self.seq_len + j * self.chunk_size) % self.data_size
                iters.append(start_idx)
            yield iters

    def __len__(self):
        return self.n_steps


__all__ = ["EquiDistributedSampler"]


class EquiDistributedSampler(BatchSampler):
    """Equidistant *batch* sampler that is friendly to PyTorch Lightning + DDP.

    Differences from the original single‑GPU sampler:
    -------------------------------------------------
    * Inherits from **BatchSampler**, so Lightning will not try to replace it.
    * Splits a *global* logical batch across *num_replicas* GPUs.
    * Uses ``torch.distributed.broadcast_object_list`` to synchronise the
      random start index – works with any backend (GLOO, NCCL, MPI).
    * Exposes the attributes Lightning expects (``batch_size``, ``drop_last``,
      ``sampler``).
    * Provides ``set_epoch()`` like ``DistributedSampler`` so Lightning can
      reshuffle every epoch.
    """

    # ------------------------------------------------------------------
    # Construction ------------------------------------------------------
    # ------------------------------------------------------------------
    def __init__(
        self,
        data_size: int,
        seq_len: int,
        global_batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        init_idx: Optional[int] = None,
        drop_last: bool = False,
    ) -> None:
        # ---------------------------------------------------------------
        # Replica info ---------------------------------------------------
        # ---------------------------------------------------------------
        if num_replicas is None or rank is None:
            if torch.distributed.is_initialized():
                if num_replicas is None:
                    num_replicas = torch.distributed.get_world_size()
                if rank is None:
                    rank = torch.distributed.get_rank()
            else:
                num_replicas = num_replicas or 1
                rank = rank or 0

        if global_batch_size % num_replicas != 0:
            if drop_last:
                local_batch_size = global_batch_size // num_replicas
            else:
                raise ValueError("global_batch_size must be divisible by num_replicas unless drop_last=True")
        else:
            local_batch_size = global_batch_size // num_replicas

        self.data_size: int = data_size
        self.seq_len: int = seq_len
        self.global_batch_size: int = global_batch_size
        self.local_batch_size: int = local_batch_size
        self.num_replicas: int = num_replicas
        self.rank: int = rank
        self.init_idx: Optional[int] = init_idx
        self.drop_last: bool = drop_last

        # Equally‑spaced indices along the dataset
        self.chunk_size = math.ceil(self.data_size / self.global_batch_size)
        self.n_steps = math.ceil(self.chunk_size / self.seq_len)

        # Attributes required by BatchSampler so Lightning is happy
        self.batch_size: int = self.local_batch_size  # per‑GPU size
        self.sampler = None  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Utilities ---------------------------------------------------------
    # ------------------------------------------------------------------
    def _sync_start_idx(self, start_pos: int) -> int:
        """Synchronise *start_pos* across all ranks.

        Uses ``broadcast_object_list`` so we don't care whether the backend is
        NCCL (GPU‑only) or GLOO (CPU‑capable). Works for any small Python
        object, not just tensors.
        """
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            obj_list = [start_pos]
            torch.distributed.broadcast_object_list(obj_list, src=0)
            start_pos = int(obj_list[0])
        return start_pos

    def set_epoch(self, epoch: int):
        rng = np.random.RandomState(seed=epoch)
        self.init_idx = int(rng.randint(self.data_size))

    # ------------------------------------------------------------------
    # Iterator ----------------------------------------------------------
    # ------------------------------------------------------------------
    def __iter__(self):
        # ------------------------------------------------------------------
        # Choose a starting index -----------------------------------------
        # ------------------------------------------------------------------
        if self.init_idx is None:
            start_pos = np.random.randint(self.data_size, dtype=np.int64) if self.rank == 0 else 0
            start_pos = self._sync_start_idx(int(start_pos))
        else:
            start_pos = self._sync_start_idx(int(self.init_idx))

        # ------------------------------------------------------------------
        # Yield batches ----------------------------------------------------
        # ------------------------------------------------------------------
        for step in range(self.n_steps):
            batch = []
            global_offset = step * self.seq_len
            for j in range(self.local_batch_size):
                global_j = self.rank * self.local_batch_size + j
                idx = (start_pos + global_offset + global_j * self.chunk_size) % self.data_size
                batch.append(int(idx))

            if len(batch) == self.local_batch_size or not self.drop_last:
                yield batch

    # ------------------------------------------------------------------
    # Length ------------------------------------------------------------
    # ------------------------------------------------------------------
    def __len__(self):
        return self.n_steps
