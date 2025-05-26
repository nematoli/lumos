import math

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


class EquiBatchSampler(BatchSampler):
    def __init__(self, data_size, seq_len, batch_size, init_idx=None, drop_last=True):
        self.data_size = data_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.init_idx = init_idx
        self.drop_last = drop_last

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        self.chunk_size = math.ceil(self.data_size / self.batch_size)
        self.n_steps = math.ceil(self.chunk_size / self.seq_len)

        # Even split across ranks (drop remainder)
        self.steps_per_rank = self.n_steps // self.world_size
        self.start_step = self.rank * self.steps_per_rank
        self.end_step = (self.rank + 1) * self.steps_per_rank

        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _shared_init_idx(self):
        if self.init_idx is not None:
            return self.init_idx  # user-supplied seed
        if not dist.is_initialized():
            return np.random.randint(self.data_size)  # single-GPU fallback

        # pick on rank-0, broadcast
        idx = torch.randint(self.data_size, (1,), device="cuda")
        dist.broadcast(idx, 0)
        return idx.item()

    def __iter__(self):
        init_idx = self._shared_init_idx()
        init_idx = (init_idx + self.epoch * self.seq_len) % self.data_size

        for i in range(self.start_step, self.end_step):
            yield [(init_idx + i * self.seq_len + j * self.chunk_size) % self.data_size for j in range(self.batch_size)]

    def __len__(self):
        return self.steps_per_rank
