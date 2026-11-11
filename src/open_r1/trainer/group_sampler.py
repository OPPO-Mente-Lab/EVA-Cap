
from typing import Dict, Iterator, List, Optional

import torch
from torch.utils.data import Sampler
import math


class DistributedGroupRepeatSampler(Sampler):
    """
    Distributed sampler that ensures:
      - In each global iteration, samples come from the same group/type.
      - Each unique sample in a global iteration is repeated `mini_repeat` times (across devices).
      - Works with DDP: each rank receives its local slice of each global_chunk.
      - Supports deterministic per-epoch shuffling via set_epoch.

    Args:
        data_source_dict: Dict[group_id -> list of dataset indices]
        batch_size: per-device unique samples (b)
        mini_repeat: how many times each unique sample is repeated in a global iteration (R)
        repeat_count: how many times to repeat each global_chunk (like your previous repeat_count)
        shuffle: whether to shuffle
        seed: base seed
        distributed: whether running distributed
        num_replicas: total number of ranks (world_size)
        rank: current rank id
        drop_last: if True, drop leftover unique-chunks that don't fill a full unique_chunk_size
    """

    def __init__(
        self,
        data_source_dict: Dict[int, List[int]],
        batch_size: int,
        mini_repeat: int,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        distributed: bool = False,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        drop_last: bool = True,
    ):
        super().__init__(data_source_dict)

        self.groups = {g: list(idxs) for g, idxs in data_source_dict.items()}
        self.batch_size = int(batch_size)
        self.mini_repeat = int(mini_repeat)
        self.repeat_count = int(repeat_count)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.drop_last = bool(drop_last)

        # distributed info
        if distributed:
            if num_replicas is None or rank is None:
                raise ValueError("num_replicas and rank must be provided when distributed=True")
            self.world_size = int(num_replicas)
            self.rank = int(rank)
        else:
            self.world_size = 1
            self.rank = 0

        # sanity check: (batch_size * world_size) must be divisible by mini_repeat
        if (self.batch_size * self.world_size) % self.mini_repeat != 0:
            raise ValueError(
                "batch_size * world_size must be divisible by mini_repeat. "
                f"Got batch_size={self.batch_size}, world_size={self.world_size}, mini_repeat={self.mini_repeat}"
            )

        # number of unique samples in one global chunk
        self.unique_chunk_size = (self.batch_size * self.world_size) // self.mini_repeat

        # pad each group so its length is divisible by unique_chunk_size (or drop_last later)
        self.group_padded = {}
        for g, items in self.groups.items():
            items = list(items)
            if len(items) == 0:
                self.group_padded[g] = []
                continue

            if self.drop_last:
                # truncate to multiple of unique_chunk_size
                n_chunks = len(items) // self.unique_chunk_size
                items = items[: n_chunks * self.unique_chunk_size]
            else:
                # pad by repeating from head to make divisible
                n_chunks = math.ceil(len(items) / self.unique_chunk_size)
                need = n_chunks * self.unique_chunk_size
                if need > len(items):
                    extra = need - len(items)
                    # simple padding by reusing from start
                    items = items + items[:extra]
            self.group_padded[g] = items

        # compute number of global_chunks
        self.global_chunks_per_group = {
            g: (len(items) // self.unique_chunk_size) if self.unique_chunk_size > 0 else 0
            for g, items in self.group_padded.items()
        }
        self.total_global_chunks = sum(self.global_chunks_per_group.values())

        # num_samples per rank (each global_chunk contributes batch_size samples per rank,
        # and repeat_count multiplies that)
        self.num_samples = self.total_global_chunks * self.batch_size * self.repeat_count

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # 1) shuffle each group's padded indices (same shuffle across ranks because same seed+epoch)
        groups_shuffled: Dict[int, List[int]] = {}
        for gid, items in self.group_padded.items():
            if len(items) == 0:
                groups_shuffled[gid] = []
                continue
            if self.shuffle:
                perm = torch.randperm(len(items), generator=g).tolist()
                groups_shuffled[gid] = [items[i] for i in perm]
            else:
                groups_shuffled[gid] = list(items)

        # 2) split each group's shuffled list into unique-chunks (each of size unique_chunk_size)
        global_chunks: List[tuple] = []  # list of (gid, unique_chunk_list)
        for gid, items in groups_shuffled.items():
            if self.unique_chunk_size == 0:
                continue
            for i in range(0, len(items), self.unique_chunk_size):
                unique_chunk = items[i : i + self.unique_chunk_size]
                if len(unique_chunk) < self.unique_chunk_size:
                    # this shouldn't happen if we padded or dropped correctly
                    if self.drop_last:
                        continue
                    else:
                        # pad from start of unique_chunk (rare)
                        needed = self.unique_chunk_size - len(unique_chunk)
                        unique_chunk = unique_chunk + unique_chunk[:needed]
                global_chunks.append((gid, unique_chunk))

        # 3) shuffle order of global_chunks (mix types across steps)
        if self.shuffle:
            perm = torch.randperm(len(global_chunks), generator=g).tolist()
            global_chunks = [global_chunks[i] for i in perm]

        # 4) expand each unique_chunk into a global_chunk (each unique repeated mini_repeat times),
        #    then slice out this rank's local part and append to output
        out: List[int] = []
        for gid, unique_chunk in global_chunks:
            # form expanded global chunk (length = batch_size * world_size)
            expanded: List[int] = []
            for uid in unique_chunk:
                expanded.extend([uid] * self.mini_repeat)

            # sanity check
            assert len(expanded) == self.batch_size * self.world_size

            # repeat this global_chunk repeat_count times
            for _ in range(self.repeat_count):
                # take this rank's share: every world_size-th element starting at index == rank
                # that gives exactly batch_size elements
                part = expanded[self.rank :: self.world_size]
                # extend output with part (these are the indices this rank will see for this chunk)
                out.extend(part)

        # # 4) 核心修正：适配单卡多生成 (num_return_sequences > 1)
        # out: List[int] = []
        
        # # 你的配置里：G = num_generations, R_local = num_generations_per_device
        # # 每一组 Prompt 需要的 Rank 数量 = G / R_local
        # # 例如 G=8, R_local=2, 则每 4 个 Rank 负责同一个 Prompt
        
        # for gid, unique_chunk in global_chunks:
        #     # 计算当前 Rank 应该处理这组 unique_chunk 中的哪一个
        #     # 这里的 batch_size 对应模型一次 forward 接收的 prompt 数量
        #     # 映射关系：Rank -> Prompt_Index
        #     idx_in_chunk = (self.rank // self.mini_repeat) * self.batch_size
            
        #     # 选出属于当前 Rank 的 prompts
        #     part = unique_chunk[idx_in_chunk : idx_in_chunk + self.batch_size]
            
        #     for _ in range(self.repeat_count):
        #         out.extend(part)

        # ensure output length equals num_samples (truncate if longer; pad if shorter)
        if len(out) > self.num_samples:
            out = out[: self.num_samples]
        elif len(out) < self.num_samples:
            # pad by repeating from beginning (rare if padding logic above correct)
            need = self.num_samples - len(out)
            if len(out) == 0:
                raise RuntimeError("Sampler produced empty output but expected non-zero.")
            out.extend(out[:need])

        return iter(out)

    def __len__(self) -> int:
        return self.num_samples