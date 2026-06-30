from functools import reduce
from operator import concat
from typing import Iterator, Sequence

from src.packages import *

class PviBatchSampler:
    """
    Custom sampler class to be plugged into torch DataLoader.
    """
    def __init__(self,
                 global_indices: Sequence[int],
                 grouping: list[list[int]],
                 batch_size: int=32,
                 cluster_size: int=None,
                 drop_last: bool=False) -> None:

        self._alias = type(self).__name__  # temporary alias

        self.global_indices = global_indices
        self.grouping = grouping
        self.batch_size = batch_size
        self.drop_last = drop_last

        if cluster_size is None:
            self.cluster_size = len(grouping)
        elif isinstance(cluster_size, int):
            self.cluster_size = max(cluster_size, 1)
        else:
            raise TypeError(f"Invalid type {type(cluster_size)} for cluster_size")

        if self.drop_last:
            self.num_batches = len(global_indices) // batch_size
        else:
            q, r = divmod(len(global_indices), batch_size)
            self.num_batches = q + bool(r)

        print(f"{self._alias}: Ready.")

        self.batch_subgroups: list[list[int]] = []

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        # When making the subgroups, we refer to the global indices (of the master dataset)
        # go ensure accurate mapping.
        # But if we want to use the subgroups in the Subset class, we need to remap the global
        # indices to local

        batch_subgroups = self._stratified_random(global_indices=self.global_indices,
                                                  grouping=self.grouping,
                                                  batch_size=self.batch_size,
                                                  cluster_size=self.cluster_size,)

        batch_subgroups = self._remap_local_indices(batch_subgroups)

        return iter(batch_subgroups)

    def _remap_local_indices(self, subgroups_global: list[list[int]]) -> list[list[int]]:
        idx_lookup = {idx_global: idx_local
                      for idx_local, idx_global in enumerate(self.global_indices)}

        return [[idx_lookup[ig] for ig in g] for g in subgroups_global]

    def _stratified_random(self,
                           global_indices: Sequence[int],
                           grouping: list[list[int]],
                           batch_size: int,
                           cluster_size: int,
                           ) -> list[list[int]]:
        # group by files (or subjects)
        file_subgroups = []
        for group in grouping:
            file_subgroups.append(list(set(global_indices) & set(group)))

        # level-1 shuffle
        random.shuffle(file_subgroups)
        assert len(file_subgroups) > 0

        flatten = []
        for i in range(0, len(file_subgroups), cluster_size):
            xss = file_subgroups[slice(i, i + cluster_size)] # list of lists

            assert len(xss) > 0

            xss = reduce(concat, xss)

            random.shuffle(xss) # level-2 shuffle
            flatten.extend(xss)

        # grouped by batches
        batch_subgroups = [flatten[slice(i,i + batch_size)] for i in range(0, len(flatten), batch_size)]

        assert sum([len(g) for g in batch_subgroups]) == len(global_indices)

        if not batch_subgroups:
            return []
        else:
            if self.drop_last and len(batch_subgroups[-1]) < batch_size:
                batch_subgroups = batch_subgroups[:-1]  # Remove last incomplete batch

        self.batch_subgroups = batch_subgroups

        return batch_subgroups