from typing import Any, Iterator, Protocol, Sequence

import jax
import jax.numpy as jnp


class Dataset(Protocol):
    """Interface for datasets."""

    def __len__(self) -> int:
        """Number of records in the dataset."""

    def __getitem__(self, index: int) -> Any:
        """Retrieves record for the given index."""


class ArrayDataset:
    """Dataset wrapping arrays.

    Each sample will be retrieved by indexing arrays along the first dimension.

    This is a copy from TensorDataset from PyTorch.
    """

    def __init__(self, *arrays: jax.Array) -> None:
        assert all(arrays[0].shape[0] == arr.shape[0] for arr in arrays), (
            "Size mismatch between tensors"
        )
        self.arrays = arrays

    def __getitem__(self, index: int | Sequence[int]) -> tuple[jax.Array]:
        return tuple(arr[index] for arr in self.arrays)

    def __len__(self) -> int:
        return self.arrays[0].shape[0]


class MiniBatchIterator:
    """Homemade data loader."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int = 32,
        shuffle: bool = True,
        key: jax.Array | None = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.key = key

        n_samples = len(dataset)
        self._num_batches = n_samples // batch_size
        if n_samples % batch_size != 0:
            # Add 1 when `batch_size` does not evenly divide the dataset.
            self._num_batches += 1

    @property
    def num_batches(self) -> int:
        return self._num_batches

    def new_split(self, key: jax.Array) -> list[jax.Array]:
        perm_index = jax.random.permutation(key, len(self.dataset))
        return jnp.array_split(perm_index, self.num_batches)

    def __iter__(self) -> Iterator:
        # Split and persist new key.
        self.key, subkey = jax.random.split(self.key)
        for index in self.new_split(subkey):
            yield self.dataset[index]


if __name__ == "__main__":
    from vnn.xy import xy_factory

    seed = 42
    x, y = xy_factory("polynomial")()

    key = jax.random.key(seed)
    dataset = ArrayDataset(x, y)
    iterator = MiniBatchIterator(dataset, shuffle=True, key=key)

    print(next(iter(iterator)))
