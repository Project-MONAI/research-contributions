"""Cache the data using redis.

TODO(meijieru): zeromp may be better.
"""

from typing import Callable, MutableMapping, Optional, Sequence, Union

import bagua.torch_api.contrib.cache_loader as bagua_cache_loader
import torch
import torch.utils.data.dataset as torch_dataset

import monai.data as monai_data
import monai.transforms as monai_transforms

_ALL_DATASET_NAMES = set()
_SERIALIZATION_HIJACKED = False


def hijack_bagua_serialization(method: str):
    """Replace bagua serialization."""
    global _SERIALIZATION_HIJACKED
    if _SERIALIZATION_HIJACKED:
        raise RuntimeError("Already hijacked.")

    import pickle

    if method == "lz4":
        import lz4

        compress, decompress = lz4.frame.compress, lz4.frame.decompress
    elif method == "lzma":
        import pylzma as lzma

        compress, decompress = lzma.compress, lzma.decompress
    elif method == "zlib":
        import zlib

        compress, decompress = zlib.compress, zlib.decompress
    else:
        raise ValueError(f"Unknown compress method: {method}")

    bagua_cache_loader.serialize = lambda val: compress(pickle.dumps(val))
    bagua_cache_loader.deserialize = lambda val: pickle.loads(decompress(val))
    _SERIALIZATION_HIJACKED = True


def is_deterministic_transform(transform) -> bool:
    return not (
        isinstance(transform, monai_transforms.Randomizable) or not isinstance(transform, monai_transforms.Transform)
    )


class CachedDataset(torch_dataset.Dataset):
    def __init__(
        self,
        data: Sequence,
        transform: Optional[Union[Sequence[Callable], Callable]] = None,
        as_contiguous: bool = True,
        backend: str = "redis",
        hosts: Optional[Sequence[MutableMapping[str, str]]] = None,
        dataset_name: str = "",
        writer_buffer_size: int = 20,
        **kwargs,
    ) -> None:
        super().__init__()

        if hosts is None:
            raise ValueError("We don't init bagua, have to manually launch redis")

        # NOTE(meijieru): check if the dataset name is unique, to avoid
        # potential confliction.
        if not dataset_name or dataset_name in _ALL_DATASET_NAMES:
            raise ValueError("Must have an unique name for each dataset.")
        _ALL_DATASET_NAMES.add(dataset_name)

        self._dataset = monai_data.Dataset(data=data)
        self._cache_loader = bagua_cache_loader.CacheLoader(
            backend, dataset_name, writer_buffer_size, hosts=hosts, **kwargs
        )
        self.transform = transform
        self.as_contiguous = as_contiguous

    def __len__(self):
        return len(self._dataset)

    def _apply_non_deterministic_transform(self, item):
        for trans in self.transform.transforms:  # type:ignore
            if not is_deterministic_transform(trans):
                item = monai_transforms.apply_transform(trans, item)
        return item

    def _apply_deterministic_transform(self, item):
        for trans in self.transform.transforms:  # type:ignore
            # execute all the deterministic transforms
            if not is_deterministic_transform(trans):
                break
            item = monai_transforms.apply_transform(trans, item)
        if self.as_contiguous:
            item = monai_transforms.convert_to_contiguous(item, memory_format=torch.contiguous_format)
        return item

    def _load_item(self, index: int):
        return self._apply_deterministic_transform(self._dataset[index])

    def __getitem__(self, item):
        cached_item = self._cache_loader.get(item, self._load_item)
        return self._apply_non_deterministic_transform(cached_item)
