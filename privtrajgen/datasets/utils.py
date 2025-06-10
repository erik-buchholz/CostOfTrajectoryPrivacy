#!/usr/bin/env python3
"""Helper classes and methods for datasets."""
from typing import List, Callable, Iterable, Optional, Set


class DictProperty:
    """
    Simulates a dict but might trigger a generation function if values has not been set yet.
    """

    def __init__(self, parent_dict: dict, keys: List[str], creation_fn: Callable[[str], None]):
        self.parent = parent_dict
        self._keys = set(keys)
        self.creation_fn = creation_fn

    def __getitem__(self, key: str) -> object:
        if key in self._keys:
            if key not in self.parent:
                self.creation_fn(key)
            return self.parent[key]
        else:
            raise KeyError(f"Key {key} not in keys.")

    def __setitem__(self, key: str, value: object) -> None:
        raise NotImplementedError("Setting values is not supported.")

    def __delitem__(self, key: str,) -> None:
        raise NotImplementedError("Deleting values is not supported.")

    def __iter__(self) -> Iterable[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: str) -> bool:
        return key in self._keys

    def __repr__(self) -> str:
        return f"DictProperty({self.parent}, {self._keys})"

    def __str__(self) -> str:
        return f"DictProperty({self.parent}, {self._keys})"

    def keys(self) -> Set[str]:
        return self._keys

    def values(self) -> List[object]:
        return [self.__getitem__(key) for key in self._keys]

    def get(self, key:str , default: Optional[object] = None) -> object:
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def pop(self, key: str, default: Optional[object] = None) -> object:
        raise NotImplementedError("Popping values is not supported.")
