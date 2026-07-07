from __future__ import annotations

from typing import Any

__all__ = ["ANNIndexer", "DataLoader", "DatasetManager", "MergedDataLoader"]


def __getattr__(name: str) -> Any:
    if name == "ANNIndexer":
        from .ann_indexer import ANNIndexer

        return ANNIndexer
    if name in {"DataLoader", "DatasetManager"}:
        from .data_reader import DataLoader, DatasetManager

        return {"DataLoader": DataLoader, "DatasetManager": DatasetManager}[name]
    if name == "MergedDataLoader":
        from .merged_loader import MergedDataLoader

        return MergedDataLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
