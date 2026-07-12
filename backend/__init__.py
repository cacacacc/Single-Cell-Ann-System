"""Lazy public exports for backend modules.

The package root exposes the most commonly used classes without importing heavy
scientific dependencies during ``import backend``.  Each object is imported only
when first accessed, which keeps Flask startup and tests lighter when optional
packages such as FAISS, ChromaDB or AnnData are not needed.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ANNIndexer", "DataLoader", "DatasetManager", "MergedDataLoader"]


def __getattr__(name: str) -> Any:
    """Resolve supported public symbols on demand."""
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
