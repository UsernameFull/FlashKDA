"""Loader for the AOT-compiled FlashKDA Ascend extension."""

from ._load import is_loaded, load_extension

__all__ = ["is_loaded", "load_extension"]
