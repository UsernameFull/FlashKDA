"""Ascend NPU backend for FlashKDA."""

from .ops.kda_fwd import fwd, is_available

__all__ = ["fwd", "is_available"]
