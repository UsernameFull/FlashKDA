"""AscendC (RTC-compiled) KDA v1 backend.

The public entry point serves one K2 implementation: ``persistent_loop``, the
chunk-generic device-side loop.
"""
from typing import Any

_API_NAMES = {"CHUNK", "PERSISTENT_LOOP", "get_last_profile", "kda_bt16_fwd_ascendc"}
_LAYOUT_NAMES = {"pack_tokens", "unpack_tokens"}


def __getattr__(name: str) -> Any:
    if name in _API_NAMES:
        from . import api

        return getattr(api, name)
    if name in _LAYOUT_NAMES:
        from . import layout

        return getattr(layout, name)
    raise AttributeError(name)

__all__ = [
    "CHUNK",
    "PERSISTENT_LOOP",
    "get_last_profile",
    "kda_bt16_fwd_ascendc",
    "pack_tokens",
    "unpack_tokens",
]
__version__ = "0.1.0"
