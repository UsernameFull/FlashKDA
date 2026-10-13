from __future__ import annotations

from pathlib import Path

import torch


_PACKAGE_ROOT = Path(__file__).resolve().parent


def _library_candidates() -> list[Path]:
    patterns = ("*flash_kda_npu_kernels*.so", "*flash_kda_npu_kernels*.dll")
    libraries: list[Path] = []
    for root in (_PACKAGE_ROOT, _PACKAGE_ROOT / "lib", _PACKAGE_ROOT.parent / "lib"):
        for pattern in patterns:
            libraries.extend(root.glob(pattern))
    return sorted(set(libraries))


def is_loaded() -> bool:
    try:
        torch.ops.flash_kda_npu.fwd
        return True
    except (AttributeError, RuntimeError):
        return False


def load_extension() -> None:
    """Load the AOT extension produced by ``flash_kda_npu/csrc``."""

    if is_loaded():
        return

    candidates = _library_candidates()
    if not candidates:
        raise FileNotFoundError(
            "flash_kda_npu AOT extension was not found. Build with "
            "ASCEND_HOME_PATH set, or install a wheel that includes "
            "libflash_kda_npu_kernels."
        )

    # torch.ops.load_library accepts POSIX-style paths on all platforms.
    torch.ops.load_library(str(candidates[0].resolve()).replace("\\", "/"))


__all__ = ["is_loaded", "load_extension"]
