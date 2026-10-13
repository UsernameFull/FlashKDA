import math

import pytest
import torch
from torch.nn import functional as F

from flash_kda_npu import fwd as npu_fwd
from flash_kda_npu.cpp_extension_asc import is_loaded
from flash_kda_npu.ops.kda_fwd import is_available


requires_npu = pytest.mark.skipif(
    not is_available() or not is_loaded(),
    reason="requires an NPU and the AOT FlashKDA extension",
)


@requires_npu
@pytest.mark.parametrize(
    "length,heads",
    [(2048, 64), (8192, 96)],
    ids=["T2048-H64", "T8192-H96"],
)
def test_npu_precision_against_fallback(length, heads):
    dim = 128
    torch.manual_seed(42)
    q = F.normalize(
        torch.randn((1, length, heads, dim), dtype=torch.float32, device="npu"),
        p=2, dim=-1,
    ).to(torch.bfloat16)
    k = F.normalize(
        torch.randn((1, length, heads, dim), dtype=torch.float32, device="npu"),
        p=2, dim=-1,
    ).to(torch.bfloat16)
    v = torch.randn((1, length, heads, dim), dtype=torch.bfloat16, device="npu")
    g = torch.randn((1, length, heads, dim), dtype=torch.bfloat16, device="npu")
    beta = torch.randn((1, length, heads), dtype=torch.bfloat16, device="npu")
    A_log = torch.rand(heads, dtype=torch.float32, device="npu")
    dt_bias = torch.rand(heads, dim, dtype=torch.float32, device="npu")
    out_npu = torch.zeros_like(v)
    out_fallback = torch.zeros_like(v)

    kwargs = {
        "A_log": A_log,
        "dt_bias": dt_bias,
        "lower_bound": -5.0,
        "initial_state": None,
        "final_state": None,
        "cu_seqlens": None,
    }
    npu_fwd(q, k, v, g, beta, 1.0 / math.sqrt(dim), out_npu, **kwargs)
    from flash_kda_npu.ops.kda_fwd import _torch_fallback

    _torch_fallback(
        q, k, v, g, beta, 1.0 / math.sqrt(dim), out_fallback, **kwargs
    )

    torch.testing.assert_close(out_npu, out_fallback, rtol=5e-3, atol=5e-3)
