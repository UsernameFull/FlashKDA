import math

import pytest
import torch
from torch.nn import functional as F

from flash_kda_npu import fwd as npu_fwd
from flash_kda_npu.ops.kda_fwd import is_available
from tests.flashkda_reference import flashkda_reference_forward


requires_npu = pytest.mark.skipif(
    not is_available(), reason="requires an Ascend NPU"
)


@requires_npu
def test_native_npu_precision_against_reference():
    torch.manual_seed(42)
    batch, steps, heads, dim = 1, 128, 2, 128
    scale = 1.0 / math.sqrt(dim)
    q = F.normalize(
        torch.randn((batch, steps, heads, dim), dtype=torch.float32), dim=-1
    ).to(torch.bfloat16)
    k = F.normalize(
        torch.randn((batch, steps, heads, dim), dtype=torch.float32), dim=-1
    ).to(torch.bfloat16)
    v = torch.randn((batch, steps, heads, dim), dtype=torch.bfloat16)
    g = torch.randn((batch, steps, heads, dim), dtype=torch.bfloat16)
    beta = torch.randn((batch, steps, heads), dtype=torch.bfloat16)
    A_log = torch.rand(heads, dtype=torch.float32)
    dt_bias = torch.rand(heads, dim, dtype=torch.float32)

    q = q.to("npu")
    k = k.to("npu")
    v = v.to("npu")
    g = g.to("npu")
    beta = beta.to("npu")
    A_log = A_log.to("npu")
    dt_bias = dt_bias.to("npu")
    out = torch.empty_like(v)
    final_state = torch.empty((batch, heads, dim, dim), dtype=torch.float32, device="npu")

    npu_fwd(
        q, k, v, g, beta, scale, out,
        A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,
        initial_state=None, final_state=final_state, cu_seqlens=None,
    )

    ref_out, ref_state = flashkda_reference_forward(
        q, k, v, g.float(), beta.float(), scale, A_log, dt_bias,
        lower_bound=-5.0, chunk_size=16, initial_state=None,
    )

    torch.testing.assert_close(
        out.float(), ref_out.float().to("npu"), rtol=5e-3, atol=5e-3
    )
    torch.testing.assert_close(
        final_state.float(), ref_state.float().to("npu"), rtol=5e-3, atol=5e-3
    )
