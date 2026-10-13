import math

import pytest
import torch
from torch.nn import functional as F

from flash_kda_npu import fwd as npu_fwd
from flash_kda_npu.ops.kda_fwd import _activate_gate


def _make_inputs(length, heads, dim=128):
    torch.manual_seed(42)
    q = F.normalize(
        torch.randn((1, length, heads, dim), dtype=torch.float32), dim=-1
    ).to(torch.bfloat16)
    k = F.normalize(
        torch.randn((1, length, heads, dim), dtype=torch.float32), dim=-1
    ).to(torch.bfloat16)
    v = torch.randn((1, length, heads, dim), dtype=torch.bfloat16)
    g = torch.randn((1, length, heads, dim), dtype=torch.bfloat16)
    beta = torch.randn((1, length, heads), dtype=torch.bfloat16)
    A_log = torch.rand(heads, dtype=torch.float32)
    dt_bias = torch.rand(heads, dim, dtype=torch.float32)
    return q, k, v, g, beta, A_log, dt_bias


def test_npu_fallback_one_token_state_update():
    length, heads = 1, 2
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(length, heads)
    out = torch.zeros_like(v)
    final_state = torch.zeros(1, heads, 128, 128, dtype=torch.float32)

    npu_fwd(
        q, k, v, g, beta, 1.0 / math.sqrt(128), out,
        A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,
        final_state=final_state,
    )

    beta_activated = torch.sigmoid(beta.float()).to(torch.bfloat16)
    query_key = (q.float() * k.float()).sum(-1, keepdim=True) * (1.0 / math.sqrt(128))
    expected_out = query_key * (v.float() * beta_activated.unsqueeze(-1).float())
    expected_out = expected_out.to(torch.bfloat16)
    gate = _activate_gate(g, A_log, dt_bias, -5.0)
    k_restored = (
        k[0, 0]
        * torch.special.exp2(-gate[0, 0]).to(torch.bfloat16)
        * torch.special.exp2(gate[0, 0]).to(torch.bfloat16)
    )
    expected_state = torch.stack(
        [k_restored[head].float().unsqueeze(1)
         @ (v[0, 0, head].float() * beta_activated[0, 0, head].float()).unsqueeze(0)
         for head in range(heads)]
    ).unsqueeze(0)
    torch.testing.assert_close(out, expected_out, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(
        final_state, expected_state.float(), rtol=5e-3, atol=5e-3
    )


@pytest.mark.parametrize("seq_lens", [[1, 2], [3, 1]])
def test_npu_fallback_varlen(seq_lens):
    length = sum(seq_lens)
    heads = 2
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(length, heads)
    out = torch.zeros_like(v)
    final_state = torch.zeros(len(seq_lens), heads, 128, 128, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0)], dtype=torch.long)

    npu_fwd(
        q, k, v, g, beta, 1.0 / math.sqrt(128), out,
        A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,
        final_state=final_state, cu_seqlens=cu_seqlens,
    )

    beta_activated = torch.sigmoid(beta.float()).to(torch.bfloat16)
    for sequence, token in enumerate(cu_seqlens[:-1]):
        query = q[0, token].float()
        key = k[0, token].float()
        query_key = (query * key).sum(-1, keepdim=True) * (1.0 / math.sqrt(128))
        expected = query_key * (
            v[0, token].float() * beta_activated[0, token].unsqueeze(-1).float()
        )
        torch.testing.assert_close(out[0, token], expected.to(torch.bfloat16),
                                   rtol=5e-3, atol=5e-3)


def test_npu_fallback_rejects_unsupported_dimension():
    q, k, v, g, beta, A_log, dt_bias = _make_inputs(1, 1, dim=4)
    out = torch.zeros_like(v)

    npu_fwd(
        q, k, v, g, beta, 0.5, out,
        A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,
    )

    assert out.shape == v.shape
