"""Native AscendC coverage; fallback is a failure, never a passing test."""
import pytest
import torch

import flash_kda
from flash_kda_npu.ops import kda_fwd
from flash_kda_npu.tests.helpers import (
    STATE_IDS, STATE_IO, assert_close, assert_relative_rms, make_inputs, make_state, recurrent_reference,
)

pytestmark = pytest.mark.skipif(not kda_fwd.is_available(), reason="requires an Ascend NPU")


@pytest.fixture
def native_only(monkeypatch):
    def refuse_fallback(*args, **kwargs):
        pytest.fail("native case unexpectedly selected the PyTorch fallback")
    monkeypatch.setattr(kda_fwd, "_torch_fallback", refuse_fallback)


def check_native(batch, steps, heads, state_dtype=torch.float32,
                 has_in=True, has_out=True, gate_dtype=torch.bfloat16,
                 gate_value=None, lower_bound=-5.0):
    inputs = make_inputs(batch, steps, heads, "npu", gate_dtype=gate_dtype)
    if gate_value is not None:
        inputs[3].fill_(gate_value)
    initial = make_state(batch, heads, 128, state_dtype, "npu") if has_in else None
    saved_initial = initial.clone() if initial is not None else None
    final = torch.full((batch, heads, 128, 128), float("nan"), device="npu") if has_out else None
    out = torch.full_like(inputs[2], float("nan"))
    q, k, v, g, beta, a_log, bias = inputs
    result = flash_kda.fwd(q, k, v, g, beta, 128 ** -0.5, out,
                           A_log=a_log, dt_bias=bias, lower_bound=lower_bound,
                           initial_state=initial, final_state=final)
    torch.npu.synchronize()
    assert result is None
    from flash_kda_npu.ops.ascendc import get_last_profile
    assert get_last_profile()["launch_counts"].get("kda_k2_persistent_loop", 0) == 1
    expected_out, expected_state = recurrent_reference(inputs, saved_initial, lower_bound=lower_bound)
    assert_close(out, expected_out)
    if final is not None:
        # CUDA gate sweeps use relative RMS: weak decay accumulates bf16
        # rounding across tokens, making per-element relative error unstable.
        if gate_value is not None:
            assert_relative_rms(final, expected_state)
        else:
            assert_close(final, expected_state)
    if initial is not None:
        torch.testing.assert_close(initial, saved_initial, rtol=0, atol=0)


@pytest.mark.parametrize("has_in,has_out", STATE_IO, ids=STATE_IDS)
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batch,steps,heads", [(1, 64, 1), (1, 256, 4), (2, 128, 4)])
def test_native_state_io(native_only, batch, steps, heads, state_dtype, has_in, has_out):
    check_native(batch, steps, heads, state_dtype, has_in, has_out)


@pytest.mark.parametrize("heads", [1, 4, 32, 96])
def test_native_head_mapping(native_only, heads):
    check_native(1, 64, heads)


@pytest.mark.parametrize("batch,steps,heads", [(4, 256, 1), (8, 128, 4)])
def test_native_batched(native_only, batch, steps, heads):
    check_native(batch, steps, heads)


@pytest.mark.parametrize("gate_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("gate_value", [-8.0, -4.0, 0.0, 8.0])
def test_native_gate_sweep(native_only, gate_dtype, gate_value):
    check_native(1, 128, 2, gate_dtype=gate_dtype, gate_value=gate_value)


@pytest.mark.parametrize("lower_bound", [-1.0, -3.0])
def test_native_lower_bound(native_only, lower_bound):
    check_native(1, 128, 2, lower_bound=lower_bound)


@pytest.mark.parametrize("steps", [1024, 8192])
def test_native_multichunk(native_only, steps):
    check_native(1, steps, 1)


@pytest.mark.parametrize("steps", [131072, 1048576])
def test_native_long(native_only, request, steps):
    if not request.config.getoption("--run-long"):
        pytest.skip("use --run-long for the CUDA-scale long-sequence cases")
    check_native(1, steps, 1)


@pytest.mark.parametrize("chunk", [16, 32, 64])
def test_native_configured_chunk(native_only, chunk):
    from flash_kda_npu.ops.ascendc import CHUNK, kda_bt16_fwd_ascendc
    if CHUNK != chunk:
        pytest.skip(f"requires a fresh process with KDA_CHUNK={chunk}")
    inputs = make_inputs(1, chunk * 3, 2, "npu", gate_dtype=torch.float32)
    q, k, v, g, beta, a_log, bias = inputs
    out, state = kda_bt16_fwd_ascendc(q, k, v, g, beta.float(),
                                    A_log=a_log, bias=bias, output_final_state=True)
    torch.npu.synchronize()
    expected_out, expected_state = recurrent_reference(inputs)
    assert_close(out, expected_out)
    assert_close(state, expected_state)
