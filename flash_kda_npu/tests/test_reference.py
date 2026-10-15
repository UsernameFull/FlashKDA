"""CPU checks for the independent oracle and CUDA/NPU case correspondence."""
import ast
from pathlib import Path

import pytest
import torch

from flash_kda_npu.tests import cuda_cases
from flash_kda_npu.tests.torch_ref import (
    CUDA_GATE_CASE_IDS, assert_cuda_accuracy, assert_post_transient_accuracy,
    chunked_reference, make_cuda_gate_case, make_cuda_inputs, make_cuda_state,
    recurrent_reference,
)


def test_cuda_case_manifest_matches_source():
    root = Path(__file__).resolve().parents[2]
    def constants(path):
        return {node.targets[0].id: ast.dump(node.value)
                for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    actual = constants(Path(cuda_cases.__file__))
    expected = constants(root / "tests/test_fwd_full.py")
    for name, value in actual.items():
        assert expected[name] == value, f"NPU case matrix drifted from CUDA: {name}"
    count = (len(cuda_cases.H_VALUES) * len(cuda_cases.T_VALUES)
             + len(cuda_cases.VARLEN_H) * len(cuda_cases.VARLEN_CASES)
             + len(cuda_cases.BATCH_H) * len(cuda_cases.BATCH_CASES)) * 8
    assert count + len(cuda_cases.LONG_T_VALUES) + len(cuda_cases.LONG_VARLEN_CASES) == 580


def test_cuda_parametrization_matches_source():
    root = Path(__file__).resolve().parents[2]

    def decorators(path):
        return {node.name: [ast.dump(d) for d in node.decorator_list]
                for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name in {
                    "test_fwd_fixed", "test_fwd_varlen", "test_fwd_batched",
                    "test_fwd_long", "test_fwd_long_varlen"}}

    assert decorators(Path(__file__).with_name("test_fwd_full.py")) == decorators(
        root / "tests/test_fwd_full.py")


@pytest.mark.parametrize("chunk_size", [1, 7, 16, 32])
@pytest.mark.parametrize("lengths", [None, [0, 1, 17, 0, 20], [0, 0]])
@pytest.mark.parametrize("has_initial", [False, True])
def test_chunked_oracle_matches_token_recurrence(chunk_size, lengths, has_initial):
    batch, steps = (2, 19) if lengths is None else (1, sum(lengths))
    inputs = make_cuda_inputs(batch, steps, 2)
    # Include zero and tiny norms: both references must use sqrt(sum(x²)+eps).
    inputs[0][:, :1].zero_()
    inputs[1][:, :1].mul_(1e-5)
    sequences = batch if lengths is None else len(lengths)
    initial = make_cuda_state(sequences, 2, torch.float32) if has_initial else None
    kwargs = dict(seq_lens=lengths, scale=0.17, lower_bound=-3.0)
    actual = chunked_reference(inputs, initial, chunk_size=chunk_size, **kwargs)
    expected = recurrent_reference(inputs, initial, **kwargs)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)


def test_cuda_fixture_properties():
    a = make_cuda_inputs(1, 1024, 2)
    torch.randn(17)  # The fixture must not depend on or reset the global RNG.
    b = make_cuda_inputs(1, 1024, 2)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert abs(a[3].float().mean().item()) < 0.02
    assert 0.95 < a[3].float().std().item() < 1.05
    assert 0.4 < a[-1].mean().item() < 0.6
    state16 = make_cuda_state(2, 3, torch.bfloat16)
    state32 = make_cuda_state(2, 3, torch.float32)
    torch.testing.assert_close(state16.float(), state32, rtol=0, atol=0)
    assert state32.max() > 90000


@pytest.mark.parametrize("case", range(len(CUDA_GATE_CASE_IDS)), ids=CUDA_GATE_CASE_IDS)
def test_gate_oracle_matches_token_recurrence(case):
    inputs, initial = make_cuda_gate_case(case, [37])
    actual = chunked_reference(inputs, initial)
    expected = recurrent_reference(inputs, initial)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)


def test_accuracy_checks_reject_hidden_late_errors():
    expected = torch.ones(1, 2048, 1, 1)
    expected[:, :64] = 1e6  # Global RMS can hide corruption after the transient.
    actual = expected.clone()
    actual[:, 1500:] *= 1.1
    record = lambda key, value: None
    with pytest.raises(AssertionError, match="output window"):
        assert_cuda_accuracy(actual, expected, "output", record, threshold=0.01)
    with pytest.raises(AssertionError, match="post-transient"):
        assert_post_transient_accuracy(actual, expected, [2048], record)
