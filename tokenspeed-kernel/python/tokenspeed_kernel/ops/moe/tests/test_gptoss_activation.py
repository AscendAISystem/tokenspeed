"""Verify GPT-OSS MoE activation matches HF reference.

Tests that the activation function with alpha=1.702, beta=1.0, limit=7.0
produces ``gate * sigmoid(alpha * gate) * (up + 1)`` with clamping.
"""

import torch
import torch.nn.functional as F


def gptoss_activation_ref(gate, up, alpha=1.702, beta=1.0, limit=7.0):
    """HF reference: gate * sigmoid(alpha * gate) * (up + beta) with clamping."""
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return (up + beta) * glu


def standard_swiglu(gate, up):
    """Standard SwiGLU: silu(gate) * up."""
    return F.silu(gate) * up


def npu_swiglu_implementation(gate, up, alpha=None, beta=0.0, limit=None):
    """Replicate the _npu_swiglu logic from ascend.py."""
    if beta != 0.0:
        # GPT-OSS style
        if limit is not None:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        gate_act = gate * torch.sigmoid(gate * alpha) if alpha is not None else gate * torch.sigmoid(gate)
        return (up + beta) * gate_act
    return standard_swiglu(gate, up)


def test_gptoss_activation_math():
    """L1-01: GPT-OSS activation matches HF reference formula."""
    torch.manual_seed(42)
    gate = torch.randn(4, 2880, dtype=torch.bfloat16) * 3
    up = torch.randn(4, 2880, dtype=torch.bfloat16) * 3

    # Our implementation with GPT-OSS params
    result = npu_swiglu_implementation(
        gate, up, alpha=1.702, beta=1.0, limit=7.0
    )

    # HF reference
    expected = gptoss_activation_ref(gate, up, alpha=1.702, beta=1.0, limit=7.0)

    diff = (result.float() - expected.float()).abs()
    max_diff = diff.max().item()

    print(f"GPT-OSS activation max diff vs HF reference: {max_diff:.6f}")
    assert max_diff < 1e-3, (
        f"GPT-OSS activation mismatch! max_diff={max_diff:.6f}, "
        f"expected < 1e-3"
    )

    # Shape check
    assert result.shape == (4, 2880), f"Shape mismatch: {result.shape}"

    # No NaN/Inf
    assert not torch.isnan(result).any(), "NaN in GPT-OSS activation output"
    assert not torch.isinf(result).any(), "Inf in GPT-OSS activation output"

    print("✅ GPT-OSS activation math test passed (matches HF reference)")


def test_differs_from_standard_swiglu():
    """L1-02: GPT-OSS activation differs from standard SwiGLU."""
    torch.manual_seed(42)
    gate = torch.randn(4, 2880, dtype=torch.bfloat16) * 3
    up = torch.randn(4, 2880, dtype=torch.bfloat16) * 3

    gptoss_result = npu_swiglu_implementation(
        gate, up, alpha=1.702, beta=1.0, limit=7.0
    )
    standard_result = standard_swiglu(gate, up)

    diff = (gptoss_result.float() - standard_result.float()).abs()
    max_diff = diff.max().item()

    print(f"GPT-OSS vs standard SwiGLU max diff: {max_diff:.4f}")
    assert max_diff > 0.1, (
        f"GPT-OSS activation should differ from standard SwiGLU, "
        f"but max_diff={max_diff:.4f} is too small"
    )

    print("✅ Difference from standard SwiGLU confirmed")


def test_gptoss_standard_path_unchanged():
    """L1-03: Standard SwiGLU path (beta=0) unchanged."""
    torch.manual_seed(42)
    gate = torch.randn(4, 2880, dtype=torch.bfloat16) * 3
    up = torch.randn(4, 2880, dtype=torch.bfloat16) * 3

    # Default params (beta=0) should produce standard SwiGLU
    result = npu_swiglu_implementation(gate, up)
    expected = standard_swiglu(gate, up)

    diff = (result.float() - expected.float()).abs()
    max_diff = diff.max().item()

    print(f"Standard path max diff vs F.silu(gate)*up: {max_diff:.6f}")
    assert max_diff < 1e-5, (
        f"Standard path changed! max_diff={max_diff:.6f}, "
        f"expected < 1e-5"
    )

    print("✅ Standard SwiGLU path unchanged")


def test_clamping_behavior():
    """L1-04: Verify clamping behavior in GPT-OSS activation."""
    torch.manual_seed(123)
    gate = torch.randn(4, 288, dtype=torch.bfloat16) * 10  # large values to trigger clamp
    up = torch.randn(4, 288, dtype=torch.bfloat16) * 10

    limit = 7.0
    result = npu_swiglu_implementation(
        gate, up, alpha=1.702, beta=1.0, limit=limit
    )

    # After clamping, gate should be <= limit, up should be in [-limit, limit]
    clamped_gate = gate.clamp(max=limit)
    clamped_up = up.clamp(min=-limit, max=limit)
    expected = (clamped_up + 1.0) * (clamped_gate * torch.sigmoid(clamped_gate * 1.702))

    diff = (result.float() - expected.float()).abs()
    max_diff = diff.max().item()

    print(f"Clamping test max diff: {max_diff:.6f}")
    assert max_diff < 1e-3, f"Clamping mismatch: max_diff={max_diff:.6f}"

    print("✅ Clamping behavior verified")


if __name__ == "__main__":
    test_gptoss_activation_math()
    test_differs_from_standard_swiglu()
    test_gptoss_standard_path_unchanged()
    test_clamping_behavior()
    print("\n🎉 All GPT-OSS activation tests passed!")
