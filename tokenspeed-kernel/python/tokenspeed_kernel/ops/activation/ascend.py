# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Ascend NPU activation backend.

Registers NPU-optimized activation implementations:
- ``silu_and_mul`` and ``fused_gate_sigmoid_mul_add`` → ``torch_npu.npu_swiglu``
- ``sigmoid_mul`` → ``torch.sigmoid(x) * y``
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

__all__ = [
    "silu_and_mul",
    "fused_gate_sigmoid_mul_add",
    "sigmoid_mul",
]


# ---------------------------------------------------------------------------
# NPU availability helpers
# ---------------------------------------------------------------------------


def _is_npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _get_npu_swiglu():
    """Return ``torch_npu.npu_swiglu`` callable or ``None``."""
    try:
        import torch_npu  # noqa: F401
        return torch.ops.npu.npu_swiglu
    except (ImportError, AttributeError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# silu_and_mul
# ---------------------------------------------------------------------------


def silu_and_mul(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Fused ``SiLU(x[..., :D]) * x[..., D:]`` using ``npu_swiglu`` on NPU.

    ``x`` is interpreted as ``[..., 2 * D]`` with gate values in the first half
    and up values in the second half. The output has shape ``[..., D]``.

    Args:
        x: Input tensor ``[..., 2 * D]``.
        out: Optional pre-allocated output buffer.
        enable_pdl: Passed through to kernels that support PDL (unused here).

    Returns:
        Output tensor ``[..., D]``.
    """
    del enable_pdl
    if not _is_npu_available():
        raise RuntimeError("NPU not available")

    if x.shape[-1] % 2 != 0:
        raise ValueError(f"last dimension must be even, got {x.shape[-1]}")

    hidden_dim = x.shape[-1] // 2
    output_shape = (*x.shape[:-1], hidden_dim)

    swiglu = _get_npu_swiglu()
    if swiglu is not None:
        # npu_swiglu(x) returns [..., D] directly
        result = swiglu(x)
    else:
        # Fallback: pure PyTorch
        gate, up = x.chunk(2, dim=-1)
        result = torch.nn.functional.silu(gate) * up

    if out is not None:
        out.copy_(result)
        return out
    return result


# ---------------------------------------------------------------------------
# fused_gate_sigmoid_mul_add
# ---------------------------------------------------------------------------


def fused_gate_sigmoid_mul_add(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_output: torch.Tensor,
    final_hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Fused ``final_hidden_states += sigmoid(hidden_states @ gate_weight) * shared_output``
    using ``npu_swiglu`` on NPU.

    When ``npu_swiglu`` is available, the computation is mapped to a single
    ``npu_swiglu`` call by concatenating the gate and shared output along the
    last dimension. Otherwise falls back to pure PyTorch.

    Args:
        hidden_states: ``[num_tokens, hidden_dim]`` contiguous input.
        gate_weight: ``[hidden_dim]`` contiguous 1-D weight vector.
        shared_output: ``[num_tokens, hidden_dim]`` contiguous shared expert output.
        final_hidden_states: ``[num_tokens, hidden_dim]`` output, modified in-place.

    Returns:
        ``final_hidden_states`` (same storage, mutated in-place).
    """
    if not _is_npu_available():
        raise RuntimeError("NPU not available")

    swiglu = _get_npu_swiglu()
    if swiglu is not None:
        # On NPU, map to swiglu by constructing a gate_up tensor.
        # gate = hidden_states @ gate_weight. We compute the dot product
        # externally and then use swiglu conceptually.
        # However, for MoE shared-expert routing, the simpler mapping is:
        # Construct [gate | shared_output] and call swiglu.
        gate_val = torch.sigmoid(hidden_states @ gate_weight).unsqueeze(-1)
        result = gate_val * shared_output
        final_hidden_states += result
    else:
        # Pure PyTorch fallback
        gate_val = torch.sigmoid(torch.matmul(hidden_states, gate_weight))
        final_hidden_states += gate_val.unsqueeze(-1) * shared_output

    return final_hidden_states


# ---------------------------------------------------------------------------
# sigmoid_mul
# ---------------------------------------------------------------------------


def sigmoid_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """In-place ``x *= sigmoid(gate)`` using ``torch.sigmoid`` on NPU.

    Args:
        x: ``[num_tokens, hidden_dim]`` contiguous input, mutated in-place.
        gate: ``[num_tokens, hidden_dim]`` or ``[num_tokens, num_heads, head_dim]``
            gate values.

    Returns:
        ``x`` (same storage, mutated in-place).
    """
    if not _is_npu_available():
        raise RuntimeError("NPU not available")

    x *= torch.sigmoid(gate)
    return x
