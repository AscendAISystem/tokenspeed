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
# IMPLIED, BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Ascend NPU layernorm backend.

Registers NPU-optimized RMSNorm implementations:
- ``rmsnorm`` and ``fused_add_rmsnorm`` → ``torch_npu.npu_add_rms_norm``
- ``qk_rmsnorm`` → ``torch.nn.functional.rms_norm`` × 2
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

__all__ = [
    "rmsnorm",
    "fused_add_rmsnorm",
    "qk_rmsnorm",
]


# ---------------------------------------------------------------------------
# NPU availability helpers
# ---------------------------------------------------------------------------


def _is_npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _get_npu_add_rms_norm():
    """Return ``torch_npu.npu_add_rms_norm`` callable or ``None``."""
    try:
        import torch_npu  # noqa: F401
        return torch.ops.npu.npu_add_rms_norm
    except (ImportError, AttributeError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# rmsnorm
# ---------------------------------------------------------------------------


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """RMS Normalization on Ascend NPU.

    Uses ``npu_add_rms_norm`` when available (supports fused residual),
    otherwise falls back to ``F.rms_norm``.

    Args:
        x: Input tensor ``[..., hidden_size]``.
        weight: Learnable weight ``[hidden_size]``.
        eps: Epsilon for numerical stability.
        residual: Optional residual tensor to add before normalization.
        out: Optional pre-allocated output buffer.

    Returns:
        Normalized output, or ``(output, residual_out)`` when residual is given.
    """
    if not _is_npu_available():
        raise RuntimeError("NPU not available")

    hidden_size = x.shape[-1]
    if weight.shape[0] != hidden_size:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} does not match hidden size {hidden_size}"
        )
    if residual is not None and residual.shape != x.shape:
        raise ValueError(
            f"residual shape {tuple(residual.shape)} does not match input shape {tuple(x.shape)}"
        )

    x_2d = x.reshape(-1, hidden_size)
    if out is None:
        out = torch.empty_like(x)
    # NOTE: out_2d is a VIEW of out. After tuple unpacking from add_rms_norm,
    # out_2d gets reassigned to the returned result tensor. We capture the
    # result in a separate variable and copy back if out was pre-allocated.

    # Ensure weight dtype matches input dtype (NPU kernel requires matching dtypes)
    weight_cast = weight.to(x.dtype) if weight.dtype != x.dtype else weight

    out_prealloc = out is not None
    add_rms_norm = _get_npu_add_rms_norm()
    if add_rms_norm is not None and residual is not None:
        # npu_add_rms_norm(x, residual, weight, eps) -> (output, norm_scale, new_residual)
        residual_2d = residual.reshape(-1, hidden_size)
        result_2d, _, residual_out_2d = add_rms_norm(x_2d, residual_2d, weight_cast, eps)
        if out_prealloc:
            out.reshape(-1, hidden_size).copy_(result_2d)
        residual_out = residual_out_2d.reshape(residual.shape)
        return (out.reshape(x.shape) if out_prealloc else result_2d.reshape(x.shape)), residual_out
    elif add_rms_norm is not None:
        # Use npu_add_rms_norm without residual (returns output, norm_scale, updated_input)
        result_2d, _, _ = add_rms_norm(x_2d, x_2d, weight_cast, eps)
        if out_prealloc:
            out.reshape(-1, hidden_size).copy_(result_2d)
            return out.reshape(x.shape)
        return result_2d.reshape(x.shape)
    else:
        # Fallback: F.rms_norm
        if residual is not None:
            x = x + residual
            residual_out = x.clone()
            result = F.rms_norm(x, (hidden_size,), weight=weight, eps=eps)
            if out is not None:
                out.copy_(result)
            return out.reshape(x.shape) if out is not None else result, residual_out
        else:
            result = F.rms_norm(x, (hidden_size,), weight=weight, eps=eps)
            if out is not None:
                out.copy_(result)
            return out.reshape(x.shape) if out is not None else result


# ---------------------------------------------------------------------------
# fused_add_rmsnorm
# ---------------------------------------------------------------------------


def fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused add + RMSNorm using ``npu_add_rms_norm`` on Ascend NPU.

    Computes ``residual_out = x + residual`` and
    ``output = rms_norm(residual_out, weight)`` in a single fused operation.

    Args:
        x: Input tensor ``[..., hidden_size]``.
        residual: Residual tensor ``[..., hidden_size]``, updated in-place.
        weight: Learnable weight ``[hidden_size]``.
        eps: Epsilon for numerical stability.

    Returns:
        ``(output, residual_out)`` where ``residual_out`` is the updated residual.
    """
    if not _is_npu_available():
        raise RuntimeError("NPU not available")

    hidden_size = x.shape[-1]
    x_2d = x.reshape(-1, hidden_size)
    residual_2d = residual.reshape(-1, hidden_size)

    # Ensure weight dtype matches input dtype
    weight_cast = weight.to(x.dtype) if weight.dtype != x.dtype else weight

    add_rms_norm = _get_npu_add_rms_norm()
    if add_rms_norm is not None:
        out_2d, _, residual_out_2d = add_rms_norm(x_2d, residual_2d, weight_cast, eps)
        out = out_2d.reshape(x.shape)
        residual_out = residual_out_2d.reshape(residual.shape)
    else:
        # Fallback
        residual_out = x + residual
        out = F.rms_norm(residual_out, (hidden_size,), weight=weight, eps=eps)

    return out, residual_out


# ---------------------------------------------------------------------------
# qk_rmsnorm
# ---------------------------------------------------------------------------


def qk_rmsnorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head RMSNorm for Q and K using ``F.rms_norm`` on Ascend NPU.

    Args:
        q: Query tensor.
        k: Key tensor.
        q_weight: Q RMSNorm weight ``[head_dim]``.
        k_weight: K RMSNorm weight ``[head_dim]``.
        eps: Epsilon for numerical stability.

    Returns:
        ``(q_out, k_out)`` normalized tensors.
    """
    if not _is_npu_available():
        raise RuntimeError("NPU not available")

    hidden_size = q.shape[-1]
    head_dim = q_weight.shape[0]
    num_q_heads = hidden_size // head_dim

    # Reshape to [n_tokens, heads, head_dim] for per-head RMS norm
    q_3d = q.reshape(-1, num_q_heads, head_dim)
    k_3d = k.reshape(-1, k.shape[-1] // head_dim, head_dim)

    # Apply F.rms_norm per head (normalized_shape = [head_dim])
    q_out = F.rms_norm(q_3d, (head_dim,), weight=q_weight, eps=eps)
    k_out = F.rms_norm(k_3d, (head_dim,), weight=k_weight, eps=eps)

    return q_out.reshape(q.shape), k_out.reshape(k.shape)
