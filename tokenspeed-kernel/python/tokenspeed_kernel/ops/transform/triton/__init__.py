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

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_NPU_AVAILABLE: bool | None = None


def _is_npu_available() -> bool:
    global _NPU_AVAILABLE
    if _NPU_AVAILABLE is None:
        _NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()
    return _NPU_AVAILABLE


def _hadamard_128_npu(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Pure PyTorch Walsh-Hadamard transform for last dim 128 (NPU fallback).

    Uses the iterative butterfly algorithm: for stride in [1,2,4,8,16,32,64],
    pair elements at distance stride and compute (a+b, a-b).
    """
    shape = x.shape
    x_f = x.reshape(-1, 128).float()
    n = 128
    stride = 1
    while stride < n:
        # Reshape to group pairs at distance `stride`
        x_f = x_f.reshape(-1, n // (stride * 2), stride * 2)
        a = x_f[:, :, :stride]
        b = x_f[:, :, stride:2 * stride]
        x_f = torch.cat([a + b, a - b], dim=-1)
        x_f = x_f.reshape(-1, n)
        stride *= 2
    return (x_f * scale).reshape(shape).to(x.dtype)


@triton.jit
def _hadamard_128_kernel(
    x,
    out,
    n_rows: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    row = tl.program_id(0)
    out_block = tl.program_id(1)
    out_offsets = out_block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    in_offsets = tl.arange(0, 128)

    vals = tl.load(x + row * 128 + in_offsets).to(tl.float32)
    bits = out_offsets[:, None] & in_offsets[None, :]
    parity = bits ^ (bits >> 1)
    parity = parity ^ (parity >> 2)
    parity = parity ^ (parity >> 4)
    parity = parity & 1
    signs = tl.where(parity == 0, 1.0, -1.0)
    acc = tl.sum(signs * vals[None, :], axis=1) * scale

    tl.store(
        out + row * 128 + out_offsets,
        acc,
        mask=(row < n_rows) & (out_offsets < 128),
    )


@register_kernel(
    "transform",
    "hadamard_transform",
    name="triton_hadamard_transform_128",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures(
        "x",
        "dense",
        {torch.bfloat16, torch.float16, torch.float32},
    ),
    traits={
        "last_dim": frozenset({128}),
    },
    priority=Priority.PORTABLE,
)
def triton_hadamard_transform_128(
    x: torch.Tensor,
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    """Apply a length-128 Sylvester Hadamard transform along the last dim."""
    if x.shape[-1] != 128:
        raise ValueError(
            f"triton_hadamard_transform_128 requires last dim 128, got {x.shape[-1]}"
        )
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(
            f"triton_hadamard_transform_128 does not support dtype {x.dtype}"
        )

    # NPU fallback: use pure PyTorch Hadamard
    if _is_npu_available():
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("triton_hadamard_transform_128 requires NPU device")
        return _hadamard_128_npu(x, scale=scale)

    if not x.is_cuda:
        raise RuntimeError("triton_hadamard_transform_128 requires a CUDA tensor")

    shape = x.shape
    x_2d = x.reshape(-1, 128).contiguous()
    out = torch.empty_like(x_2d)
    if x_2d.shape[0] == 0:
        return out.reshape(shape)
    _hadamard_128_kernel[(x_2d.shape[0], 8)](
        x_2d,
        out,
        n_rows=x_2d.shape[0],
        scale=float(scale),
        BLOCK_OUT=16,
        num_warps=8,
        num_stages=1,
    )
    return out.reshape(shape)


__all__ = ["triton_hadamard_transform_128"]
