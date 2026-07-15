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

"""Ascend NPU GEMM backend.

Registers NPU-optimized matrix multiplication implementations:
- ``torch.matmul`` for dense (fp16/bf16/fp32) matrices.
- ``torch_npu.npu_quant_matmul`` for FP8 scaled and block-scaled GEMMs.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    Platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import (
    ScaleFormat,
    format_signature,
    format_signatures,
    tensor_format,
)

_fp8_dtype = Platform.get().fp8e4m3fn.dtype

# Scale format for FP8 tensor-scaled GEMM
_FP8_TENSOR_SCALE = ScaleFormat(
    storage_dtype=torch.float32,
    granularity="tensor",
)

# Scale format for FP8 block-scaled GEMM (128x128 blocks)
_MXFP8_BLOCK_SCALE = ScaleFormat(
    storage_dtype=torch.float32,
    granularity="block",
    block_shape=(128, 128),
)

# Format signatures for dense (non-quantized) GEMM
_DENSE_FORMAT_SIGNATURES = format_signatures(
    ("a", "b"), "dense", {torch.float16, torch.bfloat16, torch.float32}
)

# Format signatures for FP8 tensor-scaled GEMM
_FP8_SCALED_FORMAT_SIGNATURES = format_signatures(
    ("a", "b"), "scaled-fp8", {_fp8_dtype}, scale=_FP8_TENSOR_SCALE
)

# Format signatures for FP8 block-scaled GEMM
_MXFP8_FORMAT_SIGNATURES = format_signatures(
    ("a", "b"), "mxfp8", {_fp8_dtype}, scale=_MXFP8_BLOCK_SCALE
)


def _is_npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _get_quant_matmul():
    """Return ``torch_npu.npu_quant_matmul`` if available, else ``None``."""
    try:
        import torch_npu  # noqa: F401
        return torch.ops.npu.npu_quant_matmul
    except (ImportError, AttributeError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# Dense MatMul (torch.matmul)
# ---------------------------------------------------------------------------


@register_kernel(
    "gemm",
    "mm",
    name="npu_mm",
    solution="ascend",
    capability=CapabilityRequirement(vendors=frozenset({"huawei"})),
    signatures=_DENSE_FORMAT_SIGNATURES,
    traits={
        "n_align_16": frozenset({False, True}),
        "k_align_16": frozenset({False, True}),
        "n_align_64": frozenset({False, True}),
        "n_align_128": frozenset({False, True}),
        "k_align_128": frozenset({False, True}),
    },
    priority=Priority.SPECIALIZED,
    tags={"portability"},
)
def npu_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor | None = None,
    B_scales: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    *,
    alpha: torch.Tensor | None = None,
    block_size: list[int] | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense matrix multiply using ``torch.matmul`` on Ascend NPU.

    Args:
        A: Activation tensor ``[M, K]``.
        B: Weight tensor ``[K, N]``.
        A_scales: Ignored for dense matmul.
        B_scales: Ignored for dense matmul.
        out_dtype: Output dtype (defaults to ``A.dtype``).
        alpha: Ignored for dense matmul.
        block_size: Ignored for dense matmul.
        bias: Optional bias vector ``[N]`` added to output.

    Returns:
        Output tensor ``[M, N]``.
    """
    del A_scales, B_scales, alpha, block_size
    if not _is_npu_available():
        raise RuntimeError("NPU not available")

    # Handle weight format: B may be [N, K] (transposed layout) or [K, N].
    # When B.shape[0] != A.shape[-1], B is in [N, K] layout → transpose to [K, N].
    if B.shape[0] == A.shape[-1]:
        out = torch.matmul(A, B)
    else:
        out = torch.matmul(A, B.T)

    if out_dtype is not None and out.dtype != out_dtype:
        out = out.to(out_dtype)
    if bias is not None:
        out = out + bias.to(dtype=out.dtype)
    return out


# ---------------------------------------------------------------------------
# FP8 Scaled MatMul (npu_quant_matmul)
# ---------------------------------------------------------------------------


@register_kernel(
    "gemm",
    "mm",
    name="npu_mm_fp8_scaled",
    solution="ascend",
    capability=CapabilityRequirement(vendors=frozenset({"huawei"})),
    signatures=_FP8_SCALED_FORMAT_SIGNATURES,
    traits={
        "quant": frozenset({"fp8"}),
    },
    priority=Priority.SPECIALIZED,
    tags={"portability"},
)
def npu_mm_fp8_scaled(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor | None = None,
    B_scales: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    *,
    alpha: torch.Tensor | None = None,
    block_size: list[int] | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """FP8 tensor-scaled matrix multiply using ``npu_quant_matmul``.

    Args:
        A: FP8 activation tensor ``[M, K]``.
        B: FP8 weight tensor ``[K, N]``.
        A_scales: Per-tensor activation scale (float32 scalar).
        B_scales: Per-tensor weight scale (float32 scalar).
        out_dtype: Output dtype (fp16 or bf16).
        alpha: Ignored in this implementation.
        block_size: Ignored for tensor-scaled.
        bias: Optional bias vector.

    Returns:
        Output tensor ``[M, N]``.
    """
    del alpha, block_size
    if not _is_npu_available():
        raise RuntimeError("NPU not available")

    quant_matmul = _get_quant_matmul()
    out_dtype = out_dtype or torch.float16

    if quant_matmul is not None:
        # npu_quant_matmul supports multiple call conventions;
        # use the per-tensor scalar format.
        out = quant_matmul(
            A, B,
            A_scales, B_scales,
            dtype=out_dtype,
        )
    else:
        # Fallback: dequantize and matmul
        A_fp = A.float()
        B_fp = B.float()
        if A_scales is not None and A_scales.numel() == 1:
            A_fp = A_fp * A_scales
        if B_scales is not None and B_scales.numel() == 1:
            B_fp = B_fp * B_scales
        out = torch.matmul(A_fp, B_fp).to(out_dtype)

    if bias is not None:
        out = out + bias.to(dtype=out.dtype)
    return out


# ---------------------------------------------------------------------------
# FP8 Block-Scaled MatMul (npu_quant_matmul)
# ---------------------------------------------------------------------------


@register_kernel(
    "gemm",
    "mm",
    name="npu_mm_fp8_blockscale",
    solution="ascend",
    capability=CapabilityRequirement(vendors=frozenset({"huawei"})),
    signatures=_MXFP8_FORMAT_SIGNATURES,
    traits={
        "quant": frozenset({"mxfp8"}),
    },
    priority=Priority.SPECIALIZED,
    tags={"portability"},
)
def npu_mm_fp8_blockscale(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor | None = None,
    B_scales: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    *,
    alpha: torch.Tensor | None = None,
    block_size: list[int] | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """FP8 block-scaled matrix multiply using ``npu_quant_matmul``.

    Args:
        A: FP8 activation tensor ``[M, K]``.
        B: FP8 weight tensor ``[K, N]``.
        A_scales: Per-block activation scales.
        B_scales: Per-block weight scales.
        out_dtype: Output dtype.
        alpha: Ignored in this implementation.
        block_size: Block dimensions ``[block_n, block_k]``.
        bias: Optional bias vector.

    Returns:
        Output tensor ``[M, N]``.
    """
    del alpha
    if not _is_npu_available():
        raise RuntimeError("NPU not available")

    quant_matmul = _get_quant_matmul()
    out_dtype = out_dtype or torch.float16

    if quant_matmul is not None and block_size is not None:
        out = quant_matmul(
            A, B,
            A_scales, B_scales,
            dtype=out_dtype,
            block_size=block_size,
        )
    else:
        # Fallback: dequantize block by block
        A_fp = _dequantize_block_fp8(A, A_scales, block_size) if A_scales is not None and block_size is not None else A.float()
        B_fp = _dequantize_block_fp8(B, B_scales, block_size) if B_scales is not None and block_size is not None else B.float()
        out = torch.matmul(A_fp, B_fp).to(out_dtype)

    if bias is not None:
        out = out + bias.to(dtype=out.dtype)
    return out


def _dequantize_block_fp8(
    x: torch.Tensor,
    scales: torch.Tensor | None,
    block_size: list[int] | None,
) -> torch.Tensor:
    """Dequantize a block-scaled FP8 tensor.

    Args:
        x: FP8 tensor.
        scales: Float scale tensor matching the block layout.
        block_size: ``[block_n, block_k]`` dimensions.

    Returns:
        Dequantized float32 tensor.
    """
    if scales is None or block_size is None:
        return x.float()

    x_f32 = x.float()
    block_k = block_size[1]

    # Per-row block scales (activation): scales[M, num_blocks_k]
    if scales.dim() == 2 and scales.shape[0] == x_f32.shape[0]:
        num_blocks = (x_f32.shape[-1] + block_k - 1) // block_k
        for i in range(num_blocks):
            start_k = i * block_k
            end_k = min(start_k + block_k, x_f32.shape[-1])
            x_f32[:, start_k:end_k] *= scales[:, i:i + 1]

    return x_f32
