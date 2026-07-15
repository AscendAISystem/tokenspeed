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

"""Ascend NPU rotary embedding backend.

Registers NPU-optimized RoPE implementation using ``torch_npu.npu_rotary_embedding``.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

# ---------------------------------------------------------------------------
# NPU availability helpers
# ---------------------------------------------------------------------------


def _is_npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _get_npu_rotary_embedding():
    """Return ``torch_npu.npu_rotary_embedding`` callable or ``None``."""
    try:
        import torch_npu  # noqa: F401
        return torch.ops.npu.npu_rotary_embedding
    except (ImportError, AttributeError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# apply_rope — registered kernel for select_kernel("embedding", "rope")
# ---------------------------------------------------------------------------


@register_kernel(
    "embedding",
    "rope",
    name="npu_embedding_rope",
    solution="ascend",
    capability=CapabilityRequirement(vendors=frozenset({"huawei"})),
    signatures=format_signatures(
        ("q", "k"), "dense", {torch.float16, torch.bfloat16}
    ),
    priority=Priority.SPECIALIZED,
    traits={
        "partial_rotary": frozenset({True, False}),
        "is_neox": frozenset({True, False}),
        "has_fused_kv": frozenset({True, False}),
        "has_q_out": frozenset({True, False}),
        "has_k_out": frozenset({True, False}),
    },
    tags={"portability"},
)
def npu_embedding_rope(
    *,
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
    fused_set_kv_buffer_arg: Any = None,
    q_rope_out: torch.Tensor | None = None,
    k_rope_out: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> None:
    """Apply rotary positional embedding using ``npu_rotary_embedding``.

    Args:
        positions: Token positions, 1D ``[num_tokens]``.
        q: Query tensor ``[num_tokens, num_q_heads * head_size]``.
        k: Key tensor ``[num_tokens, num_kv_heads * head_size]``.
        head_size: Per-head dimension.
        cos_sin_cache: ``[max_position, rotary_dim]`` packed as concat(cos, sin).
        is_neox: If True, use Neox-style half-split rotation.
        fused_set_kv_buffer_arg: Optional fused KV-cache write (not supported on NPU).
        q_rope_out: Optional output buffer for rotated Q.
        k_rope_out: Optional output buffer for rotated K.
        enable_pdl: Passed through (unused here).
    """
    del enable_pdl
    if not _is_npu_available():
        raise RuntimeError("NPU not available")

    rotary_dim = cos_sin_cache.shape[-1]
    assert rotary_dim % 2 == 0, "rotary_dim must be even"
    assert rotary_dim <= head_size, "rotary_dim must be <= head_size"

    num_tokens = positions.shape[0]
    if num_tokens == 0:
        return

    num_q_heads = q.shape[-1] // head_size
    num_kv_heads = k.shape[-1] // head_size

    # Reshape to [num_tokens, num_heads, head_size]
    q_3d = q.reshape(num_tokens, -1, head_size)
    k_3d = k.reshape(num_tokens, -1, head_size)

    # Determine output views
    q_out_3d = q_rope_out.reshape(num_tokens, num_q_heads, head_size) if q_rope_out is not None else q_3d
    k_out_3d = k_rope_out.reshape(num_tokens, num_kv_heads, head_size) if k_rope_out is not None else k_3d

    rotary_emb = _get_npu_rotary_embedding()
    if rotary_emb is not None:
        # npu_rotary_embedding: applies rotation using cos/sin cache
        # Format depends on the API version; typically:
        #   npu_rotary_embedding(x, cos, sin) -> rotated_x
        cos = cos_sin_cache[positions, :rotary_dim // 2]  # [num_tokens, half_rotary]
        sin = cos_sin_cache[positions, rotary_dim // 2:rotary_dim]  # [num_tokens, half_rotary]

        if is_neox:
            # Neox style: apply rotation in halves
            q_rotated = _apply_neox_rope(q_3d, cos, sin, rotary_dim, rotary_emb)
            k_rotated = _apply_neox_rope(k_3d, cos, sin, rotary_dim, rotary_emb)
        else:
            # GPT-J style: interleaved pairs
            q_rotated = _apply_gptj_rope(q_3d, cos, sin, rotary_dim, rotary_emb)
            k_rotated = _apply_gptj_rope(k_3d, cos, sin, rotary_dim, rotary_emb)

        q_out_3d.copy_(q_rotated)
        k_out_3d.copy_(k_rotated)
    else:
        # Pure PyTorch fallback
        _fallback_rope(q_3d, k_3d, q_out_3d, k_out_3d, positions, cos_sin_cache, head_size, rotary_dim, is_neox)

    # Fused KV-cache write: write rotated K and original V into KV cache buffers
    if fused_set_kv_buffer_arg is not None:
        if (fused_set_kv_buffer_arg.k_scale is not None
                or fused_set_kv_buffer_arg.v_scale is not None):
            raise ValueError("k_scale/v_scale are not supported in NPU RoPE backend")
        if fused_set_kv_buffer_arg.cache_loc is None:
            raise ValueError("fused_set_kv_buffer_arg.cache_loc is required")

        cache_loc = fused_set_kv_buffer_arg.cache_loc
        value = fused_set_kv_buffer_arg.value  # V tensor
        # Write rotated K into KV cache
        k_buffer_view = fused_set_kv_buffer_arg.k_buffer.view(
            fused_set_kv_buffer_arg.k_buffer.shape[0], num_kv_heads, head_size
        )
        k_buffer_view[cache_loc] = k_out_3d
        # Write V into KV cache
        v_buffer_view = fused_set_kv_buffer_arg.v_buffer.view(
            fused_set_kv_buffer_arg.v_buffer.shape[0], num_kv_heads, head_size
        )
        v_buffer_view[cache_loc] = value.view(num_tokens, num_kv_heads, head_size)


# ---------------------------------------------------------------------------
# NPU RoPE helper implementations
# ---------------------------------------------------------------------------


def _apply_neox_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
    rotary_emb,
) -> torch.Tensor:
    """Apply Neox-style RoPE using npu_rotary_embedding or native ops."""
    half = rotary_dim // 2
    x1 = x[..., :half]
    x2 = x[..., half:rotary_dim]

    cos_2d = cos.unsqueeze(1)  # [num_tokens, 1, half]
    sin_2d = sin.unsqueeze(1)

    try:
        # Try the NPU rotary embedding op directly
        rotated = rotary_emb(x, cos, sin)
        return rotated
    except (RuntimeError, TypeError):
        pass

    # Pure PyTorch Neox rotation
    o1 = x1 * cos_2d - x2 * sin_2d
    o2 = x2 * cos_2d + x1 * sin_2d
    rotated = torch.cat([o1, o2], dim=-1)
    if rotary_dim < x.shape[-1]:
        rotated = torch.cat([rotated, x[..., rotary_dim:]], dim=-1)
    return rotated


def _apply_gptj_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
    rotary_emb,
) -> torch.Tensor:
    """Apply GPT-J-style RoPE using npu_rotary_embedding or native ops.

    In GPT-J layout, pairs are interleaved: [x0, x1, x0, x1, ...].
    """
    half = rotary_dim // 2

    try:
        rotated = rotary_emb(x, cos, sin)
        return rotated
    except (RuntimeError, TypeError):
        pass

    # Pure PyTorch GPT-J rotation
    x_rot_part = x[..., :rotary_dim]
    x_reshaped = x_rot_part.reshape(*x.shape[:-1], half, 2)
    x1 = x_reshaped[..., 0]  # even indices
    x2 = x_reshaped[..., 1]  # odd indices

    cos_2d = cos.unsqueeze(1)
    sin_2d = sin.unsqueeze(1)

    o1 = x1 * cos_2d - x2 * sin_2d
    o2 = x2 * cos_2d + x1 * sin_2d

    rotated = torch.stack([o1, o2], dim=-1).reshape(*x.shape[:-1], rotary_dim)
    if rotary_dim < x.shape[-1]:
        rotated = torch.cat([rotated, x[..., rotary_dim:]], dim=-1)
    return rotated


def _fallback_rope(
    q_3d: torch.Tensor,
    k_3d: torch.Tensor,
    q_out_3d: torch.Tensor,
    k_out_3d: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_size: int,
    rotary_dim: int,
    is_neox: bool,
) -> None:
    """Pure PyTorch fallback for RoPE when NPU API is unavailable."""
    half = rotary_dim // 2
    cos = cos_sin_cache[positions, :half]  # [num_tokens, half]
    sin = cos_sin_cache[positions, half:rotary_dim]

    cos = cos.unsqueeze(1)  # [num_tokens, 1, half]
    sin = sin.unsqueeze(1)

    for x_3d, out_3d in [(q_3d, q_out_3d), (k_3d, k_out_3d)]:
        if is_neox:
            x1 = x_3d[..., :half].float()
            x2 = x_3d[..., half:rotary_dim].float()
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            rotated = torch.cat([o1, o2], dim=-1).to(x_3d.dtype)
        else:
            x_rot = x_3d[..., :rotary_dim].reshape(*x_3d.shape[:-1], half, 2)
            x1 = x_rot[..., 0].float()
            x2 = x_rot[..., 1].float()
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            rotated = torch.stack([o1, o2], dim=-1).reshape(*x_3d.shape[:-1], rotary_dim).to(x_3d.dtype)

        if rotary_dim < head_size:
            out_3d.copy_(torch.cat([rotated, x_3d[..., rotary_dim:]], dim=-1))
        else:
            out_3d.copy_(rotated)
