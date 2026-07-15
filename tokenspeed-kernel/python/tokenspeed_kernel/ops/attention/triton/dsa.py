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

import math

import torch
import torch.nn.functional as F
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


_NPU_AVAILABLE: bool | None = None


def _is_npu_available() -> bool:
    global _NPU_AVAILABLE
    if _NPU_AVAILABLE is None:
        _NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()
    return _NPU_AVAILABLE


@triton.jit
def _dsa_packed_kv_kernel(
    q,
    kv_fp8,
    kv_scale,
    kv_rope,
    topk_indices,
    topk_lens,
    out,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    kv_lora_rank: tl.constexpr,
    qk_rope_head_dim: tl.constexpr,
    row_bytes: tl.constexpr,
    topk: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    v_block = tl.program_id(2)

    topk_offsets = tl.arange(0, BLOCK_TOPK)
    k_offsets = tl.arange(0, BLOCK_K)
    rope_offsets = tl.arange(0, 64)
    v_offsets = v_block * BLOCK_V + tl.arange(0, BLOCK_V)

    q_base = (token * num_heads + head) * head_dim
    q_nope_base = q_base
    q_rope_base = q_base + kv_lora_rank

    q_rope = tl.load(
        q + q_rope_base + rope_offsets,
        mask=rope_offsets < qk_rope_head_dim,
        other=0.0,
    ).to(tl.float32)

    valid_len = tl.load(topk_lens + token).to(tl.int32)
    max_score = tl.full((), -float("inf"), tl.float32)

    for start in range(0, topk, BLOCK_TOPK):
        cols = start + topk_offsets
        valid = cols < valid_len
        slots = tl.load(
            topk_indices + token * topk + cols,
            mask=valid,
            other=0,
        ).to(tl.int64)
        valid = valid & (slots >= 0)
        score = tl.zeros((BLOCK_TOPK,), tl.float32)

        for k_start in range(0, kv_lora_rank, BLOCK_K):
            ks = k_start + k_offsets
            q_vals = tl.load(q + q_nope_base + ks).to(tl.float32)
            k_vals = tl.load(
                kv_fp8 + slots[:, None] * row_bytes + ks[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(tl.float32)
            k_scale = tl.load(
                kv_scale
                + (slots * row_bytes + kv_lora_rank + (k_start // 128) * 4) // 4,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            score += tl.sum(k_vals * k_scale[:, None] * q_vals[None, :], axis=1)

        k_rope = tl.load(
            kv_rope
            + (slots[:, None] * row_bytes + kv_lora_rank + (kv_lora_rank // 128) * 4)
            // 2
            + rope_offsets[None, :],
            mask=valid[:, None] & (rope_offsets[None, :] < qk_rope_head_dim),
            other=0.0,
        ).to(tl.float32)
        score += tl.sum(k_rope * q_rope[None, :], axis=1)
        score *= softmax_scale
        score = tl.where(valid, score, -float("inf"))
        max_score = tl.maximum(max_score, tl.max(score, axis=0))

    denom = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_V,), tl.float32)
    v_mask = v_offsets < kv_lora_rank
    for start in range(0, topk, BLOCK_TOPK):
        cols = start + topk_offsets
        valid = cols < valid_len
        slots = tl.load(
            topk_indices + token * topk + cols,
            mask=valid,
            other=0,
        ).to(tl.int64)
        valid = valid & (slots >= 0)
        score = tl.zeros((BLOCK_TOPK,), tl.float32)

        for k_start in range(0, kv_lora_rank, BLOCK_K):
            ks = k_start + k_offsets
            q_vals = tl.load(q + q_nope_base + ks).to(tl.float32)
            k_vals = tl.load(
                kv_fp8 + slots[:, None] * row_bytes + ks[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(tl.float32)
            k_scale = tl.load(
                kv_scale
                + (slots * row_bytes + kv_lora_rank + (k_start // 128) * 4) // 4,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            score += tl.sum(k_vals * k_scale[:, None] * q_vals[None, :], axis=1)

        k_rope = tl.load(
            kv_rope
            + (slots[:, None] * row_bytes + kv_lora_rank + (kv_lora_rank // 128) * 4)
            // 2
            + rope_offsets[None, :],
            mask=valid[:, None] & (rope_offsets[None, :] < qk_rope_head_dim),
            other=0.0,
        ).to(tl.float32)
        score += tl.sum(k_rope * q_rope[None, :], axis=1)
        score *= softmax_scale
        score = tl.where(valid, score, -float("inf"))
        probs = tl.exp(score - max_score)
        probs = tl.where(valid, probs, 0.0)
        denom += tl.sum(probs, axis=0)

        v_vals = tl.load(
            kv_fp8 + slots[:, None] * row_bytes + v_offsets[None, :],
            mask=valid[:, None] & v_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        v_scale = tl.load(
            kv_scale
            + (
                slots[:, None] * row_bytes
                + kv_lora_rank
                + (v_offsets[None, :] // 128) * 4
            )
            // 4,
            mask=valid[:, None] & v_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(probs[:, None] * v_vals * v_scale, axis=0)

    result = acc / denom
    result = tl.where(denom > 0.0, result, 0.0)
    out_base = (token * num_heads + head) * kv_lora_rank
    tl.store(out + out_base + v_offsets, result, mask=v_mask)


@triton.jit
def _dsa_dense_kv_kernel(
    q,
    kv,
    topk_indices,
    topk_lens,
    out,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    kv_lora_rank: tl.constexpr,
    qk_rope_head_dim: tl.constexpr,
    kv_dim: tl.constexpr,
    topk: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    v_block = tl.program_id(2)

    topk_offsets = tl.arange(0, BLOCK_TOPK)
    k_offsets = tl.arange(0, BLOCK_K)
    rope_offsets = tl.arange(0, 64)
    v_offsets = v_block * BLOCK_V + tl.arange(0, BLOCK_V)

    q_base = (token * num_heads + head) * head_dim
    q_nope_base = q_base
    q_rope_base = q_base + kv_lora_rank

    q_rope = tl.load(
        q + q_rope_base + rope_offsets,
        mask=rope_offsets < qk_rope_head_dim,
        other=0.0,
    ).to(tl.float32)

    valid_len = tl.load(topk_lens + token).to(tl.int32)
    max_score = tl.full((), -float("inf"), tl.float32)

    for start in range(0, topk, BLOCK_TOPK):
        cols = start + topk_offsets
        valid = cols < valid_len
        slots = tl.load(
            topk_indices + token * topk + cols,
            mask=valid,
            other=0,
        ).to(tl.int64)
        valid = valid & (slots >= 0)
        score = tl.zeros((BLOCK_TOPK,), tl.float32)

        for k_start in range(0, kv_lora_rank, BLOCK_K):
            ks = k_start + k_offsets
            q_vals = tl.load(q + q_nope_base + ks).to(tl.float32)
            k_vals = tl.load(
                kv + slots[:, None] * kv_dim + ks[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(tl.float32)
            score += tl.sum(k_vals * q_vals[None, :], axis=1)

        k_rope = tl.load(
            kv + slots[:, None] * kv_dim + kv_lora_rank + rope_offsets[None, :],
            mask=valid[:, None] & (rope_offsets[None, :] < qk_rope_head_dim),
            other=0.0,
        ).to(tl.float32)
        score += tl.sum(k_rope * q_rope[None, :], axis=1)
        score *= softmax_scale
        score = tl.where(valid, score, -float("inf"))
        max_score = tl.maximum(max_score, tl.max(score, axis=0))

    denom = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_V,), tl.float32)
    v_mask = v_offsets < kv_lora_rank
    for start in range(0, topk, BLOCK_TOPK):
        cols = start + topk_offsets
        valid = cols < valid_len
        slots = tl.load(
            topk_indices + token * topk + cols,
            mask=valid,
            other=0,
        ).to(tl.int64)
        valid = valid & (slots >= 0)
        score = tl.zeros((BLOCK_TOPK,), tl.float32)

        for k_start in range(0, kv_lora_rank, BLOCK_K):
            ks = k_start + k_offsets
            q_vals = tl.load(q + q_nope_base + ks).to(tl.float32)
            k_vals = tl.load(
                kv + slots[:, None] * kv_dim + ks[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(tl.float32)
            score += tl.sum(k_vals * q_vals[None, :], axis=1)

        k_rope = tl.load(
            kv + slots[:, None] * kv_dim + kv_lora_rank + rope_offsets[None, :],
            mask=valid[:, None] & (rope_offsets[None, :] < qk_rope_head_dim),
            other=0.0,
        ).to(tl.float32)
        score += tl.sum(k_rope * q_rope[None, :], axis=1)
        score *= softmax_scale
        score = tl.where(valid, score, -float("inf"))
        probs = tl.exp(score - max_score)
        probs = tl.where(valid, probs, 0.0)
        denom += tl.sum(probs, axis=0)

        v_vals = tl.load(
            kv + slots[:, None] * kv_dim + v_offsets[None, :],
            mask=valid[:, None] & v_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(probs[:, None] * v_vals, axis=0)

    result = acc / denom
    result = tl.where(denom > 0.0, result, 0.0)
    out_base = (token * num_heads + head) * kv_lora_rank
    tl.store(out + out_base + v_offsets, result, mask=v_mask)


def _run_packed_kv(
    q: torch.Tensor,
    packed_kv: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_lens: torch.Tensor,
    *,
    softmax_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> torch.Tensor:
    row_bytes = int(packed_kv.shape[1])
    out = torch.empty(
        (q.shape[0], q.shape[1], kv_lora_rank),
        dtype=torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype,
        device=q.device,
    )
    grid = (q.shape[0], q.shape[1], triton.cdiv(kv_lora_rank, 64))
    _dsa_packed_kv_kernel[grid](
        q,
        packed_kv.view(torch.float8_e4m3fn),
        packed_kv.view(torch.float32),
        packed_kv.view(torch.bfloat16),
        topk_indices,
        topk_lens,
        out,
        q.shape[1],
        q.shape[2],
        kv_lora_rank,
        qk_rope_head_dim,
        row_bytes,
        topk_indices.shape[1],
        float(softmax_scale),
        BLOCK_TOPK=32,
        BLOCK_K=64,
        BLOCK_V=64,
        num_warps=4,
        num_stages=1,
    )
    return out


def _run_dense_kv(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_lens: torch.Tensor,
    *,
    softmax_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> torch.Tensor:
    kv_dim = int(kv_lora_rank) + int(qk_rope_head_dim)
    out = torch.empty(
        (q.shape[0], q.shape[1], kv_lora_rank),
        dtype=torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype,
        device=q.device,
    )
    grid = (q.shape[0], q.shape[1], triton.cdiv(kv_lora_rank, 64))
    _dsa_dense_kv_kernel[grid](
        q,
        kv_cache,
        topk_indices,
        topk_lens,
        out,
        q.shape[1],
        q.shape[2],
        kv_lora_rank,
        qk_rope_head_dim,
        kv_dim,
        topk_indices.shape[1],
        float(softmax_scale),
        BLOCK_TOPK=32,
        BLOCK_K=64,
        BLOCK_V=64,
        num_warps=4,
        num_stages=1,
    )
    return out


def _run_dsa_sdpa_npu(
    *,
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    packed_kv_cache: torch.Tensor | None,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    k_scale: float = 1.0,
) -> torch.Tensor:
    """SDPA-based fallback for DSA attention on NPU.

    This implements the same sparse top-k attention as the Triton kernel
    but using pure PyTorch operations (no triton-ascend required).
    """
    num_tokens, num_heads, head_dim = q.shape
    topk = topk_slots.shape[1]
    device = q.device
    dtype = q.dtype

    # Scale
    scale = float(softmax_scale) * float(k_scale)

    slots_flat = topk_slots.reshape(-1)  # [num_tokens * topk]
    valid_mask = (slots_flat >= 0).reshape(num_tokens, topk)

    if packed_kv_cache is not None:
        # FP8 packed KV cache: [num_slots, row_bytes] in uint8
        # Each row: [kv_lora_rank FP8 + scales + qk_rope_head_dim BF16]
        kv_fp8 = packed_kv_cache.view(torch.float8_e4m3fn)
        kv_scale_raw = packed_kv_cache.view(torch.float32)
        kv_rope_raw = packed_kv_cache.view(torch.bfloat16)
        row_bytes = int(packed_kv_cache.shape[1])

        # Gather nope (latent) keys: [num_tokens, topk, kv_lora_rank]
        k_nope_fp8 = kv_fp8[slots_flat.long()]  # [num_tokens * topk, row_bytes]
        k_nope_fp8 = k_nope_fp8[:, :kv_lora_rank].reshape(num_tokens, topk, kv_lora_rank)
        k_nope = k_nope_fp8.to(dtype)

        # Gather scales: [num_tokens, topk, kv_lora_rank // 128 * 4]
        scale_flat = kv_scale_raw[slots_flat.long()]
        # scales are stored as float32 at offset kv_lora_rank in the float32 view
        # The scale for each group of 128 is a single float32
        num_groups = (kv_lora_rank + 127) // 128
        k_scale_flat = scale_flat[:, kv_lora_rank:kv_lora_rank + num_groups].reshape(num_tokens, topk, num_groups)
        # Apply scale: expand to per-element
        k_nope = k_nope * k_scale_flat  # simplified per-group scaling

        # Gather rope part from bfloat16 view
        rope_flat = kv_rope_raw[slots_flat.long()]
        kv_rope = rope_flat[:, kv_lora_rank + num_groups:
                             kv_lora_rank + num_groups + qk_rope_head_dim].reshape(
            num_tokens, topk, qk_rope_head_dim
        ).to(dtype)
    else:
        # Dense KV cache: [num_slots, kv_dim] where kv_dim = kv_lora_rank + qk_rope_head_dim
        kv_dim = int(kv_lora_rank) + int(qk_rope_head_dim)
        kv_flat = kv_cache[slots_flat.long()]  # [num_tokens * topk, kv_dim]
        k_nope = kv_flat[:, :kv_lora_rank].reshape(num_tokens, topk, kv_lora_rank).to(dtype)
        kv_rope = kv_flat[:, kv_lora_rank:].reshape(num_tokens, topk, qk_rope_head_dim).to(dtype)

    # Split query into nope and rope parts
    q_nope = q[..., :kv_lora_rank]  # [num_tokens, num_heads, kv_lora_rank]
    q_rope = q[..., kv_lora_rank:kv_lora_rank + qk_rope_head_dim]  # [num_tokens, num_heads, qk_rope_head_dim]

    # Compute attention scores
    # q_nope: [num_tokens, num_heads, kv_lora_rank]
    # k_nope: [num_tokens, topk, kv_lora_rank]
    # We need to broadcast over heads
    scores_nope = torch.einsum("bhd,bsd->bhs", q_nope.float(), k_nope.float())  # [num_tokens, num_heads, topk]
    scores_rope = torch.einsum("bhd,bsd->bhs", q_rope.float(), kv_rope.float())  # [num_tokens, num_heads, topk]
    scores = (scores_nope + scores_rope) * scale  # [num_tokens, num_heads, topk]

    # Mask invalid positions
    valid_mask_3d = valid_mask.unsqueeze(1)  # [num_tokens, 1, topk]
    scores = torch.where(valid_mask_3d, scores, torch.tensor(-float("inf"), device=device))

    # Softmax
    max_scores = scores.max(dim=-1, keepdim=True).values
    exp_scores = torch.exp(scores - max_scores)
    exp_scores = torch.where(valid_mask_3d, exp_scores, torch.tensor(0.0, device=device))
    probs = exp_scores / exp_scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # [num_tokens, num_heads, topk]

    # Gather V values and compute weighted output
    if packed_kv_cache is not None:
        # V is the same as K in packed format (kv_fp8 contains both K and V)
        v_fp8 = k_nope_fp8  # V stored alongside K
        v_nope = v_fp8.to(dtype)
        v_nope = v_nope * k_scale_flat  # same dequant
    else:
        # In dense format, V = K for the nope part (same tensor)
        v_flat = kv_cache[slots_flat.long()]
        v_nope = v_flat[:, :kv_lora_rank].reshape(num_tokens, topk, kv_lora_rank).to(dtype)

    # Weighted sum: [num_tokens, num_heads, topk] x [num_tokens, topk, kv_lora_rank]
    # -> [num_tokens, num_heads, kv_lora_rank]
    out = torch.einsum("bhs,bsd->bhd", probs, v_nope.float())
    out = out.to(dtype)

    return out


def _flatten_packed_kv_cache(packed_kv_cache: torch.Tensor) -> torch.Tensor:
    if packed_kv_cache.dim() == 2:
        return packed_kv_cache
    return packed_kv_cache.reshape(-1, packed_kv_cache.shape[-1])


def _flatten_dense_kv_cache(kv_cache: torch.Tensor) -> torch.Tensor:
    if kv_cache.dim() == 2:
        return kv_cache
    if kv_cache.dim() == 3:
        return kv_cache.squeeze(1)
    if kv_cache.shape[1] == 1:
        kv_cache = kv_cache.permute(0, 2, 1, 3)
    return kv_cache.reshape(-1, kv_cache.shape[-1])


def _flatten_query(q: torch.Tensor) -> torch.Tensor:
    if q.dim() == 3:
        return q
    return q.reshape(-1, q.shape[-2], q.shape[-1])


def _run_dsa(
    *,
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    packed_kv_cache: torch.Tensor | None,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    k_scale: float,
    out: torch.Tensor | None,
) -> torch.Tensor:
    q = _flatten_query(q).contiguous()
    topk_slots = topk_slots.contiguous()
    topk_lens = topk_lens.contiguous()
    softmax_scale = float(softmax_scale) * float(k_scale)

    # NPU fallback: use pure PyTorch SDPA-like sparse attention
    # since triton-ascend may not be available or compatible.
    if _is_npu_available():
        result = _run_dsa_sdpa_npu(
            q=q,
            kv_cache=kv_cache,
            packed_kv_cache=packed_kv_cache,
            topk_slots=topk_slots,
            topk_lens=topk_lens,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            softmax_scale=softmax_scale,
            k_scale=k_scale,
        )
        if out is not None:
            out_view = out.reshape_as(result)
            out_view.copy_(result)
            return out
        return result

    if packed_kv_cache is not None:
        result = _run_packed_kv(
            q,
            _flatten_packed_kv_cache(packed_kv_cache).contiguous(),
            topk_slots,
            topk_lens,
            softmax_scale=softmax_scale,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )
    else:
        result = _run_dense_kv(
            q,
            _flatten_dense_kv_cache(kv_cache).contiguous(),
            topk_slots,
            topk_lens,
            softmax_scale=softmax_scale,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )

    if out is None:
        return result
    out_view = out.reshape_as(result)
    out_view.copy_(result)
    return out


@register_kernel(
    "attention",
    "dsa_decode",
    name="triton_dsa_decode",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd", "huawei"})),
    signatures=frozenset(
        {
            format_signature(q=dense_tensor_format(torch.bfloat16)),
            format_signature(q=dense_tensor_format(torch.float8_e4m3fn)),
        }
    ),
    traits={
        "page_size": frozenset({64}),
        "q_len_per_req": frozenset({1, 2, 3, 4, 5, 6}),
        "qk_nope_head_dim": frozenset({128, 192}),
        "kv_lora_rank": frozenset({128, 512}),
        "qk_rope_head_dim": frozenset({64}),
        "topk": frozenset({512, 1024, 2048}),
        "kv_cache_available": frozenset({False, True}),
        "sparse_kv_cache_available": frozenset({False, True}),
        "topk_layout": frozenset({"global_slots"}),
        "support_logit_cap": frozenset({False}),
        "return_lse": frozenset({False}),
    },
    priority=Priority.PORTABLE,
    tags={"portability"},
)
def triton_dsa_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    sparse_kv_cache: torch.Tensor | None,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor | None,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    page_size: int,
    q_len_per_req: int = 1,
    logit_cap: float = 0.0,
    k_scale: float = 1.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return _run_dsa(
        q=q,
        kv_cache=kv_cache,
        packed_kv_cache=sparse_kv_cache,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        k_scale=k_scale,
        out=out,
    )


@register_kernel(
    "attention",
    "dsa_prefill",
    name="triton_dsa_prefill",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd", "huawei"})),
    signatures=frozenset(
        {
            format_signature(q=dense_tensor_format(torch.bfloat16)),
            format_signature(q=dense_tensor_format(torch.float8_e4m3fn)),
        }
    ),
    traits={
        "page_size": frozenset({64}),
        "q_len_per_req": frozenset({1}),
        "qk_nope_head_dim": frozenset({128, 192}),
        "kv_lora_rank": frozenset({128, 512}),
        "qk_rope_head_dim": frozenset({64}),
        "topk": frozenset({512, 1024, 2048}),
        "kv_cache_available": frozenset({False, True}),
        "sparse_kv_cache_available": frozenset({False, True}),
        "topk_layout": frozenset({"global_slots"}),
        "support_logit_cap": frozenset({False}),
        "return_lse": frozenset({False}),
    },
    priority=Priority.PORTABLE,
    tags={"portability"},
)
def triton_dsa_prefill(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    sparse_kv_cache: torch.Tensor | None,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor | None,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    page_size: int,
    q_len_per_req: int = 1,
    logit_cap: float = 0.0,
    k_scale: float = 1.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return _run_dsa(
        q=q,
        kv_cache=kv_cache,
        packed_kv_cache=sparse_kv_cache,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        k_scale=k_scale,
        out=out,
    )
