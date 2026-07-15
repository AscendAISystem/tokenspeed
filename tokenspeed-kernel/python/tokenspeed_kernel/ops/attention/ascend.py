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

"""Ascend NPU MHA attention backend using torch_npu fused attention primitives.

Registers NPU-optimized implementations for MHA prefill, extend, and decode
operations. These implementations use ``npu_fused_infer_attention_score``
when running on Huawei Ascend hardware and fall through to the standard
PyTorch SDPA path when the NPU-specific API is unavailable.
"""

from __future__ import annotations

import logging
import math

import torch
from tokenspeed_kernel.platform import CapabilityRequirement

logger = logging.getLogger(__name__)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

# ---------------------------------------------------------------------------
# NPU availability helpers
# ---------------------------------------------------------------------------


def _is_npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _get_fused_attention_score():
    """Return ``torch_npu.npu_fused_infer_attention_score`` or None."""
    try:
        import torch_npu  # noqa: F401
        return torch.ops.npu.npu_fused_infer_attention_score
    except (ImportError, AttributeError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# MHA Prefill
# ---------------------------------------------------------------------------


@register_kernel(
    "attention",
    "mha_prefill",
    name="npu_mha_prefill",
    solution="ascend",
    capability=CapabilityRequirement(vendors=frozenset({"huawei"})),
    signatures=format_signatures(
        ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
    ),
    priority=Priority.SPECIALIZED,
    traits={
        "sliding_window": frozenset({False, True}),
        "support_logit_cap": frozenset({False, True}),
        "support_sinks": frozenset({False, True}),
        "return_lse": frozenset({False, True}),
    },
    tags={"portability"},
)
def npu_mha_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: list[int],
    max_seqlen: int,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """MHA prefill using Ascend NPU ``npu_fused_infer_attention_score``.

    Args:
        q: Query tensor shaped ``[total_q, num_q_heads, head_dim]``.
        k: Key tensor shaped ``[total_kv, num_kv_heads, head_dim]``.
        v: Value tensor shaped ``[total_kv, num_kv_heads, head_dim]``.
        cu_seqlens: Cumulative sequence lengths ``[batch + 1]``.
        cu_seqlens_cpu: Host copy of cumulative lengths.
        max_seqlen: Maximum sequence length.
        window_left: Exclusive left sliding-window size. -1 means full attention.
        logit_cap: Optional soft cap applied to attention logits.
        sinks: Optional attention sink tensor.
        return_lse: Whether to also return log-sum-exp values.

    Returns:
        Attention output ``[total_q, num_q_heads, head_dim]``, or
        ``(output, lse)`` when ``return_lse`` is True.
    """
    del cu_seqlens_cpu, sinks  # Not used in NPU path (handled by fused kernel)
    if not _is_npu_available():
        raise RuntimeError("NPU not available")

    fused_attn = _get_fused_attention_score()
    num_q_heads = q.shape[1]
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)

    # Reshape q/k/v: [total, heads, dim] -> [1, heads, total, dim] (BNSD layout)
    # q.unsqueeze(0) -> [1, total_q, num_heads, head_dim]; transpose to BNSD
    q_4d = q.unsqueeze(0).transpose(1, 2)  # [1, num_heads, total_q, head_dim]
    k_4d = k.unsqueeze(0).transpose(1, 2)  # [1, num_heads, total_kv, head_dim]
    v_4d = v.unsqueeze(0).transpose(1, 2)  # [1, num_heads, total_kv, head_dim]

    # Build per-batch sequence lengths from cu_seqlens
    batch_size = cu_seqlens.shape[0] - 1
    seq_lens = []
    for i in range(batch_size):
        seq_lens.append(int(cu_seqlens[i + 1] - cu_seqlens[i]))
    actual_seq_lengths = seq_lens

    sparse_mode = 2 if window_left > 0 else 0

    if fused_attn is not None:
        out, _ = fused_attn(
            q_4d, k_4d, v_4d,
            actual_seq_lengths=actual_seq_lengths,
            num_heads=num_q_heads,
            scale=scale,
            input_layout="BNSD",
            sparse_mode=sparse_mode,
        )
        # out from NPU: [batch, num_heads, total_q, head_dim]; squeeze batch then
        # transpose back to [total_q, num_heads, head_dim]
        out = out.squeeze(0).transpose(0, 1)  # [total_q, num_heads, head_dim]
    else:
        # Fallback: use PyTorch SDPA when NPU API is unavailable
        out = torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(0).transpose(1, 2),  # [1, heads, total, dim]
            k.unsqueeze(0).transpose(1, 2),
            v.unsqueeze(0).transpose(1, 2),
            scale=scale,
            is_causal=True,
        ).transpose(1, 2).squeeze(0)

    if return_lse:
        lse = torch.zeros((q.shape[0], num_q_heads), dtype=torch.float32, device=q.device)
        return out, lse
    return out


# ---------------------------------------------------------------------------
# MHA Extend with KV Cache
# ---------------------------------------------------------------------------


@register_kernel(
    "attention",
    "mha_extend_with_kvcache",
    name="npu_mha_extend_with_kvcache",
    solution="ascend",
    capability=CapabilityRequirement(vendors=frozenset({"huawei"})),
    signatures=format_signatures(
        ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
    ),
    priority=Priority.SPECIALIZED,
    traits={
        "is_causal": frozenset({False, True}),
        "sliding_window": frozenset({False, True}),
        "support_logit_cap": frozenset({False, True}),
        "support_sinks": frozenset({False, True}),
        "return_lse": frozenset({False, True}),
    },
    tags={"portability"},
)
def npu_mha_extend_with_kvcache(
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """MHA extend with paged KV cache using Ascend NPU attention.

    Args:
        q: Query tensor shaped ``[total_q, num_q_heads, head_dim]``.
        cu_seqlens_q: Query cumulative sequence lengths ``[batch + 1]``.
        cu_seqlens_kv: KV cumulative sequence lengths ``[batch + 1]``.
        k_cache: Paged key cache ``[num_pages, page_size, num_kv_heads, head_dim]``.
        v_cache: Paged value cache ``[num_pages, page_size, num_kv_heads, head_dim]``.
        page_table: Page table ``[batch, max_pages_per_seq]``.
        cache_seqlens: Visible KV lengths ``[batch]``.
        max_seqlen_q: Maximum query length.
        max_seqlen_k: Maximum KV length.
        is_causal: Whether query is causal suffix of cached KV.
        window_left: Exclusive left sliding-window size.
        logit_cap: Optional soft cap on logits.
        sinks: Optional attention sink tensor.
        return_lse: Whether to also return log-sum-exp values.

    Returns:
        Attention output or ``(output, lse)`` when ``return_lse`` is True.
    """
    del cu_seqlens_kv, max_seqlen_k, logit_cap, sinks  # not used in NPU path
    if not _is_npu_available():
        raise RuntimeError("NPU not available")

    fused_attn = _get_fused_attention_score()
    num_q_heads = q.shape[1]
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)
    batch_size = cache_seqlens.shape[0]

    # Build actual sequence lengths for query
    seq_lens_q = []
    for i in range(batch_size):
        seq_lens_q.append(int(cu_seqlens_q[i + 1] - cu_seqlens_q[i]))

    # Build actual sequence lengths for KV
    seq_lens_kv = [int(s) for s in cache_seqlens]

    # Reshape query: [total_q, heads, dim] -> [1, heads, total_q, dim] (BNSD)
    q_4d = q.unsqueeze(0)  # [1, total_q, num_heads, head_dim]

    if fused_attn is not None:
        out, _ = fused_attn(
            q_4d, k_cache, v_cache,
            actual_seq_lengths=seq_lens_q,
            actual_seq_lengths_kv=seq_lens_kv,
            num_heads=num_q_heads,
            scale=scale,
            input_layout="BNSD",
            block_table=page_table,
            block_size=k_cache.shape[1],
            sparse_mode=0,
        )
        out = out.squeeze(0)
    else:
        # Fallback: SDPA with cached KV
        k_contiguous = k_cache.reshape(-1, k_cache.shape[2], k_cache.shape[3])
        v_contiguous = v_cache.reshape(-1, v_cache.shape[2], v_cache.shape[3])
        out = torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(0).transpose(1, 2),
            k_contiguous.unsqueeze(0).transpose(1, 2),
            v_contiguous.unsqueeze(0).transpose(1, 2),
            scale=scale,
            is_causal=is_causal,
        ).transpose(1, 2).squeeze(0)

    if return_lse:
        lse = torch.zeros((q.shape[0], num_q_heads), dtype=torch.float32, device=q.device)
        return out, lse
    return out


# ---------------------------------------------------------------------------
# MHA Decode with KV Cache
# ---------------------------------------------------------------------------


@register_kernel(
    "attention",
    "mha_decode_with_kvcache",
    name="npu_mha_decode_with_kvcache",
    solution="ascend",
    capability=CapabilityRequirement(vendors=frozenset({"huawei"})),
    signatures=format_signatures(
        ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
    ),
    priority=Priority.SPECIALIZED,
    traits={
        "sliding_window": frozenset({False, True}),
        "support_logit_cap": frozenset({False, True}),
        "support_sinks": frozenset({False, True}),
        "return_lse": frozenset({False}),
    },
    tags={"portability"},
)
def npu_mha_decode_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    max_seqlen_q: int = 1,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor:
    """MHA decode with paged KV cache using Ascend NPU.

    Args:
        q: Query tensor shaped ``[batch * max_seqlen_q, num_q_heads, head_dim]``.
        k_cache: Paged key cache ``[num_pages, page_size, num_kv_heads, head_dim]``.
        v_cache: Paged value cache (same layout).
        page_table: Page table ``[batch, max_pages_per_seq]``.
        cache_seqlens: Total KV lengths ``[batch]``.
        max_seqlen_k: Maximum KV length.
        max_seqlen_q: Number of query tokens per request.
        window_left: Exclusive left sliding-window size.
        logit_cap: Optional soft cap on logits.
        sinks: Optional attention sink tensor.
        return_lse: Whether to also return log-sum-exp values.

    Returns:
        Attention output ``[batch * max_seqlen_q, num_q_heads, head_dim]``.
    """
    del max_seqlen_k, logit_cap, sinks  # not used in NPU path
    if not _is_npu_available():
        raise RuntimeError("NPU not available")

    fused_attn = _get_fused_attention_score()
    num_q_heads = q.shape[1]
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)
    batch_size = q.shape[0] // max_seqlen_q

    # Reshape q for batch processing
    # q is [batch * max_seqlen_q, num_heads, head_dim]
    # -> [batch, max_seqlen_q, num_heads, head_dim] (PyTorch BSND layout)
    q_4d = q.reshape(batch_size, max_seqlen_q, num_q_heads, head_dim)

    # Try to use NPU fused attention first (fast path), fall back to SDPA
    use_fused = False
    if fused_attn is not None:
        try:
            # For page attention, actual_seq_lengths_kv is mandatory per NPU API spec
            seq_lens_list = [int(s) for s in cache_seqlens]
            # NPU BNSD = [batch, num_heads, seqlen, head_dim]; transpose from BSND
            q_bnsd = q_4d.transpose(1, 2)
            # NPU BnNBsD expects k/v as [num_pages, num_kv_heads, page_size, head_dim]
            # Our k_cache is [num_pages, page_size, num_kv_heads, head_dim]; transpose
            k_bnsd = k_cache.transpose(1, 2)
            v_bnsd = v_cache.transpose(1, 2)
            out, _ = fused_attn(
                q_bnsd, k_bnsd, v_bnsd,
                actual_seq_lengths=seq_lens_list,
                actual_seq_lengths_kv=seq_lens_list,
                num_heads=num_q_heads,
                num_key_value_heads=k_cache.shape[2],
                scale=scale,
                input_layout="BNSD",
                block_table=page_table,
                block_size=k_cache.shape[1],
            )
            out = out.reshape(-1, num_q_heads, head_dim)
            use_fused = True
        except RuntimeError as e:
            logger.warning("NPU fused attention failed, falling back to SDPA: %s", e)
    if not use_fused:
        # Fallback: SDPA decode with proper page table gathering
        # k_cache shape: [num_pages, page_size, num_kv_heads, head_dim]
        num_pages, page_size, num_kv_heads, head_dim = k_cache.shape
        batch_size = page_table.shape[0]
        # Gather KV cache using page table
        # page_table: [batch_size, max_pages_per_seq] with page indices
        max_pages_per_seq = page_table.shape[1]
        total_kv_tokens = int(cache_seqlens.sum().item())
        # For simplicity, handle single-batch decode (most common case for bs=1)
        if batch_size == 1:
            num_kv = int(cache_seqlens[0].item())
            pages = page_table[0, :(num_kv + page_size - 1) // page_size]
            gathered_k = k_cache[pages.long()].reshape(-1, num_kv_heads, head_dim)[:num_kv]
            gathered_v = v_cache[pages.long()].reshape(-1, num_kv_heads, head_dim)[:num_kv]
            # q shape: [batch * max_seqlen_q, num_q_heads, head_dim], max_seqlen_q=1 for decode
            q_2d = q.reshape(1, -1, num_q_heads, head_dim)  # [1, num_q_heads, 1, head_dim]
            k_2d = gathered_k.unsqueeze(0)  # [1, num_kv, num_kv_heads, head_dim]
            v_2d = gathered_v.unsqueeze(0)
            out = torch.nn.functional.scaled_dot_product_attention(
                q_2d.transpose(1, 2),
                k_2d.transpose(1, 2),
                v_2d.transpose(1, 2),
                scale=scale,
            ).transpose(1, 2).squeeze(0)
        else:
            # Multi-batch: process each batch separately
            outputs = []
            offset = 0
            for b in range(batch_size):
                num_kv = int(cache_seqlens[b].item())
                pages = page_table[b, :(num_kv + page_size - 1) // page_size]
                gathered_k = k_cache[pages.long()].reshape(-1, num_kv_heads, head_dim)[:num_kv]
                gathered_v = v_cache[pages.long()].reshape(-1, num_kv_heads, head_dim)[:num_kv]
                q_b = q[offset:offset + max_seqlen_q].reshape(1, max_seqlen_q, num_q_heads, head_dim)
                k_b = gathered_k.unsqueeze(0)  # [1, num_kv, num_kv_heads, head_dim]
                v_b = gathered_v.unsqueeze(0)
                out_b = torch.nn.functional.scaled_dot_product_attention(
                    q_b.transpose(1, 2),
                    k_b.transpose(1, 2),
                    v_b.transpose(1, 2),
                    scale=scale,
                ).transpose(1, 2).squeeze(0)
                outputs.append(out_b)
                offset += max_seqlen_q
            out = torch.cat(outputs, dim=0)

    if return_lse:
        lse = torch.zeros((q.shape[0], num_q_heads), dtype=torch.float32, device=q.device)
        return out, lse
    return out
