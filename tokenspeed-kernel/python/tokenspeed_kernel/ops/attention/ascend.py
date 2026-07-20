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
# Graph task update support (for NPU graph capture with fused attention)
# ---------------------------------------------------------------------------
# Stores handles + tensor references created during graph capture so that
# ``cuda_graph_wrapper.py`` can hot-update parameters before each replay.
#
# Lifecycle:
#   capture → npu_mha_decode_with_kvcache() appends handle entries
#          → cuda_graph_wrapper retrieves them via get_decode_attn_capture_handles()
#   replay  → cuda_graph_wrapper calls update_decode_attn_graph_params()
#             before graph.replay() with current sequence lengths

_decode_attn_capture_handles: list[dict] = []

# ---------------------------------------------------------------------------
# Causal mask for sparse_mode=3 (CANN V3 requires explicit atten_mask)
# ---------------------------------------------------------------------------
# Pre-built 2048×2048 causal mask for sparse_mode=3 (rightDownCausal).
# CANN V3 requires an explicit atten_mask (bool/uint8) when sparse_mode != 0.
# Shape: (2048, 2048), value: True = masked out, False = allowed.
# Lower triangle + diagonal = False (allowed), upper triangle = True (masked).
# Created once per device and cached lazily.
_MAX_ATTEN_SIZE = 2048
_causal_mask_2048: dict[str, torch.Tensor] = {}


def _get_causal_mask(device: torch.device) -> torch.Tensor:
    """Return a 2048×2048 causal bool mask suitable for ``sparse_mode=3``.

    CANN ``sparse_mode=3`` (rightDownCausal) requires an explicit
    ``atten_mask`` even in TND layout on CANN 2.10.0 (V3 API).
    The mask must be bool or uint8 — float dtypes (bf16/fp16) are rejected.

    The returned mask has shape ``(2048, 2048)`` where ``True`` means
    masked out (upper triangle, j > i) and ``False`` means allowed
    (lower triangle including diagonal, j ≤ i).

    Created once per device and cached in ``_causal_mask_2048``.
    """
    global _causal_mask_2048
    key = device.type  # cache per device type (e.g. "npu")
    if key not in _causal_mask_2048:
        mask = torch.ones(
            (_MAX_ATTEN_SIZE, _MAX_ATTEN_SIZE),
            dtype=torch.bool,
            device=device,
        )
        # Upper triangle (j > i) stays True (masked);
        # lower triangle + diagonal become False (allowed)
        mask = torch.triu(mask, diagonal=1)
        _causal_mask_2048[key] = mask
    return _causal_mask_2048[key]


def get_decode_attn_capture_handles() -> list[dict]:
    """Return all stored capture handles for decode attention and clear list.

    Called by ``cuda_graph_wrapper`` after each graph capture session.
    Each dict contains:
        handle            – opaque handle from ``graph_task_group_end``
        q/k/v             – tensor references in TND layout
        num_heads,
        num_key_value_heads,
        scale,
        block_size        – static parameters from capture
        block_table       – page_table tensor reference
        sparse_mode       – sparsity mode (3=causal, 4=sliding window)
        pre_tokens        – left sliding window size (sparse_mode=4 only)
        next_tokens       – right window (always 0 for decode)
        atten_mask        – bool mask for sparse_mode != 0
        max_seqlen_q      – query tokens per batch (usually 1)
    """
    handles = _decode_attn_capture_handles[:]
    _decode_attn_capture_handles.clear()
    return handles


def update_decode_attn_graph_params(
    handles: list[dict],
    seq_lens_list: list[int],
    stream: torch.npu.Stream | None = None,
) -> None:
    """Hot-update captured decode attention parameters before graph replay.

    Must be called on the capture stream **before** ``graph.replay()``.

    TND layout: the function converts the per-batch KV lengths from
    ``seq_lens_list`` into cumulative sums needed by ``input_layout="TND"``.
    Query cumulative lengths are derived from ``max_seqlen_q`` stored in each
    handle entry.

    Args:
        handles: List of handle dicts from ``get_decode_attn_capture_handles()``.
        seq_lens_list: Current per-batch KV sequence lengths as Python
                       ``list[int]``.  Used to build cumulative KV lengths.
        stream: NPU stream for the update.  Defaults to current stream.
    """
    fused_attn = _get_fused_attention_score()
    if fused_attn is None:
        logger.warning(
            "update_decode_attn_graph_params: fused_attn unavailable, skipping"
        )
        return
    if stream is None:
        stream = torch.npu.current_stream()
    if handles:
        logger.info(
            "graph_task_update: updating %d handle(s) seq_lens=%s",
            len(handles),
            seq_lens_list,
        )
    for entry in handles:
        handle = entry["handle"]
        max_sq = entry.get("max_seqlen_q", 1)
        # Build TND cumulative q lengths from max_seqlen_q
        cum_q = []
        running = 0
        for _ in range(len(seq_lens_list)):
            running += max_sq
            cum_q.append(running)
        # Build TND cumulative KV lengths from per-batch seq_lens_list
        cum_kv = []
        running = 0
        for sl in seq_lens_list:
            running += sl
            cum_kv.append(running)

        torch.npu.graph_task_update_begin(stream, handle)
        kwargs = dict(
            actual_seq_lengths=cum_q,
            actual_seq_lengths_kv=cum_kv,
            num_heads=entry["num_heads"],
            num_key_value_heads=entry["num_key_value_heads"],
            scale=entry["scale"],
            input_layout="TND",
            block_table=entry["block_table"],
            block_size=entry["block_size"],
            sparse_mode=entry.get("sparse_mode", 3),
        )
        sparse_mode = entry.get("sparse_mode", 3)
        if sparse_mode == 4:
            kwargs["pre_tokens"] = entry.get("pre_tokens", 0)
            kwargs["next_tokens"] = entry.get("next_tokens", 0)
        if sparse_mode != 0:
            kwargs["atten_mask"] = entry["atten_mask"]
        fused_attn(
            entry["q"],
            entry["k"],
            entry["v"],
            **kwargs,
        )
        torch.npu.graph_task_update_end(stream)


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
    """MHA prefill using TND layout + ``npu_fused_infer_attention_score``.

    Uses CANN's fused attention kernel with TND (Tile-Nested-Descriptor)
    layout for efficient block-diagonal causal masking in packed sequences,
    with graceful fallback to per-batch PyTorch SDPA when the fused kernel
    is unavailable.

    .. note::

       The fused kernel natively handles GQA via separate ``num_heads``
       and ``num_key_value_heads`` parameters, so no manual KV head
       repetition is needed in the fused path.

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
    del cu_seqlens_cpu, sinks
    if not _is_npu_available():
        raise RuntimeError("NPU not available")

    fused_attn = _get_fused_attention_score()
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)
    total_q = q.shape[0]
    total_kv = k.shape[0]

    # ---- TND layout + fused attention path (when available) ----
    if fused_attn is not None:
        # TND layout: [total_seq, num_heads, head_dim] (no batch dim)
        q_tnd = q  # [total_q, num_q_heads, head_dim] — already TND shape
        k_tnd = k  # [total_kv, num_kv_heads, head_dim]
        v_tnd = v

        # cu_seqlens must be Python list[int] for actual_seq_lengths
        act_seq = cu_seqlens.tolist() if isinstance(cu_seqlens, torch.Tensor) else cu_seqlens

        if window_left > 0:
            # Sliding window: sparse_mode=4 + pre_tokens
            sparse_mode = 4
            causal_tokens = window_left
            logger.debug("npu_mha_prefill: using TND fused attention sparse_mode=4 (sliding window, pre_tokens=%s)", causal_tokens)
        else:
            # Standard causal: sparse_mode=3 requires explicit atten_mask (CANN V3 API)
            sparse_mode = 3
            causal_tokens = None
            atten_mask = _get_causal_mask(q.device)
            logger.debug("npu_mha_prefill: using TND fused attention sparse_mode=3 (causal with 2048x2048 bool atten_mask)")

        kwargs = dict(
            num_heads=num_q_heads,
            num_key_value_heads=num_kv_heads,
            scale=scale,
            input_layout="TND",
            actual_seq_lengths=act_seq,
            actual_seq_lengths_kv=act_seq,
            sparse_mode=sparse_mode,
            softmax_lse_flag=return_lse,
        )
        if sparse_mode == 4:
            kwargs["pre_tokens"] = causal_tokens
            kwargs["next_tokens"] = 0
        elif sparse_mode == 3:
            kwargs["atten_mask"] = atten_mask

        out, lse_tmp = fused_attn(q_tnd, k_tnd, v_tnd, **kwargs)
        # fused output: [total_q, num_q_heads, head_dim] — already correct shape
    else:
        # ---- SDPA per-batch fallback (correct causal masking) ----
        logger.debug("npu_mha_prefill: using per-batch SDPA fallback (causal masking)")

        act_seq = cu_seqlens.tolist() if isinstance(cu_seqlens, torch.Tensor) else cu_seqlens
        batch_size = len(act_seq) - 1

        k_sdpa = k
        v_sdpa = v
        if num_q_heads != num_kv_heads:
            n_reps = num_q_heads // num_kv_heads
            k_sdpa = k.repeat_interleave(n_reps, dim=1)
            v_sdpa = v.repeat_interleave(n_reps, dim=1)

        outputs = []
        for i in range(batch_size):
            start = act_seq[i]
            end = act_seq[i + 1]
            q_i = q[start:end].unsqueeze(0).transpose(1, 2)   # [1, H, sl, D]
            k_i = k_sdpa[start:end].unsqueeze(0).transpose(1, 2)
            v_i = v_sdpa[start:end].unsqueeze(0).transpose(1, 2)
            out_i = torch.nn.functional.scaled_dot_product_attention(
                q_i, k_i, v_i,
                scale=scale,
                is_causal=True,
            ).transpose(1, 2).squeeze(0)  # [sl, H, D]
            outputs.append(out_i)

        out = torch.cat(outputs, dim=0)

    if return_lse:
        if fused_attn is not None:
            # fused path: lse_tmp shape is [total_q, num_q_heads, 1]; squeeze to [total_q, num_q_heads]
            lse = lse_tmp.squeeze(-1)
        else:
            # fallback SDPA path: dummy lse
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
    del cu_seqlens_kv, max_seqlen_k, logit_cap, sinks
    if not _is_npu_available():
        raise RuntimeError("NPU not available")

    fused_attn = _get_fused_attention_score()
    num_q_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)
    total_q = q.shape[0]
    batch_size = cache_seqlens.shape[0]

    # TND: actual_seq_lengths must be cumulative sums without leading zero
    # (CANN V3 expects length == batch_size, last element == total tokens)
    cu_q = cu_seqlens_q.tolist() if isinstance(cu_seqlens_q, torch.Tensor) else list(cu_seqlens_q)
    # Remove leading 0 and keep cumulative: [0, s1, s1+s2] -> [s1, s1+s2]
    act_seq_q = cu_q[1:]  # length = batch_size, last = total_q
    # Build KV cumulative lengths from per-batch cache_seqlens
    kv_cum = []
    running = 0
    for i in range(batch_size):
        running += int(cache_seqlens[i])
        kv_cum.append(running)
    act_seq_kv = kv_cum  # length = batch_size, last = total_kv

    # ---- TND layout + fused attention path (when available) ----
    if fused_attn is not None:
        # TND layout: q is [total_q, num_q_heads, head_dim] — already TND
        # k_cache/v_cache: original [num_pages, page_size, num_kv_heads, head_dim]
        # CANN expects [num_pages, num_kv_heads, page_size, head_dim] for TND page attention
        q_tnd = q  # [total_q, num_q_heads, head_dim]
        k_tnd = k_cache.transpose(1, 2)  # -> [num_pages, num_kv_heads, page_size, head_dim]
        v_tnd = v_cache.transpose(1, 2)

        if window_left > 0:
            sparse_mode = 4
            # For sparse_mode=4 (sliding window), CANN also requires atten_mask.
            # Use all-False (no additional masking); shape must be [1, 1, 2048, 2048]
            atten_mask = torch.zeros(
                (1, 1, _MAX_ATTEN_SIZE, _MAX_ATTEN_SIZE),
                dtype=torch.bool,
                device=q.device,
            )
            logger.debug(
                "npu_mha_extend: using TND fused attention sparse_mode=4 "
                "(sliding window, pre_tokens=%s)", window_left
            )
        elif is_causal:
            sparse_mode = 3
            atten_mask = _get_causal_mask(q.device)
            logger.debug(
                "npu_mha_extend: using TND fused attention sparse_mode=3 "
                "(causal, is_causal=True)"
            )
        else:
            sparse_mode = 0
            atten_mask = None
            logger.debug(
                "npu_mha_extend: using TND fused attention sparse_mode=0 "
                "(non-causal)"
            )

        kwargs = dict(
            num_heads=num_q_heads,
            num_key_value_heads=num_kv_heads,
            scale=scale,
            input_layout="TND",
            actual_seq_lengths=act_seq_q,
            actual_seq_lengths_kv=act_seq_kv,
            block_table=page_table,
            block_size=k_cache.shape[1],
            sparse_mode=sparse_mode,
            softmax_lse_flag=return_lse,
        )
        if sparse_mode == 4:
            kwargs["pre_tokens"] = window_left
            kwargs["next_tokens"] = 0
        if sparse_mode != 0:
            kwargs["atten_mask"] = atten_mask

        out, lse_tmp = fused_attn(q_tnd, k_tnd, v_tnd, **kwargs)
        # output: [total_q, num_q_heads, head_dim]
    else:
        # ---- SDPA fallback ----
        logger.debug("npu_mha_extend: using SDPA fallback")
        k_contiguous = k_cache.reshape(-1, k_cache.shape[2], k_cache.shape[3])
        v_contiguous = v_cache.reshape(-1, v_cache.shape[2], v_cache.shape[3])
        if num_q_heads != num_kv_heads:
            n_reps = num_q_heads // num_kv_heads
            k_contiguous = k_contiguous.repeat_interleave(n_reps, dim=1)
            v_contiguous = v_contiguous.repeat_interleave(n_reps, dim=1)
        out = torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(0).transpose(1, 2),
            k_contiguous.unsqueeze(0).transpose(1, 2),
            v_contiguous.unsqueeze(0).transpose(1, 2),
            scale=scale,
            is_causal=is_causal,
        ).transpose(1, 2).squeeze(0)

    if return_lse:
        if fused_attn is not None:
            lse = lse_tmp.squeeze(-1)
        else:
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
    """MHA decode with paged KV cache using Ascend NPU (TND layout).

    Uses TND (Tile-Nested-Descriptor) layout for efficient batching of
    variable-length decode sequences with paged KV cache.

    The kernel natively handles GQA via separate ``num_heads`` and
    ``num_key_value_heads`` parameters, so no manual KV head repetition
    is needed.

    Args:
        q: Query tensor shaped ``[total_q, num_q_heads, head_dim]``
           where ``total_q = batch * max_seqlen_q`` — TND layout.
        k_cache: Paged key cache ``[num_pages, page_size, num_kv_heads, head_dim]``.
        v_cache: Paged value cache (same layout).
        page_table: Page table ``[batch, max_pages_per_seq]``.
        cache_seqlens: Total KV lengths ``[batch]``.
        max_seqlen_k: Maximum KV length.
        max_seqlen_q: Number of query tokens per request (usually 1 for decode).
        window_left: Exclusive left sliding-window size.  -1 means full causal.
        logit_cap: Optional soft cap on logits.
        sinks: Optional attention sink tensor.
        return_lse: Whether to also return log-sum-exp values.

    Returns:
        Attention output ``[total_q, num_q_heads, head_dim]``.
    """
    del max_seqlen_k, logit_cap, sinks  # not used in NPU path
    if not _is_npu_available():
        raise RuntimeError("NPU not available")

    fused_attn = _get_fused_attention_score()
    num_q_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)
    batch_size = q.shape[0] // max_seqlen_q
    total_q = q.shape[0]  # = batch_size * max_seqlen_q

    # Detect CUDA graph capture mode: inside torch.npu.graph(), stream sync is
    # not allowed on Ascend NPU.  We must use only basic tensor ops (matmul,
    # softmax) and avoid any .item() calls that would trigger a CPU sync.
    is_capturing = torch.npu.is_current_stream_capturing()

    # ---- TND layout: q is [total_q, num_q_heads, head_dim] ----
    # In TND, actual_seq_lengths must be cumulative sums (length=batch_size,
    # last element=total tokens).  For decode each request has max_seqlen_q
    # query tokens.
    if is_capturing:
        # ---- Graph-capture path with fused attention + graph_task_group ----
        # Use graph_task_group_begin/end to wrap npu_fused_infer_attention_score
        # so its parameters (seq_lens, block_table) can be hot-updated via
        # graph_task_update before each replay, avoiding the performance
        # penalty of manual matmul (padded KV, ~3 kernels vs 1 fused kernel).
        #
        # During capture we MUST avoid any .item() / int(tensor) calls because
        # they trigger NPU→CPU sync which is forbidden on a captured stream.
        # The cumulative seq lengths here are DUMMY placeholders; the real values
        # are passed via update_decode_attn_graph_params() before each replay.
        logger.info(
            "npu_mha_decode_with_kvcache: graph capture mode, using graph_task_group"
        )
        stream = torch.npu.current_stream()

        # TND layout: q is [total_q, num_q_heads, head_dim] — already TND
        # k_cache/v_cache: transpose [num_pages, page_size, K, D] -> [num_pages, K, page_size, D]
        q_tnd = q  # [total_q, num_q_heads, head_dim]
        k_tnd = k_cache.transpose(1, 2)  # [num_pages, num_kv_heads, page_size, head_dim]
        v_tnd = v_cache.transpose(1, 2)

        # Dummy cumulative q lengths: [max_seqlen_q, 2*max_seqlen_q, ..., total_q]
        cum_q_dummy = []
        running = 0
        for _ in range(batch_size):
            running += max_seqlen_q
            cum_q_dummy.append(running)

        # Dummy cumulative KV lengths: [1, 2, ..., batch_size]
        cum_kv_dummy = []
        running = 0
        for _ in range(batch_size):
            running += 1
            cum_kv_dummy.append(running)

        # Sparse mode: sliding window (4) or causal (3).
        # During graph capture, sparse_mode is baked into the captured graph
        # so it stays constant across replays.
        sparse_mode = 4 if window_left > 0 else 3

        # Build kwargs for fused attention (TND + page attention)
        kwargs = dict(
            num_heads=num_q_heads,
            num_key_value_heads=num_kv_heads,
            scale=scale,
            input_layout="TND",
            actual_seq_lengths=cum_q_dummy,
            actual_seq_lengths_kv=cum_kv_dummy,
            block_table=page_table,
            block_size=k_cache.shape[1],
            sparse_mode=sparse_mode,
        )
        if sparse_mode == 4:
            kwargs["pre_tokens"] = window_left
            kwargs["next_tokens"] = 0
            # All-False mask (no extra masking beyond sliding window)
            kwargs["atten_mask"] = torch.zeros(
                (1, 1, _MAX_ATTEN_SIZE, _MAX_ATTEN_SIZE),
                dtype=torch.bool,
                device=q.device,
            )
        else:
            # sparse_mode=3: causal requires explicit atten_mask (CANN V3 API)
            kwargs["atten_mask"] = _get_causal_mask(q.device)

        torch.npu.graph_task_group_begin(stream)
        out, _ = fused_attn(q_tnd, k_tnd, v_tnd, **kwargs)
        handle = torch.npu.graph_task_group_end(stream)

        # Store handle + tensor references for pre-replay parameter update.
        # The graph_task_update path uses these stored tensors directly.
        _decode_attn_capture_handles.append({
            "handle": handle,
            "q": q_tnd,
            "k": k_tnd,
            "v": v_tnd,
            "num_heads": num_q_heads,
            "num_key_value_heads": num_kv_heads,
            "scale": scale,
            "block_table": page_table,
            "block_size": k_cache.shape[1],
            "sparse_mode": sparse_mode,
            "pre_tokens": window_left if sparse_mode == 4 else None,
            "next_tokens": 0 if sparse_mode == 4 else None,
            "atten_mask": kwargs["atten_mask"],
            "max_seqlen_q": max_seqlen_q,
        })
        # TND output is already [total_q, num_q_heads, head_dim]; reshape for safety
        out = out.reshape(-1, num_q_heads, head_dim)
    else:
        # ---- Normal (non-capturing) path: try TND fused then SDPA ----
        use_fused = False
        if fused_attn is not None:
            try:
                # TND layout: q is [total_q, num_q_heads, head_dim] — use directly
                q_tnd = q
                k_tnd = k_cache.transpose(1, 2)
                v_tnd = v_cache.transpose(1, 2)

                # Build cumulative q lengths from max_seqlen_q
                cum_seq_q = []
                running = 0
                for _ in range(batch_size):
                    running += max_seqlen_q
                    cum_seq_q.append(running)

                # Build cumulative KV lengths from cache_seqlens
                cum_seq_kv = []
                running = 0
                for i in range(batch_size):
                    running += int(cache_seqlens[i])
                    cum_seq_kv.append(running)

                # Sparse mode
                if window_left > 0:
                    sparse_mode = 4
                else:
                    sparse_mode = 3

                kwargs = dict(
                    num_heads=num_q_heads,
                    num_key_value_heads=num_kv_heads,
                    scale=scale,
                    input_layout="TND",
                    actual_seq_lengths=cum_seq_q,
                    actual_seq_lengths_kv=cum_seq_kv,
                    block_table=page_table,
                    block_size=k_cache.shape[1],
                    sparse_mode=sparse_mode,
                )
                if sparse_mode == 4:
                    kwargs["pre_tokens"] = window_left
                    kwargs["next_tokens"] = 0
                # sparse_mode 3 and 4 both need atten_mask on CANN V3
                kwargs["atten_mask"] = (
                    _get_causal_mask(q.device) if sparse_mode == 3
                    else torch.zeros(
                        (1, 1, _MAX_ATTEN_SIZE, _MAX_ATTEN_SIZE),
                        dtype=torch.bool,
                        device=q.device,
                    )
                )

                out, _ = fused_attn(q_tnd, k_tnd, v_tnd, **kwargs)
                # TND output is [total_q, num_q_heads, head_dim] — already correct
                use_fused = True
            except RuntimeError as e:
                logger.warning(
                    "NPU fused attention failed, falling back to SDPA: %s", e
                )
        if not use_fused:
            # Fallback: SDPA decode with proper page table gathering
            num_pages, page_size, num_kv_heads, head_dim = k_cache.shape
            max_pages_per_seq = page_table.shape[1]
            if batch_size == 1:
                num_kv = int(cache_seqlens[0].item())
                pages = page_table[0, :(num_kv + page_size - 1) // page_size]
                gathered_k = k_cache[pages.long()].reshape(-1, num_kv_heads, head_dim)[:num_kv]
                gathered_v = v_cache[pages.long()].reshape(-1, num_kv_heads, head_dim)[:num_kv]
                if num_q_heads != num_kv_heads:
                    n_reps = num_q_heads // num_kv_heads
                    gathered_k = gathered_k.repeat_interleave(n_reps, dim=1)
                    gathered_v = gathered_v.repeat_interleave(n_reps, dim=1)
                q_2d = q.reshape(1, -1, num_q_heads, head_dim)
                k_2d = gathered_k.unsqueeze(0)
                v_2d = gathered_v.unsqueeze(0)
                out = torch.nn.functional.scaled_dot_product_attention(
                    q_2d.transpose(1, 2),
                    k_2d.transpose(1, 2),
                    v_2d.transpose(1, 2),
                    scale=scale,
                ).transpose(1, 2).squeeze(0)
            else:
                outputs = []
                offset = 0
                for b in range(batch_size):
                    num_kv = int(cache_seqlens[b].item())
                    pages = page_table[b, :(num_kv + page_size - 1) // page_size]
                    gathered_k = k_cache[pages.long()].reshape(-1, num_kv_heads, head_dim)[:num_kv]
                    gathered_v = v_cache[pages.long()].reshape(-1, num_kv_heads, head_dim)[:num_kv]
                    if num_q_heads != num_kv_heads:
                        n_reps = num_q_heads // num_kv_heads
                        gathered_k = gathered_k.repeat_interleave(n_reps, dim=1)
                        gathered_v = gathered_v.repeat_interleave(n_reps, dim=1)
                    q_b = q[offset:offset + max_seqlen_q].reshape(1, max_seqlen_q, num_q_heads, head_dim)
                    k_b = gathered_k.unsqueeze(0)
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
