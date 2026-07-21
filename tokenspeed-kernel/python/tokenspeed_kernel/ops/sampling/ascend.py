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

"""Ascend NPU sampling kernels.

Registers NPU-optimised sampling kernels using ``torch_npu`` native ops:
  - ``argmax`` — pure-torch argmax on NPU
  - ``top_k_top_p_sample`` — fused top-k + top-p sampling via
    ``torch_npu.npu_top_k_top_p_sample``
  - ``random_sample`` — Gumbel-max fallback sampling
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel

# ---------------------------------------------------------------------------
# NPU availability helpers
# ---------------------------------------------------------------------------


def _is_npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _get_npu_top_k_top_p_sample():
    """Return ``torch_npu.npu_top_k_top_p_sample`` callable or ``None``."""
    try:
        import torch_npu  # noqa: F401

        return torch.ops.npu.npu_top_k_top_p_sample
    except (ImportError, AttributeError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# npu_argmax — argmax kernel registered for "sampling" / "argmax"
# ---------------------------------------------------------------------------


@register_kernel(
    "sampling",
    "argmax",
    name="npu_argmax",
    solution="ascend",
    capability=CapabilityRequirement(vendors=frozenset({"huawei"})),
    signatures=frozenset(),
    priority=Priority.PERFORMANT,
    tags={"portability"},
)
def argmax_ascend(
    logits: torch.Tensor, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Row-wise argmax over the last logits dimension on NPU.

    Args:
        logits: Input logits ``(M, N)`` on NPU.
        out: Optional output buffer ``(M,)`` int32 on NPU.

    Returns:
        Argmax indices for each row.
    """
    if not _is_npu_available():
        raise RuntimeError("NPU not available")
    result = torch.argmax(logits, dim=-1).to(torch.int32)
    if out is not None:
        out.copy_(result)
        return out
    return result


# ---------------------------------------------------------------------------
# npu_top_k_top_p_sample — fused top-k + top-p sampling
# ---------------------------------------------------------------------------


@register_kernel(
    "sampling",
    "top_k_top_p_sample",
    name="npu_top_k_top_p_sample",
    solution="ascend",
    capability=CapabilityRequirement(vendors=frozenset({"huawei"})),
    signatures=frozenset(),
    priority=Priority.PERFORMANT,
    tags={"portability"},
)
def top_k_top_p_sample_ascend(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Fused top-k + top-p sampling using ``torch_npu.npu_top_k_top_p_sample``.

    Args:
        logits: Input logits ``(M, N)`` on NPU.
        top_k: Per-row top-k values ``(M,)`` int32.
        top_p: Per-row top-p values ``(M,)`` float32.
        generator: Optional torch.Generator (not all impls support it).
        seed: Optional seed override.

    Returns:
        Sampled token ids ``(M,)`` int32.
    """
    if not _is_npu_available():
        raise RuntimeError("NPU not available")

    impl = _get_npu_top_k_top_p_sample()
    if impl is not None:
        # torch_npu.npu_top_k_top_p_sample expects top_p dtype to match
        # logits dtype (float16 or bfloat16 for half, float32 for float32).
        if top_p.dtype != logits.dtype:
            top_p = top_p.to(logits.dtype)
        sampled = impl(
            logits, top_k, top_p, post_sample="multiNomial", generator=generator
        )
        # API returns (sampled_ids, filtered_logits) tuple
        if isinstance(sampled, (list, tuple)):
            sampled = sampled[0]
        return sampled.to(torch.int32)

    # Fallback: pure-torch top-k + top-p + Gumbel-max
    return _top_k_top_p_sample_fallback(logits, top_k, top_p, seed=seed)


def _top_k_top_p_sample_fallback(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    *,
    seed: int | None = None,
) -> torch.Tensor:
    """Pure-torch fallback for top-k + top-p sampling using Gumbel-max trick."""
    M, N = logits.shape

    # Temperature softmax
    probs = torch.softmax(logits.float(), dim=-1)

    # Top-k filtering (per-row)
    if top_k is not None:
        k = top_k.int()
        # Sort descending to find k-th cutoff per row
        sorted_vals, sorted_idx = probs.sort(dim=-1, descending=True)
        batch_idx = torch.arange(M, device=probs.device).unsqueeze(-1)
        # Zero out probs beyond top_k
        mask = torch.arange(N, device=probs.device).unsqueeze(0) < k.unsqueeze(-1)
        filtered = torch.zeros_like(probs)
        filtered[batch_idx, sorted_idx] = sorted_vals * mask

        # Renormalise
        sums = filtered.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        probs = filtered / sums

    # Top-p filtering (per-row)
    if top_p is not None:
        sorted_vals, sorted_idx = probs.sort(dim=-1, descending=True)
        cumsum = sorted_vals.cumsum(dim=-1)
        mask = cumsum <= top_p.unsqueeze(-1)
        # Always keep at least one token
        mask[..., 0] = True
        filtered = torch.zeros_like(probs)
        batch_idx = torch.arange(M, device=probs.device).unsqueeze(-1)
        filtered[batch_idx, sorted_idx] = sorted_vals * mask
        sums = filtered.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        probs = filtered / sums

    # Gumbel-max sampling
    uniform = torch.rand_like(probs)
    gumbel = -torch.log(-torch.log(uniform.clamp(min=1e-10)))
    scores = torch.log(probs.clamp(min=1e-10)) + gumbel
    return scores.argmax(dim=-1).to(torch.int32)


# ---------------------------------------------------------------------------
# npu_random_sample — Gumbel-max random sampling (fallback)
# ---------------------------------------------------------------------------


@register_kernel(
    "sampling",
    "random_sample",
    name="npu_random_sample",
    solution="ascend",
    capability=CapabilityRequirement(vendors=frozenset({"huawei"})),
    signatures=frozenset(),
    priority=Priority.PERFORMANT,
    tags={"portability"},
)
def random_sample_ascend(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator] | None = None,
) -> torch.Tensor:
    """Gumbel-max random sampling from probabilities.

    Args:
        probs: Probability tensor ``(M, N)`` on NPU.
        generators: Optional per-row dict of torch.Generator for
            deterministic sampling.

    Returns:
        Sampled token ids ``(M,)`` int32.
    """
    if not _is_npu_available():
        raise RuntimeError("NPU not available")

    q = torch.empty_like(probs)
    if generators is not None and len(generators) == probs.shape[0]:
        for i, gen in generators.items():
            q[i].exponential_(generator=gen)
    else:
        q.exponential_()
    return probs.div_(q).argmax(dim=-1).to(torch.int32)
