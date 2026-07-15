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

"""
Pure PyTorch kernels for computing cache locations and updating page tables.

Replaces the original Triton implementation for NPU compatibility.
"""

import torch

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


def update_req_to_page(
    req_to_page: torch.Tensor,
    req_pool_indices: torch.Tensor,
    new_occupied_pages: torch.Tensor,
    new_occupied_pages_num: torch.Tensor,
    pages_copy_starts: torch.Tensor,
) -> None:
    """
    Update req_to_page table with new occupied pages.

    Pure PyTorch replacement for the Triton ``update_req_to_page_kernel``.
    For each request, scatter the newly occupied page IDs into the page table
    at the position indicated by ``pages_copy_starts``.

    Args:
        req_to_page: Request to page table [req_pool_size+1, context_len]
        req_pool_indices: Request pool indices [batch_size]
        new_occupied_pages: New page IDs [total_pages] - flattened
        new_occupied_pages_num: Number of new pages per request [batch_size]
        pages_copy_starts: Start position in req_to_page for each request [batch_size]
    """
    batch_size = req_pool_indices.shape[0]
    if new_occupied_pages.shape[0] == 0:
        return

    cumsum_pages = torch.cumsum(new_occupied_pages_num, dim=0)

    for i in range(batch_size):
        pool_idx = req_pool_indices[i]
        n = new_occupied_pages_num[i].item()
        if n == 0:
            continue
        start = pages_copy_starts[i].item()
        offset = cumsum_pages[i - 1].item() if i > 0 else 0
        req_to_page[pool_idx, start:start + n] = new_occupied_pages[offset:offset + n]


def compute_out_cache_loc(
    out_cache_loc_ptr,
    req_pool_indices: torch.Tensor,  # [batch_size]
    input_lengths: torch.Tensor,  # [batch_size]
    cache_start: torch.Tensor,  # [batch_size]
    req_to_pages: torch.Tensor,  # [req_pool_size+1, max_pages]
    page_size: int,
) -> None:
    """
    Compute output cache locations for the variable-length (prefill / decode) path.

    Pure PyTorch replacement for the Triton ``compute_out_cache_loc_kernel``.

    For each token in each request, computes:
        position = cache_start + token_offset
        page_idx = position // page_size  (clamped to max_pages-1)
        page_id = req_to_pages[pool_idx, page_idx]
        out_cache_loc = page_id * page_size + position % page_size
    Overflow tokens (page_idx >= max_pages) are routed to slot 0.
    """
    batch_size = req_pool_indices.shape[0]
    max_pages = req_to_pages.shape[1]

    cumsum = torch.cumsum(input_lengths, dim=0)

    for i in range(batch_size):
        pool_idx = req_pool_indices[i]
        length = input_lengths[i].item()
        start_pos = cache_start[i].item()
        output_offset = cumsum[i - 1].item() if i > 0 else 0

        if length == 0:
            continue

        # Vectorize the inner token loop with torch.arange
        token_offsets = torch.arange(length, device=out_cache_loc_ptr.device)
        positions = start_pos + token_offsets

        page_indices = positions // page_size
        overflow = page_indices >= max_pages
        page_indices = page_indices.clamp(max=max_pages - 1)
        offsets_in_page = positions % page_size

        page_ids = req_to_pages[pool_idx.long(), page_indices.long()]
        cache_locs = page_ids * page_size + offsets_in_page
        cache_locs[overflow] = 0  # route overflow to safe slot 0

        out_cache_loc_ptr[output_offset:output_offset + length] = cache_locs


def compute_out_cache_loc_uniform(
    out_cache_loc_ptr,
    req_pool_indices: torch.Tensor,  # [batch_size]
    uniform_input_length: int,
    cache_start: torch.Tensor,  # [batch_size]
    req_to_pages: torch.Tensor,  # [req_pool_size+1, max_pages]
    page_size: int,
) -> None:
    """
    Specialized entry point when every request has the same ``input_length``.

    Pure PyTorch replacement; avoids per-call ``cumsum`` and conditional branches.
    Used by the multi-step drafter where each request decodes exactly
    ``spec_num_steps - 1`` tokens.
    """
    batch_size = req_pool_indices.shape[0]
    max_pages = req_to_pages.shape[1]

    for i in range(batch_size):
        pool_idx = req_pool_indices[i]
        start_pos = cache_start[i].item()
        output_offset = i * uniform_input_length

        if uniform_input_length == 0:
            continue

        token_offsets = torch.arange(uniform_input_length, device=out_cache_loc_ptr.device)
        positions = start_pos + token_offsets

        page_indices = positions // page_size
        overflow = page_indices >= max_pages
        page_indices = page_indices.clamp(max=max_pages - 1)
        offsets_in_page = positions % page_size

        page_ids = req_to_pages[pool_idx.long(), page_indices.long()]
        cache_locs = page_ids * page_size + offsets_in_page
        cache_locs[overflow] = 0

        out_cache_loc_ptr[output_offset:output_offset + uniform_input_length] = cache_locs


def fused_decode_input_prep(
    out_cache_loc_ptr,
    positions_ptr,
    seq_lens_out_ptr,
    req_pool_indices: torch.Tensor,  # [batch_size]
    valid_cache_lengths: torch.Tensor,  # [req_pool_size+1]
    uniform_input_length: int,
    req_to_pages: torch.Tensor,  # [req_pool_size+1, max_pages]
    page_size: int,
) -> None:
    """
    Decode-only fast path: fuse indexSelect + gather + add into a single loop.

    Pure PyTorch replacement for the Triton ``fused_decode_input_prep_kernel``.
    Writes ``out_cache_loc``, ``positions``, and ``seq_lens`` in one pass.
    """
    batch_size = req_pool_indices.shape[0]
    max_pages = req_to_pages.shape[1]

    for i in range(batch_size):
        pool_idx = req_pool_indices[i]
        cache_start = valid_cache_lengths[pool_idx].item()
        output_offset = i * uniform_input_length

        # seq_lens_out[i] = cache_start + uniform_input_length
        seq_lens_out_ptr[i] = cache_start + uniform_input_length

        if uniform_input_length == 0:
            continue

        token_offsets = torch.arange(uniform_input_length, device=out_cache_loc_ptr.device)
        positions = cache_start + token_offsets

        page_indices = positions // page_size
        overflow = page_indices >= max_pages
        page_indices = page_indices.clamp(max=max_pages - 1)
        offsets_in_page = positions % page_size

        page_ids = req_to_pages[pool_idx.long(), page_indices.long()]
        cache_locs = page_ids * page_size + offsets_in_page
        cache_locs[overflow] = 0

        out_cache_loc_ptr[output_offset:output_offset + uniform_input_length] = cache_locs
        positions_ptr[output_offset:output_offset + uniform_input_length] = positions


def update_block_table(forward_op, device, req_to_page):
    def flatten_and_to_device(data, dtype=torch.int32):
        if not data:
            return torch.tensor([], dtype=dtype, device=device)

        # Flatten one level if data is a list of lists
        if isinstance(data[0], (list, tuple)):
            flat = [x for inner in data for x in inner]
        else:
            flat = data

        if not flat:
            return torch.tensor([], dtype=dtype, device=device)

        tensor = torch.tensor(flat, dtype=dtype, device="cpu", pin_memory=True)
        return tensor.to(device, non_blocking=True)

    # sizes[i] is the number of newly allocated pages for request i.
    if all(n == 0 for n in forward_op.sizes):
        return

    max_pages = req_to_page.shape[1]
    # Clamp a request that would overflow req_to_page instead of crashing the
    # engine. Happens when MTP accept-rate collapse keeps a request alive past
    # context_len; its KV drops but it will be finished shortly.
    sizes = list(forward_op.sizes)
    begins = list(forward_op.begins)
    # new_occupied_pages is a list-of-lists [batch, size_i] of page ids;
    # take a shallow copy so we can trim the offending request's row.
    new_occupied_pages = [list(row) for row in forward_op.new_occupied_pages]
    request_ids = list(forward_op.request_ids)
    for i, (begin, size) in enumerate(zip(begins, sizes)):
        if begin + size > max_pages:
            clamped = max(0, max_pages - begin)
            logger.warning(
                "page copy would exceed req_to_page capacity for req %s: "
                "begin=%s + size=%s = %s > req_to_page.shape[1]=%s; "
                "clamping size to %s to avoid engine crash. The request is past "
                "its context-length bound and will be finished by the length "
                "check; KV writes after this point are dropped.",
                request_ids[i] if i < len(request_ids) else "?",
                begin,
                size,
                begin + size,
                max_pages,
                clamped,
            )
            sizes[i] = clamped
            # Keep new_occupied_pages[i] consistent with the clamped size so
            # the kernel's cumsum-based offsets stay aligned across the batch.
            new_occupied_pages[i] = new_occupied_pages[i][:clamped]

    new_occupied_pages_num = flatten_and_to_device(sizes, dtype=torch.int32)
    pages_copy_starts = flatten_and_to_device(begins, dtype=torch.int32)
    new_occupied_pages_t = flatten_and_to_device(new_occupied_pages, dtype=torch.int32)
    request_pool_indices = flatten_and_to_device(
        forward_op.request_pool_indices, dtype=torch.int64
    )
    update_req_to_page(
        req_to_page=req_to_page,
        req_pool_indices=request_pool_indices,
        new_occupied_pages=new_occupied_pages_t,
        new_occupied_pages_num=new_occupied_pages_num,
        pages_copy_starts=pages_copy_starts,
    )
