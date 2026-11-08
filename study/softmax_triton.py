"""The TRITON arm: a hand-written fused row-wise softmax kernel.

One program per row. The row is loaded once into registers, the
max and the exp-sum reductions are computed in registers, and the
normalized row is stored once: one global read and one global
write per row, which is the minimum traffic softmax allows.
Masked lanes load as minus infinity so they vanish from the max,
and their exp is exactly zero, so they never reach the output.

Configuration is frozen in PREREGISTRATION.md: BLOCK is the
smallest power of two at or above the column count, and num_warps
follows the frozen rule (4 up to BLOCK 1024, 8 up to 4096, 16
above).
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, out_ptr, stride_row, n_cols,
                   BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    ptrs = x_ptr + row * stride_row + offs
    x = tl.load(ptrs, mask=mask, other=float("-inf"))
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    tl.store(out_ptr + row * stride_row + offs, e / denom, mask=mask)


def block_for(n_cols: int) -> int:
    block = 1
    while block < n_cols:
        block *= 2
    return block


def warps_for(block: int) -> int:
    if block <= 1024:
        return 4
    if block <= 4096:
        return 8
    return 16


def run_triton(x, out):
    rows, n_cols = x.shape
    block = block_for(n_cols)
    return softmax_kernel[(rows,)](
        x, out, x.stride(0), n_cols, BLOCK=block,
        num_warps=warps_for(block))
