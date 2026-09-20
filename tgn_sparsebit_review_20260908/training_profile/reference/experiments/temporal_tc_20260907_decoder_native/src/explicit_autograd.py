"""Minimal autograd bridge for the frozen explicit_v2 CSR ALG3 operator.

This module deliberately has no project-root imports at module import time.  The
runner supplies the already-loaded ``explicit_v2`` module and the immutable CSR
tensors produced by the frozen list Plan.  That keeps this adapter separate from
the existing modules named ``run`` and ``backend``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Tuple

import torch


CSR = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class SpMMBackend(Protocol):
    """The small common surface used by production and CPU control backends."""

    def mm(
        self,
        rowptr: torch.Tensor,
        col: torch.Tensor,
        values: torch.Tensor,
        rows: int,
        cols: int,
        x: torch.Tensor,
        alg: int,
    ) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class ExplicitAlg3Backend:
    """Production wrapper around ``explicit_v2.mm``; ALG3 is fixed at call site."""

    explicit: Any

    def mm(
        self,
        rowptr: torch.Tensor,
        col: torch.Tensor,
        values: torch.Tensor,
        rows: int,
        cols: int,
        x: torch.Tensor,
        alg: int,
    ) -> torch.Tensor:
        # explicit_v2.mm owns the current CUDA stream and validates dtype/device.
        y, _resources = self.explicit.mm(rowptr, col, values, rows, cols, x, alg)
        return y


@dataclass(frozen=True)
class DenseControlBackend:
    """CPU-only reference backend for exercising the same autograd Function.

    It is not used by the GPU runner and makes no claim about the native ALG3
    implementation.  It deliberately materializes a dense matrix with indexed
    accumulation rather than delegating to another sparse API, so the small
    control has an independent CPU oracle boundary.
    """

    def mm(
        self,
        rowptr: torch.Tensor,
        col: torch.Tensor,
        values: torch.Tensor,
        rows: int,
        cols: int,
        x: torch.Tensor,
        alg: int,
    ) -> torch.Tensor:
        del alg
        if x.device.type != "cpu":
            raise ValueError("DenseControlBackend is CPU-only")
        counts = rowptr[1:] - rowptr[:-1]
        row = torch.repeat_interleave(
            torch.arange(rows, dtype=torch.int64, device=x.device), counts, output_size=col.numel()
        )
        dense = torch.zeros((rows, cols), dtype=x.dtype, device=x.device)
        # accumulate=True preserves signed duplicate entries without any sparse API.
        dense.index_put_((row, col), values.to(dtype=x.dtype), accumulate=True)
        return dense.matmul(x)


def _validate_csr(
    rowptr: torch.Tensor,
    col: torch.Tensor,
    values: torch.Tensor,
    rows: int,
    cols: int,
    x: torch.Tensor,
) -> None:
    if x.ndim != 2 or x.shape[0] != cols:
        raise ValueError(f"x must have shape ({cols}, D), got {tuple(x.shape)}")
    if x.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"x dtype must be FP32/FP64, got {x.dtype}")
    if rows < 0 or cols < 0:
        raise ValueError("CSR dimensions must be nonnegative")
    if any(t.dtype != torch.int64 for t in (rowptr, col, values)):
        raise TypeError("frozen CSR rowptr/col/values must be int64")
    if rowptr.ndim != 1 or col.ndim != 1 or values.ndim != 1:
        raise ValueError("CSR tensors must be one-dimensional")
    if rowptr.numel() != rows + 1 or col.numel() != values.numel():
        raise ValueError("inconsistent CSR sizes")
    if any(t.device != x.device for t in (rowptr, col, values)):
        raise ValueError("CSR and feature tensors must share a device")
    if not all(t.is_contiguous() for t in (rowptr, col, values)):
        raise ValueError("CSR tensors must be contiguous")


class CSRSpMMFunction(torch.autograd.Function):
    """Autograd for C @ X using the frozen CSR C and its paid transpose.

    Saving all six CSR tensors is intentional: PyTorch's saved-tensor version
    checks reject an in-place modification of either the forward CSR or the paid
    transpose before backward.  The backward result is only dX; graph indices and
    integer coefficients are immutable diagnostic inputs, not differentiable
    parameters.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        rowptr: torch.Tensor,
        col: torch.Tensor,
        values: torch.Tensor,
        transpose_rowptr: torch.Tensor,
        transpose_col: torch.Tensor,
        transpose_values: torch.Tensor,
        rows: int,
        cols: int,
        backend: SpMMBackend,
        alg: int,
    ) -> torch.Tensor:
        _validate_csr(rowptr, col, values, rows, cols, x)
        _validate_csr(
            transpose_rowptr,
            transpose_col,
            transpose_values,
            cols,
            rows,
            torch.empty((rows, x.shape[1]), dtype=x.dtype, device=x.device),
        )
        if alg != 12:
            raise ValueError("R3/R64 require frozen CUSPARSE_SPMM_CSR_ALG3 (12)")
        ctx.rows = rows
        ctx.cols = cols
        ctx.backend = backend
        ctx.alg = alg
        ctx.save_for_backward(
            rowptr,
            col,
            values,
            transpose_rowptr,
            transpose_col,
            transpose_values,
        )
        y = backend.mm(rowptr, col, values, rows, cols, x, alg)
        if y.shape != (rows, x.shape[1]) or y.dtype != x.dtype or y.device != x.device:
            raise RuntimeError("backend returned an invalid SpMM result")
        return y

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, grad_y: torch.Tensor) -> tuple[Any, ...]:
        (
            rowptr,
            col,
            values,
            transpose_rowptr,
            transpose_col,
            transpose_values,
        ) = ctx.saved_tensors
        if grad_y.ndim != 2 or grad_y.shape[0] != ctx.rows:
            raise RuntimeError("invalid gradient shape for explicit CSR SpMM")
        grad_x = ctx.backend.mm(
            transpose_rowptr,
            transpose_col,
            transpose_values,
            ctx.cols,
            ctx.rows,
            grad_y.contiguous(),
            ctx.alg,
        )
        # x plus 10 non-differentiable arguments.
        return (grad_x, None, None, None, None, None, None, None, None, None, None)


def build_paid_transpose(explicit: Any, plan: Any) -> CSR:
    """Use only the frozen explicit_v2 paid transpose helper."""

    return explicit.transpose(plan)


def explicit_consume(
    plan: Any,
    x: torch.Tensor,
    backend: SpMMBackend,
    transpose: CSR,
    alg: int = 12,
) -> torch.Tensor:
    """Return the source decoder's B x (7D) layout from a frozen list Plan."""

    rowptr, col, values = plan.rp, plan.col, plan.iv
    transpose_rowptr, transpose_col, transpose_values = transpose
    yc = CSRSpMMFunction.apply(
        x,
        rowptr,
        col,
        values,
        transpose_rowptr,
        transpose_col,
        transpose_values,
        plan.m,
        plan.n,
        backend,
        alg,
    )
    batch = plan.m // 7
    if plan.m != 7 * batch:
        raise ValueError("source mode2 Plan must have exactly seven coefficient channels")
    return yc.reshape(7, batch, x.shape[1]).permute(1, 0, 2).reshape(batch, 7 * x.shape[1])
