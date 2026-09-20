"""Paid bounded builder. Uninitialized capacity is never logical CSR data."""
import ctypes as c
import os
import torch

LIB = os.environ['GB_LIB']
lib = c.CDLL(LIB)
lib.gb_build.argtypes = [c.c_int, c.c_int] + [c.c_void_p] * 8
lib.gb_build.restype = c.c_int


def validate(n, edges, ids=True):
    if not 1 <= n <= 2048 or edges.ndim != 2 or edges.shape[0] != 2 or edges.shape[1] > 1000000:
        raise ValueError('bounded graph shape')
    if edges.dtype != torch.int64 or edges.device != torch.device('cuda', torch.cuda.current_device()):
        raise TypeError('current CUDA device and int64 required')
    if ids and edges.numel() and (edges.min().item() < 0 or edges.max().item() >= n):
        raise ValueError('vertex ID range')


class Workspace:
    def __init__(self, n, m, device):
        if not 1 <= n <= 2048 or not 0 <= m <= 1000000:
            raise ValueError('bounded workspace')
        self.n, self.m = n, m
        self.stride = ((n + 31) // 32) * 32
        self.capacity = min(2 * m, n * n)
        self.hist = torch.empty((n, self.stride), dtype=torch.int32, device=device)
        self.counts = torch.empty(n, dtype=torch.int32, device=device)
        self.rp = torch.empty(n + 1, dtype=torch.int64, device=device)
        self.col = torch.empty(self.capacity, dtype=torch.int64, device=device)
        self.values = torch.empty_like(self.col)
        self.bits = torch.empty((n, (n + 31) // 32), dtype=torch.int32, device=device)

    def run(self, edges):
        if edges.shape != (2, self.m) or edges.dtype != torch.int64 or edges.device != self.hist.device:
            raise ValueError('workspace input mismatch')
        edges = edges.contiguous()
        self.hist.zero_()
        tensors = [edges, self.hist, self.counts, self.rp, self.col, self.values, self.bits]
        rc = lib.gb_build(self.n, self.m, *(t.data_ptr() for t in tensors), torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f'gb_build CUDA error {rc}')
        return self.rp, self.col, self.values, self.bits


def graph(n, edges, checked=False):
    validate(n, edges, ids=not checked)
    work = Workspace(n, edges.shape[1], edges.device)
    return work.run(edges)
