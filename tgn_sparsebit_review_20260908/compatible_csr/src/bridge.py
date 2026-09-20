"""Source-compatible consumers built directly from existing integer CSR."""
from pathlib import Path
import sys
import types

DIAG = Path('/home/data/wangxuran/tncn_numeric_diagnosis_20260908')
sys.path.insert(0, str(DIAG/'src'))
import diagnose as d
p, torch = d.p, d.torch


def from_graph_with_bounds(n, targets, packed):
    """Same count/pack kernels; NNZ and channel offsets share the host read."""
    assert 1 <= n <= 2048 and 1 <= targets.shape[1] <= 64
    plan = p.ref.Plan.__new__(p.ref.Plan)
    plan.n, plan.b = n, targets.shape[1]
    plan.m = 7*plan.b
    targets = targets.contiguous()
    ap, ac, av, bits = packed
    counts = torch.empty(plan.m, device=targets.device, dtype=torch.int64)
    args = [n, plan.b, p.ref.ptr(ap), p.ref.ptr(ac), p.ref.ptr(av), p.ref.ptr(bits),
            p.ref.ptr(targets), p.ref.ptr(counts)]
    p.ref.call('nt_produce', *args, None, None, None, 0, 0, p.ref.stream())
    plan.rp = torch.cat([torch.zeros(1, device=targets.device, dtype=torch.int64), counts.cumsum(0)])
    plan.channel_bounds = plan.rp[::plan.b].cpu().tolist()
    plan.nnz = plan.channel_bounds[-1]
    plan.col = torch.empty(plan.nnz, device=targets.device, dtype=torch.int64)
    plan.iv = torch.empty_like(plan.col)
    p.ref.call('nt_produce', *args, p.ref.ptr(plan.rp), p.ref.ptr(plan.col), p.ref.ptr(plan.iv), 0, 1, p.ref.stream())
    plan.row = plan.perm = plan.tr = plan.tp = None
    return plan


def direct_channels(plans, fused_bounds=False):
    """Stable row/channel concatenation; nonzero columns keep their order."""
    n, batch = plans[0].n, sum(x.b for x in plans)
    boundaries = torch.stack([x.rp[::x.b] for x in plans])
    host_bounds = [x.channel_bounds for x in plans] if fused_bounds else boundaries.cpu().tolist()
    counts = boundaries[:, 1:]-boundaries[:, :-1]
    ends = counts.cumsum(0)
    shifts = ends-counts-boundaries[:, :-1]
    pieces = [x.rp[:-1].reshape(7, x.b)+shifts[i, :, None] for i, x in enumerate(plans)]
    rowptrs = torch.cat(pieces+[ends[-1, :, None]], dim=1)
    floats = [x.iv.float() for x in plans]
    result = []
    for channel in range(7):
        cols, values = [], []
        for i, plan in enumerate(plans):
            begin, end = host_bounds[i][channel:channel+2]
            cols.append(plan.col[begin:end]); values.append(floats[i][begin:end])
        col = cols[0] if len(cols)==1 else torch.cat(cols)
        value = values[0] if len(values)==1 else torch.cat(values)
        result.append(p.torch_sparse.SparseTensor(rowptr=rowptrs[channel], col=col, value=value,
            sparse_sizes=(batch,n), is_sorted=True, trust_data=True))
    return result


def install(tr, variant='direct', capture=None, verify=False, clock=None):
    assert variant in ['direct','fused_bounds']
    pred = tr.model['pred']
    clock = clock or p.Clock()
    pred._paid_graph = None
    def get_cn(self, x, edges, q, mode, decay=False, time_info=None):
        assert mode==2 and not decay and 1<=len(x)<=2048
        with clock.phase('graph_build'):
            if self._paid_graph is None:
                self._paid_graph = p.builder.graph(len(x), edges, checked=True)
        with clock.phase('integer_producer'):
            make = from_graph_with_bounds if variant=='fused_bounds' else p.ref.from_graph
            plans = [make(len(x), q[:, j:j+64], self._paid_graph) for j in range(0,q.shape[1],64)]
        with clock.phase('csr_conversion'):
            channels = direct_channels(plans, fused_bounds=variant=='fused_bounds')
        if verify or capture is not None:
            data = dict(n=len(x), b=q.shape[1], queries=q.detach().cpu().clone(),
                        csr=[tuple(t.detach().cpu().clone() for t in c.csr()) for c in channels],
                        input_csr=[dict(b=z.b, rp=z.rp.cpu().clone(), col=z.col.cpu().clone(), iv=z.iv.cpu().clone()) for z in plans])
            if verify:
                expected = [p.torch_sparse.SparseTensor.from_dense(c) for c in d.dense_plans(plans).float()]
                for actual, reference in zip(channels, expected):
                    for a,b in zip(actual.csr(),reference.csr()):
                        assert torch.equal(a,b), 'CSR differs from dense compatibility conversion'
                data['exact_dense_compatibility'] = True
            if capture is not None:
                capture.append(data)
        with clock.phase('seven_spmm'):
            # Retain source autograd creation order, including the last channel.
            return torch.cat([p.predmod.spmm_add(c,x) for c in channels],dim=-1)
    pred.get_cn_emb = types.MethodType(get_cn,pred)


def step(tr, index, variant='direct', **kwargs):
    if variant in ['direct','fused_bounds','compatible']:
        tr.model['pred']._paid_graph = None
        variant = 'source'
    return tr.step(index,variant,**kwargs)
