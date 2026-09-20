EXPECTED = {}
"""Read-only existing-binary diagnostic. EXPECTED is injected by local runner."""
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path.cwd()
OLD = ROOT / 'experiments/temporal_tc_20260905_gpu2_notc'
PREV = ROOT / 'experiments/temporal_tc_20260905_tncn_mode2_gate0'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def emit(kind, **data):
    print(json.dumps(dict(kind=kind, **data)), flush=True)


for path, value in EXPECTED.items():
    assert sha(ROOT / path) == value, path
sys.path.insert(0, str(OLD / 'src'))
from backend_v2 import Plan, call, ptr, stream, LIB
sys.path.insert(0, str(PREV / 'src'))
from common import source
from check_small import snap


def guard(initial=False):
    s = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader'], text=True)
    selected = [r.strip() for r in s.splitlines() if r.split(',')[0].strip() == os.environ['CUDA_VISIBLE_DEVICES']]
    if initial:
        assert not selected, ('GPU_OCCUPIED_BEFORE_INIT', selected)
    else:
        assert len(selected) == 1 and int(selected[0].split(',')[1]) == os.getpid(), ('GPU_ISOLATION_FAILED', selected)
    return selected


def graph(n, edges):
    """Exactly the immutable Plan graph-preparation statements."""
    device = edges.device
    sym = torch.cat([edges, edges.flip(0)], dim=1)
    a = torch.sparse_coo_tensor(sym, torch.ones(sym.shape[1], device=device, dtype=torch.int64), (n, n)).coalesce()
    ac = a.indices()[1].contiguous()
    av = a.values()
    ar = a.indices()[0]
    ap = torch.cat([torch.zeros(1, device=device, dtype=torch.int64), torch.bincount(ar, minlength=n).cumsum(0)])
    bits = torch.zeros((n, (n + 31) // 32), device=device, dtype=torch.int32)
    call('nt_bits', n, ptr(ap), ptr(ac), ptr(bits), stream())
    return ap, ac, av, bits


def from_graph(n, targets, packed):
    """Unchanged list producer, exact allocation; no unused native transpose."""
    p = Plan.__new__(Plan)
    p.n, p.b = n, targets.shape[1]
    p.m = 7 * p.b
    device = targets.device
    targets = targets.contiguous()
    ap, ac, av, bits = packed
    counts = torch.empty(p.m, device=device, dtype=torch.int64)
    args = [n, p.b, ptr(ap), ptr(ac), ptr(av), ptr(bits), ptr(targets), ptr(counts)]
    call('nt_produce', *args, None, None, None, 0, 0, stream())
    p.rp = torch.cat([torch.zeros(1, device=device, dtype=torch.int64), counts.cumsum(0)])
    p.nnz = int(p.rp[-1].item())
    p.col = torch.empty(p.nnz, device=device, dtype=torch.int64)
    p.iv = torch.empty_like(p.col)
    call('nt_produce', *args, ptr(p.rp), ptr(p.col), ptr(p.iv), 0, 1, stream())
    p.row = p.perm = p.tr = p.tp = None
    return p


def states():
    for case in json.loads((PREV / 'output/census1/census.json').read_text()):
        state = next(s for s in case['states'] if s['state'] == 'before')
        path = PREV / 'output/census1' / state['fixture']
        assert sha(path) == state['fixture_sha256']
        with np.load(path) as z:
            n = len(z['ids'])
            e = torch.tensor(z['edges'], device='cuda')
            q = torch.tensor(z['targets'], device='cuda')
        yield case, state, n, e, q


def fresh(n, e, q):
    return Plan(n, e, q, 'list', checked=True, build_transpose=False)


def dense(p):
    a = torch.sparse_csr_tensor(p.rp, p.col, p.iv, size=(p.m, p.n))
    return a.to_dense().reshape(7, p.b, p.n).cpu().numpy()


def compare(a, b):
    assert a.keys() == b.keys()
    err, bitwise = 0., True
    for k in a:
        if a[k] is None or b[k] is None:
            assert a[k] is None and b[k] is None
            continue
        assert torch.isfinite(a[k]).all() and torch.isfinite(b[k]).all()
        torch.testing.assert_close(a[k], b[k], atol=2e-4, rtol=2e-4, msg=k)
        err = max(err, float((a[k] - b[k]).abs().max()))
        bitwise = bitwise and torch.equal(a[k], b[k])
    return dict(max_abs=err, bitwise=bitwise)


def setup(ns, n, seed):
    torch.manual_seed(seed)
    m = ns['NCNPredictor'](100, 256, 1, 2).cuda()
    rng = np.random.RandomState(seed)
    x = torch.tensor(rng.randn(n, 100) * .1, dtype=torch.float32, device='cuda', requires_grad=True)
    return m, x, torch.optim.Adam(m.parameters(), lr=1e-4)


def step(m, x, opt, q, factory, checking=False):
    opt.zero_grad(set_to_none=True)
    x.grad = None
    p = factory()
    y = p.consume(x, library=True)
    features = torch.cat([x[q[0]] * x[q[1]], y], -1)
    pos, neg = m.xsmlp(features[:32]), m.xsmlp(features[32:])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(pos, torch.ones_like(pos)) + torch.nn.functional.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
    loss.backward()
    opt.step()
    if checking:
        s = snap(m, x, torch.cat([pos, neg]), opt)
        s['loss'] = loss.detach().cpu().clone()
        return s


def correct(ns):
    count = 0
    for i, (case, state, n, e, q) in enumerate(states()):
        guard()
        gg = graph(n, e)
        plans = [fresh(n, e, q), from_graph(n, q, gg), fresh(n, e, q)]
        cc = dense(plans[0])
        digest = hashlib.sha256(np.ascontiguousarray(cc, dtype=np.int64).tobytes()).hexdigest()
        assert digest == state['coefficient_sha256']
        for p in plans[1:]:
            for field in ['rp', 'col', 'iv']:
                assert torch.equal(getattr(plans[0], field), getattr(p, field))
            assert hashlib.sha256(dense(p).tobytes()).hexdigest() == digest
        factories = [lambda: fresh(n, e, q), lambda: from_graph(n, q, gg), lambda: plans[2]]
        snaps = []
        for fac in factories:
            m, x, opt = setup(ns, n, 20260905 + i)
            snaps.append([step(m, x, opt, q, fac, True) for _ in range(2)])
        checks = []
        for v in [1, 2]:
            for t in range(2):
                checks.append(dict(variant='R' + str(v), step=t, **compare(snaps[0][t], snaps[v][t])))
                count += 1
        p = plans[0]
        emit('correct', case=case['case'], fixture=state['fixture'], checks=checks,
             coefficient_sha256=digest, structures=3, nnz=p.nnz,
             C_int64_CSR_bytes=8 * (p.m + 1) + 16 * p.nnz,
             C_fp32_value_bytes=4 * p.nnz,
             graph_tensor_bytes=sum(t.numel() * t.element_size() for t in gg),
             producer_shared_bytes=8 * n + 4 * (2 * ((n + 31) // 32) + 18),
             global_Q_bytes=0, global_S_bytes=0, manual_CSC_bytes=0)
        guard()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    fixtures = list(states())
    for k in range(24):
        case, _, n, e, q = fixtures[k % len(fixtures)]
        # Uploaded fixtures are on the default stream; explicitly order reads.
        streams[k % 2].wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(streams[k % 2]):
            e2, q2 = e.clone(), q.clone()
            gg = graph(n, e2)
            a, b = fresh(n, e2, q2), from_graph(n, q2, gg)
            torch.manual_seed(20260905 + k)
            xx = torch.randn(n, 33, device='cuda') * .1
            up = torch.randn(64, 7 * 33, device='cuda') * .1
            vals = []
            for plan in [a, b]:
                x = xx.clone().requires_grad_()
                y = plan.consume(x, library=True)
                (y * up).sum().backward()
                vals.append(dict(output=y.detach(), gradient=x.grad.detach()))
            c = compare(*vals)
        streams[k % 2].synchronize()
        emit('stress', index=k, case=case['case'], **c)
        guard()
    emit('summary', mode='correct', states=18, structure_checks=54,
         optimizer_checks=count, stream_churn_checks=24, all_pass=True)


def bench(ns, process):
    order = list(itertools.permutations(['R0', 'R1', 'R2']))[process]
    retained = warmups = 0
    for i, (case, state, n, e, q) in enumerate(states()):
        guard()
        gg, cached = graph(n, e), fresh(n, e, q)
        factories = dict(R0=lambda: fresh(n, e, q),
                         R1=lambda: from_graph(n, q, gg), R2=lambda: cached)
        for name in order:
            guard()
            m, x, opt = setup(ns, n, 20260905 + i)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            observations = []
            for k in range(10):
                torch.cuda.synchronize()
                t = time.perf_counter_ns()
                start.record()
                step(m, x, opt, q, factories[name])
                end.record()
                torch.cuda.synchronize()
                us = (time.perf_counter_ns() - t) / 1000
                observations.append(dict(kind='sample', case=case['case'], dataset=case['dataset'],
                    fixture=state['fixture'], process=process, order=order, variant=name,
                    observation=k-3, warmup=k < 3, wall_us=us,
                    event_us=start.elapsed_time(end)*1000, eligible=True))
                warmups += int(k < 3)
                retained += int(k >= 3)
            try:
                guard()
            except Exception:
                for r in observations:
                    r['eligible'] = False
                    print(json.dumps(r), flush=True)
                raise
            for r in observations:
                print(json.dumps(r), flush=True)
    emit('summary', mode='bench', process=process, order=order,
         retained=retained, warmups=warmups, all_pass=True)


def main():
    guard(True)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.zeros(1, device='cuda')
    torch.cuda.synchronize()
    apps = guard()
    git = subprocess.run(['git', '-C', str(ROOT/'experiments/temporal_tc_20260905_cooccurrence_gate0/sources/TNCN'), 'rev-parse', 'HEAD'], text=True, capture_output=True)
    emit('opening', pid=os.getpid(), mode=os.environ['RUN_MODE'], process=int(os.environ['RUN_PROCESS']),
         host=platform.node(), torch=torch.__version__, numpy=np.__version__,
         cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
         capability=torch.cuda.get_device_capability(), apps=apps, binary_sha256=sha(LIB),
         hardware_snapshot=subprocess.check_output(['nvidia-smi', '--id='+os.environ['CUDA_VISIBLE_DEVICES'], '--query-gpu=uuid,driver_version,clocks.current.sm,clocks.current.memory,power.limit,pstate,temperature.gpu,memory.used,utilization.gpu', '--format=csv'], text=True),
         source_hashes=EXPECTED, author_commit=git.stdout.strip() if git.returncode == 0 else None,
         author_git_returncode=git.returncode, author_git_error=git.stderr.strip())
    ns = source('interpreter')
    if os.environ['RUN_MODE'] == 'correct':
        correct(ns)
    else:
        bench(ns, int(os.environ['RUN_PROCESS']))
    emit('closing', pid=os.getpid(), apps=guard(), all_pass=True)


if __name__ == '__main__':
    main()
