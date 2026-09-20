"""Full training-step accounting. Run via supervise.py on the reserved GPU."""
import argparse
import ast
import contextlib
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

ROOT = Path('/home/data/wangxuran/factor_tgn_sptc_20260902/project')
EXP = Path(__file__).resolve().parents[1]
DEVICE = os.environ.get('TRAIN_DEVICE', 'cuda')
SRC = ROOT/'experiments/temporal_tc_20260905_cooccurrence_gate0/sources/TNCN'
sys.path[:0] = [str(EXP/'deps'), str(SRC),
    str(ROOT/'experiments/temporal_tc_20260905_graph_builder/src')]
os.environ['GB_LIB'] = str(ROOT/'experiments/temporal_tc_20260905_graph_builder/output/build1/graph_builder.so')
os.environ['GPU_NT_LIB'] = str(ROOT/'experiments/temporal_tc_20260905_gpu2_notc/output/gpu2/native_gpu.so')
import numpy as np
import torch
import torch_sparse
import torch_scatter
import torch_geometric
import builder
import reference as ref
from modules.memory_module import TGNMemory
from modules.emb_module import GraphAttentionEmbedding
from modules.msg_func import IdentityMessage
from modules.msg_agg import LastAggregator
from modules.neighbor_loader import LastNeighborLoader
from modules.NCNDecoder.NCNPred import NCNPredictor
import modules.NCNDecoder.NCNPred as predmod


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def emit(kind, **kw):
    print(json.dumps(dict(kind=kind, **kw)), flush=True)


class Clock:
    def __init__(self, enabled=False):
        self.enabled, self.rows = enabled, []

    @contextlib.contextmanager
    def phase(self, name):
        if not self.enabled:
            yield
            return
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.perf_counter_ns()
        a.record()
        with torch.profiler.record_function(name):
            yield
        b.record()
        self.rows.append((name, (time.perf_counter_ns()-start)/1e6, a, b))

    def result(self):
        result = {}
        for name, host, a, b in self.rows:
            r = result.setdefault(name, dict(host_ms=0., timeline_ms=0., calls=0))
            r['host_ms'] += host
            r['timeline_ms'] += a.elapsed_time(b)
            r['calls'] += 1
        return result


def source_coefficients():
    tree = ast.parse((SRC/'modules/NCNDecoder/NCNPred.py').read_text())
    cl = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'NCNPredictor')
    fn = copy.deepcopy(next(x for x in cl.body if isinstance(x, ast.FunctionDef) and x.name == 'get_cn_emb'))
    fn.name = 'coefficients'
    # Stop before aggregation; every preceding source statement/mutation stays literal.
    branch = fn.body
    while True:
        cond = next((x for x in branch if isinstance(x, ast.If) and
                     'NCN_mode == 2' in ast.unparse(x.test)), None)
        if cond is not None:
            break
        branch = next(x for x in branch if isinstance(x, ast.If) and
                      'NCN_mode ==' in ast.unparse(x.test)).orelse
    at = next(i for i, x in enumerate(cond.body) if isinstance(x, ast.Assign) and
              ast.unparse(x.targets[0]).startswith('(xcn_0_1'))
    cond.body[at:] = ast.parse('return torch.stack([c.to_dense() for c in [cn_0_1,cn_1_0,cn_1_1,cn_1_2,cn_2_1,cn_2_2,special_2_2]])').body
    ns = dict(vars(predmod))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), 'source_coefficients', 'exec'), ns)
    return ns['coefficients']


COEFFICIENTS = source_coefficients()


def instrument_source(model, clock):
    """Statement groups in the original mode-2 body; profiling replay only."""
    tree = ast.parse((SRC/'modules/NCNDecoder/NCNPred.py').read_text())
    cl = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'NCNPredictor')
    fn = copy.deepcopy(next(x for x in cl.body if isinstance(x, ast.FunctionDef) and x.name == 'get_cn_emb'))
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and ast.unparse(node.test) == 'NCN_mode == 2':
            body = []
            for stmt in node.body:
                target = ast.unparse(stmt.targets[0]) if isinstance(stmt, ast.Assign) else ''
                name = 'source_coefficients_masks'
                if target in ('adj0', 'adj1'):
                    name = 'source_graph'
                elif target == 'adj2':
                    name = 'source_Q_A_squared'
                elif target == 'k3cycle':
                    name = 'source_A_cubed'
                elif target == 'special_2_2':
                    name = 'source_weighted_S'
                elif target.startswith('(xcn_0_1') or target == 'special_xcn_2_2':
                    name = 'source_seven_aggregation'
                context = ast.parse(f"_profile_clock.phase('{name}')", mode='eval').body
                body.append(ast.With(items=[ast.withitem(context_expr=context)], body=[stmt]))
            node.body = body
            break
    ns = dict(vars(predmod), _profile_clock=clock)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), 'instrumented_source', 'exec'), ns)
    model.get_cn_emb = types.MethodType(ns['get_cn_emb'], model)


def get_data(name):
    if name == 'wikipedia':
        path = ROOT/'experiments/temporal_tc_20260905_cooccurrence_gate0/data/wikipedia20k.npz'
        with np.load(path) as z:
            src, dst = z['src'].copy(), z['dst'].copy()
            times, msgs = z['time'].copy(), z['edge'][1:].copy()
        info = dict(path=str(path), sha256=sha(path), message_features='172 real features',
                    id_mapping='existing 20k archive; disjoint source/destination IDs')
    else:
        path = ROOT/'experiments/temporal_tc_20260905_joint_consumer_gate0/data/CollegeMsg.txt.gz'
        with gzip.open(path, 'rt') as f:
            raw = np.loadtxt(f, dtype=np.int64)
        raw = raw[np.argsort(raw[:, 2], kind='stable')][:20000]
        src, dst = raw[:, 0], raw[:, 1]
        times = raw[:, 2] - raw[0, 2]
        msgs = np.zeros((len(src), 1), np.float32)
        info = dict(path=str(path), sha256=sha(path), message_features='featureless; one zero channel',
                    id_mapping='original positive node IDs; time shifted to zero')
    assert len(src) == 20000 and np.all(times[1:] >= times[:-1])
    info.update(events=len(src), num_nodes=int(max(src.max(), dst.max()))+1,
                min_dst=int(dst.min()), max_dst=int(dst.max()), message_dim=msgs.shape[1],
                time_dtype='int64, matching TGB TemporalData convention; fractional timestamps truncated')
    tensors = [torch.as_tensor(a, device=DEVICE, dtype=d) for a, d in
               zip([src, dst, times, msgs], [torch.long, torch.long, torch.long, torch.float32])]
    return tensors, info


class Trainer:
    def __init__(self, name, bs):
        self.data, self.info = get_data(name)
        self.name, self.bs = name, bs
        self.n = self.info['num_nodes']
        torch.manual_seed(20260908)
        mem = TGNMemory(self.n, self.info['message_dim'], 100, 100,
                        IdentityMessage(self.info['message_dim'], 100, 100),
                        LastAggregator(), t_enc_grad=True).to(DEVICE)
        gnn = GraphAttentionEmbedding(100, 100, self.info['message_dim'], mem.time_enc).to(DEVICE)
        pred = NCNPredictor(100, 256, 1, 2).to(DEVICE)
        self.model = torch.nn.ModuleDict(dict(memory=mem, gnn=gnn, pred=pred))
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.neighbor = LastNeighborLoader(self.n, size=10, device=DEVICE)
        self.assoc = torch.empty(self.n, dtype=torch.long, device=DEVICE)
        self.clock = Clock()

    def checkpoint(self):
        return dict(model=copy.deepcopy(self.model.state_dict()),
                    optimizer=copy.deepcopy(self.optimizer.state_dict()),
                    memory=self.model['memory'].backup_memory(),
                    neighbor=copy.deepcopy(self.neighbor.__dict__),
                    rng=torch.get_rng_state().clone(), cuda_rng=torch.cuda.get_rng_state().clone() if DEVICE == 'cuda' else None)

    def restore(self, s):
        self.model.load_state_dict(s['model'])
        self.optimizer.load_state_dict(copy.deepcopy(s['optimizer']))
        self.model['memory'].restore_memory(s['memory'])
        self.neighbor.__dict__.update(copy.deepcopy(s['neighbor']))
        torch.set_rng_state(s['rng'])
        if s['cuda_rng'] is not None:
            torch.cuda.set_rng_state(s['cuda_rng'])

    def state_for_check(self):
        out = {}
        for name, p in self.model.named_parameters():
            out['parameter/'+name] = p.detach().cpu().clone()
            out['gradient/'+name] = None if p.grad is None else p.grad.detach().cpu().clone()
            for k, v in self.optimizer.state.get(p, {}).items():
                out['adam/'+name+'/'+k] = v.detach().cpu().clone() if torch.is_tensor(v) else v
        out['memory'] = self.model['memory'].memory.detach().cpu().clone()
        out['last_update'] = self.model['memory'].last_update.cpu().clone()
        out['neighbor_eid'] = self.neighbor.e_id.cpu().clone()
        # Ignore uninitialized neighbor slots; only e_id>=0 is meaningful.
        valid = self.neighbor.e_id >= 0
        out['neighbor_ids'] = self.neighbor.neighbors[valid].cpu().clone()
        return out

    def step(self, index, variant='keeper', cached=None, capture=False, check=False):
        phase = self.clock.phase
        with phase('batch_and_zero_grad'):
            lo, hi = index*self.bs, (index+1)*self.bs
            src, dst, t, msg = (a[lo:hi] for a in self.data)
            self.optimizer.zero_grad()
        with phase('negative_and_neighbor_sampling'):
            neg = torch.randint(self.info['min_dst'], self.info['max_dst']+1,
                                (len(src),), device=DEVICE, dtype=torch.long)
            ids = torch.cat([src, dst, neg]).unique()
            ids, _, _ = self.neighbor(ids)
            ids, edges, eid = self.neighbor(ids)
            self.assoc[ids] = torch.arange(len(ids), device=DEVICE)
        with phase('memory_read'):
            z, last = self.model['memory'](ids)
        with phase('gnn_embedding'):
            z = self.model['gnn'](z, last, edges, self.data[2][eid], self.data[3][eid])
        with phase('query_mapping'):
            q = torch.stack([torch.cat([self.assoc[src], self.assoc[src]]),
                             torch.cat([self.assoc[dst], self.assoc[neg]])])
        if check and cached is not None:
            assert torch.equal(q, cached['q']) and torch.equal(edges, cached['edges'])
            assert torch.equal(ids, cached['ids'])
        if z.requires_grad and check:
            z.retain_grad()
        plans, graph = [], None
        if variant == 'source':
            with phase('source_decoder'):
                pos = self.model['pred'](z, edges, q[:, :self.bs], 2)
                negout = self.model['pred'](z, edges, q[:, self.bs:], 2)
        else:
            assert len(ids) <= 2048, ('native_shape_not_admitted', self.name, self.bs, index, len(ids))
            with phase('graph_build'):
                graph = cached['graph'] if variant in ('free_graph', 'free_relations') else builder.graph(len(ids), edges, checked=True)
            with phase('relation_Q_S_coefficients'):
                if variant == 'free_relations':
                    plans = cached['plans']
                else:
                    plans = [ref.from_graph(len(ids), q[:, j:j+64], graph) for j in range(0, q.shape[1], 64)]
            with phase('seven_channel_aggregation'):
                y = torch.cat([p.consume(z, library=True) for p in plans], dim=0)
            with phase('decoder_mlp'):
                features = torch.cat([z[q[0]]*z[q[1]], y], dim=-1)
                pos = self.model['pred'].xsmlp(features[:self.bs])
                negout = self.model['pred'].xsmlp(features[self.bs:])
        with phase('loss'):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
            loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(negout, torch.zeros_like(negout))
        with phase('memory_update'):
            self.model['memory'].update_state(src, dst, t, msg)
        with phase('neighbor_insert'):
            self.neighbor.insert(src, dst)
        with phase('backward'):
            loss.backward()
        with phase('optimizer'):
            self.optimizer.step()
        with phase('detach_and_loss_read'):
            self.model['memory'].detach()
            scalar = float(loss)
        if not np.isfinite(scalar):
            raise RuntimeError(('nonfinite_loss', index))
        result = dict(index=index, loss=scalar, n=len(ids), edges=edges.shape[1])
        if capture:
            result['cache'] = dict(graph=graph, plans=plans, q=q.clone(), edges=edges.clone(), ids=ids.clone())
        if check:
            result['check'] = self.state_for_check()
            result['check'].update(logits=torch.cat([pos, negout]).detach().cpu(),
                                   loss=loss.detach().cpu(), embedding=z.detach().cpu(), dx=z.grad.detach().cpu())
        return result


def compare(a, b):
    assert a.keys() == b.keys()
    failures, worst = [], []
    for key in a:
        x, y = a[key], b[key]
        if x is None or y is None:
            if x is not None or y is not None:
                failures.append(dict(key=key, reason='None mismatch'))
            continue
        if not torch.is_tensor(x):
            if x != y:
                failures.append(dict(key=key, reason='scalar mismatch'))
            continue
        assert x.shape == y.shape
        if x.is_floating_point():
            diff = (x-y).abs()
            ok = torch.isfinite(x) & torch.isfinite(y) & (diff <= 2e-4+2e-4*y.abs())
            err = float(diff.max()) if diff.numel() else 0.
            worst.append((err, key))
        else:
            ok = x == y
        if not bool(ok.all()):
            failures.append(dict(key=key, count=int((~ok).sum()), max_abs=float((x-y).abs().max())))
    return dict(pass_all=not failures, failures=failures, worst=sorted(worst, reverse=True)[:8], fields=len(a))


def exact_coeff_check(trainer, cache):
    n, q, edges = len(cache['ids']), cache['q'], cache['edges']
    x = torch.zeros((n, 1), device=DEVICE)
    expected = COEFFICIENTS(trainer.model['pred'], x, edges, q, 2)
    actual = torch.cat([torch.sparse_csr_tensor(p.rp, p.col, p.iv,
                       size=(p.m, p.n)).to_dense().reshape(7, p.b, p.n)
                       for p in cache['plans']], dim=1)
    assert torch.equal(actual, expected), 'integer coefficient mismatch'
    special = actual[6]
    return dict(n=n, b=q.shape[1], C_nnz=int((actual != 0).sum()),
                nnz_channels=[int((a != 0).sum()) for a in actual],
                max_abs=int(actual.abs().max()), S_nnz=int((special != 0).sum()),
                S_sum=int(special.sum()), S_max=int(special.max()),
                exact=True, csr_bytes=sum(8*(p.m+1)+16*p.nnz for p in cache['plans']),
                graph_bytes=sum(a.numel()*a.element_size() for a in cache['graph']))


def run_window(tr, start, out, length=8, rounds=3):
    label = f'{tr.name}_b{tr.bs}_s{start}'
    checkpoint = tr.checkpoint()
    torch.save(checkpoint, out/(label+'_checkpoint.pt'))
    captures, anchors = [], []
    for idx in range(start, start+length):
        result = tr.step(idx, capture=True, check=True)
        captures.append(result['cache'])
        anchors.append(result['check'])
    canonical_end = tr.checkpoint()
    coeff = [exact_coeff_check(tr, c) for c in captures]
    emit('coefficient_checks', case=label, steps=coeff)
    checks = []
    for variant in ['keeper', 'source', 'free_graph', 'free_relations']:
        tr.restore(checkpoint)
        results = []
        for j, idx in enumerate(range(start, start+length)):
            r = tr.step(idx, variant, captures[j], check=True)
            cmp = compare(r['check'], anchors[j])
            checks.append(dict(variant=variant, index=idx, **cmp))
            results.append(r['check'])
        torch.save(results, out/(label+'_'+variant+'_replay_checks.pt'))
    torch.save(anchors, out/(label+'_keeper_checks.pt'))
    accepted = all(x['pass_all'] for x in checks)
    emit('training_checks', case=label, pass_all=accepted, checks=checks)
    # Numeric failures remain evidence and timing is diagnostic, never qualified speedup.
    variants = ['keeper', 'free_graph', 'free_relations']
    for rnd in range(rounds):
        order = variants[rnd:] + variants[:rnd]
        for variant in order:
            tr.restore(checkpoint)
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            begin = time.perf_counter_ns()
            losses = []
            for j, idx in enumerate(range(start, start+length)):
                losses.append(tr.step(idx, variant, captures[j])['loss'])
            b.record()
            torch.cuda.synchronize()
            wall = (time.perf_counter_ns()-begin)/1e6
            emit('timing', case=label, variant=variant, round=rnd, order=order,
                 steps=length, wall_ms=wall, timeline_ms=a.elapsed_time(b),
                 loss_first=losses[0], loss_last=losses[-1], numeric_checks_pass=accepted)
    for variant in ['keeper', 'source']:
        tr.restore(checkpoint)
        tr.clock = Clock(enabled=True)
        if variant == 'source':
            instrument_source(tr.model['pred'], tr.clock)
        torch.cuda.synchronize()
        begin = time.perf_counter_ns()
        for j, idx in enumerate(range(start, start+length)):
            tr.step(idx, variant, captures[j])
        torch.cuda.synchronize()
        wall = (time.perf_counter_ns()-begin)/1e6
        emit('phases', case=label, variant=variant, steps=length,
             wall_ms=wall, phases=tr.clock.result())
        tr.clock = Clock()
        if variant == 'source':
            del tr.model['pred'].get_cn_emb
    tr.restore(canonical_end)
    ref.guard()
    return start+length


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--dataset', choices=['wikipedia', 'college', 'both'], default='both')
    p.add_argument('--batch', choices=['32', '200', 'both'], default='both')
    args = p.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ref.guard(initial=True)
    torch.zeros(1, device=DEVICE)
    ref.guard()
    files = [Path(__file__), EXP/'src/CONTRACT.md',
             Path(os.environ['GB_LIB']), Path(os.environ['GPU_NT_LIB']),
             SRC/'examples/linkproppred/tgbl-dataset/TNCN.py']
    files += sorted((SRC/'modules').rglob('*.py'))
    manifest = {str(p): sha(p) for p in files}
    (args.output/'provenance.json').write_text(json.dumps(manifest, indent=2)+'\n')
    emit('opening', torch=torch.__version__, cuda=torch.version.cuda,
         pyg=torch_geometric.__version__, sparse=torch_sparse.__version__,
         scatter=torch_scatter.__version__, gpu=torch.cuda.get_device_name(), pid=os.getpid())
    datasets = ['wikipedia', 'college'] if args.dataset == 'both' else [args.dataset]
    batches = [32, 200] if args.batch == 'both' else [int(args.batch)]
    for dataset in datasets:
        for bs in batches:
            tr = Trainer(dataset, bs)
            emit('data', dataset=dataset, batch=bs, **tr.info)
            starts = [4] if args.smoke else ([128, 256, 512] if bs == 32 else [20, 40, 80])
            i = 0
            for start in starts:
                while i < start:
                    r = tr.step(i)
                    i += 1
                    if i % 64 == 0:
                        emit('history_progress', dataset=dataset, batch=bs, step=i, loss=r['loss'], n=r['n'])
                        ref.guard()
                i = run_window(tr, start, args.output, 2 if args.smoke else 8, 1 if args.smoke else 3)
            emit('workload_complete', dataset=dataset, batch=bs, trained_steps=i,
                 trained_events=i*bs, peak_memory_bytes=torch.cuda.max_memory_allocated())
            del tr
            torch.cuda.empty_cache()
    emit('complete', apps=ref.guard())


if __name__ == '__main__':
    main()
