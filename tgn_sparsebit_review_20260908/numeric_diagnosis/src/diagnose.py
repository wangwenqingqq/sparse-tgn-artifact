"""Causal numeric probes using immutable real-training checkpoints."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import types

OLD = Path('/home/data/wangxuran/tncn_training_profile_20260908')
sys.path.insert(0, str(OLD/'src'))
import profile_training as p
torch = p.torch
ART = OLD/'output/full_v5/artifacts'
FREQ = 'parameter/memory.time_enc.lin.weight'


def metric(x, y):
    x, y = x.detach().cpu().double(), y.detach().cpu().double()
    d = (x-y).abs()
    return dict(max_abs=float(d.max()) if d.numel() else 0.,
                rms=float(d.square().mean().sqrt()) if d.numel() else 0.,
                unequal=int((d != 0).sum()), elements=d.numel(),
                fails=int((d > 2e-4+2e-4*y.abs()).sum()))


def brief(a, b):
    return {k: metric(a[k], b[k]) for k in ['embedding', 'memory', 'logits', 'loss', 'dx', FREQ,
        'gradient/memory.time_enc.lin.weight']}


def dense_plans(plans):
    return torch.cat([torch.sparse_csr_tensor(x.rp, x.col, x.iv,
        size=(x.m, x.n)).to_dense().reshape(7, x.b, x.n) for x in plans], dim=1)


def install_compatible(tr):
    """Integer producer, original seven sparse sums and decoder autograd DAG."""
    pred = tr.model['pred']
    pred._paid_graph = None
    def get_cn(self, x, edges, q, mode, decay=False, time_info=None):
        assert mode == 2 and not decay
        if self._paid_graph is None:
            self._paid_graph = p.builder.graph(len(x), edges, checked=True)
        plans = [p.ref.from_graph(len(x), q[:, j:j+64], self._paid_graph)
                 for j in range(0, q.shape[1], 64)]
        coeff = dense_plans(plans).float()
        return torch.cat([p.predmod.spmm_add(p.torch_sparse.SparseTensor.from_dense(c), x)
                          for c in coeff], dim=-1)
    pred.get_cn_emb = types.MethodType(get_cn, pred)


def install_time64(tr):
    enc = tr.model['memory'].time_enc
    ids = [id(x) for x in enc.parameters()]
    enc.lin.double()
    assert ids == [id(x) for x in enc.parameters()]
    assert enc is tr.model['gnn'].time_enc
    def forward(self, t):
        return self.lin(t.view(-1, 1).double()).cos().to(t.dtype)
    enc.forward = types.MethodType(forward, enc)


def step(tr, index, mode='keeper', **kwargs):
    if mode == 'compatible':
        tr.model['pred']._paid_graph = None
        mode = 'source'
    return tr.step(index, mode, **kwargs)


def trace_time(tr):
    rows = []
    def hook(enc, inputs, output):
        rows.append(dict(t=inputs[0].detach().cpu().clone(), y=output.detach().cpu().clone(),
                         w=enc.lin.weight.detach().cpu().clone(), b=enc.lin.bias.detach().cpu().clone()))
    return rows, tr.model['memory'].time_enc.register_forward_hook(hook)


def isolated(tr, before, actual, out, label):
    """Source, keeper and dense FP64 at identical real FP32 inputs."""
    cache, realx = actual['cache'], actual['check']['embedding'].to('cuda')
    q, edges = cache['q'], cache['edges']
    coeff = dense_plans(cache['plans'])
    expected = p.COEFFICIENTS(tr.model['pred'], realx, edges, q, 2)
    assert torch.equal(coeff, expected)
    raw, summary = dict(x=realx.cpu(), q=q.cpu(), C=coeff.cpu()), {}
    models = {}
    for mode in ['source', 'keeper', 'dense64']:
        model = copy.deepcopy(tr.model['pred'])
        model.load_state_dict({k[len('pred.'):]: v for k, v in before['model'].items() if k.startswith('pred.')})
        if mode == 'dense64':
            model.double()
        models[mode] = model
        x = realx.detach().to(torch.float64 if mode == 'dense64' else torch.float32).requires_grad_()
        if mode == 'source':
            ys = [model.get_cn_emb(x, edges, qq, 2) for qq in q.split(tr.bs, dim=1)]
        elif mode == 'keeper':
            y = torch.cat([plan.consume(x, library=True) for plan in cache['plans']], dim=0)
            ys = list(y.split(tr.bs, dim=0))
        else:
            y = (coeff.reshape(-1, len(x)).double() @ x).reshape(7, q.shape[1], -1).permute(1, 0, 2).reshape(q.shape[1], -1)
            ys = list(y.split(tr.bs, dim=0))
        for y in ys:
            y.retain_grad()
        if mode == 'keeper':
            # Preserve the combined endpoint-product DAG of the deployed keeper.
            features = torch.cat([x[q[0]]*x[q[1]], torch.cat(ys)], dim=-1)
            logits = [model.xsmlp(f) for f in features.split(tr.bs)]
        else:
            logits = [model.xsmlp(torch.cat([x[qq[0]]*x[qq[1]], y], dim=-1))
                      for qq, y in zip(q.split(tr.bs, dim=1), ys)]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[0], torch.ones_like(logits[0]))
        loss += torch.nn.functional.binary_cross_entropy_with_logits(logits[1], torch.zeros_like(logits[1]))
        loss.backward()
        row = dict(y=torch.cat(ys).detach().cpu(), logits=torch.cat(logits).detach().cpu(),
                   loss=loss.detach().cpu(), dx=x.grad.detach().cpu(), dy=torch.cat([y.grad for y in ys]).cpu())
        row.update({'grad/'+n: v.grad.detach().cpu() for n, v in model.named_parameters() if v.grad is not None})
        raw[mode] = row
    # Common rounded FP32 upstream removes downstream/forward-gradient variation.
    dy = raw['dense64']['dy'].float().to('cuda')
    for mode in ['source', 'keeper', 'dense64']:
        x = realx.detach().to(torch.float64 if mode == 'dense64' else torch.float32).requires_grad_()
        if mode == 'source':
            y = torch.cat([models[mode].get_cn_emb(x, edges, qq, 2) for qq in q.split(tr.bs, dim=1)])
        elif mode == 'keeper':
            y = torch.cat([plan.consume(x, library=True) for plan in cache['plans']])
        else:
            y = (coeff.reshape(-1, len(x)).double() @ x).reshape(7, q.shape[1], -1).permute(1, 0, 2).reshape(q.shape[1], -1)
        y.backward(dy.to(x.dtype))
        raw[mode]['common_vjp'] = x.grad.detach().cpu()
    for pair in [('source', 'keeper'), ('source', 'dense64'), ('keeper', 'dense64')]:
        summary['/'.join(pair)] = {k: metric(raw[pair[0]][k], raw[pair[1]][k]) for k in raw[pair[0]]}
    raw['common_dy'] = dy.cpu()
    raw['predictor_before'] = {k: v.cpu() for k, v in before['model'].items() if k.startswith('pred.')}
    torch.save(raw, out/(label+'_isolated.pt'))
    p.emit('isolated', case=label, exact_integer_C=True, comparisons=summary)


def witnesses(out):
    for name in ['wikipedia', 'college']:
        label, start, bs = f'{name}_b200_s20', 20, 200
        ck = torch.load(ART/(label+'_checkpoint.pt'), weights_only=False)
        old = torch.load(ART/(label+'_keeper_checks.pt'), weights_only=False)
        srcold = torch.load(ART/(label+'_source_replay_checks.pt'), weights_only=False)
        tr = p.Trainer(name, bs)
        tr.restore(ck)
        before, canon, caches = [], [], []
        for j in range(8):
            before.append(tr.checkpoint())
            r = step(tr, start+j, capture=True, check=True)
            cmp = p.compare(r['check'], old[j])
            assert cmp['pass_all'], ('canonical_changed', label, j, cmp)
            canon.append(r['check']); caches.append(r['cache'])
        p.emit('canonical', case=label, passed=8, total=8)
        for j in [0, 7]:
            isolated(tr, before[j], dict(cache=caches[j], check=canon[j]), out, label+f'_i{start+j}')
        # Source replay; first archived frequency difference is selected before probes.
        variants = {}
        for mode in ['source', 'source_anchor_frequency', 'source_anchor_mlp', 'compatible']:
            test = p.Trainer(name, bs)
            if mode == 'compatible':
                install_compatible(test)
            test.restore(ck)
            rows, checks = [], []
            for j in range(8):
                if mode == 'source_anchor_frequency':
                    with torch.no_grad():
                        test.model['memory'].time_enc.lin.weight.copy_(before[j]['model']['memory.time_enc.lin.weight'])
                elif mode == 'source_anchor_mlp':
                    with torch.no_grad():
                        for n, param in test.model['pred'].named_parameters():
                            param.copy_(before[j]['model']['pred.'+n])
                r = step(test, start+j, 'compatible' if mode == 'compatible' else 'source', check=True)
                rows.append(r['check'])
                checks.append(dict(index=start+j, **p.compare(r['check'], canon[j]), metrics=brief(r['check'], canon[j])))
            variants[mode] = rows
            torch.save(rows, out/(label+'_'+mode+'_checks.pt'))
            p.emit('witness_trajectory', case=label, variant=mode, checks=checks)
            del test
        # Transplant only one parameter tensor into the canonical next-step state.
        j = next(j for j in range(7) if not torch.equal(srcold[j][FREQ], canon[j][FREQ]))
        probes, logs = {}, {}
        for mode in ['baseline', 'source_frequency_only']:
            test = p.Trainer(name, bs)
            test.restore(before[j+1])
            if mode != 'baseline':
                with torch.no_grad():
                    test.model['memory'].time_enc.lin.weight.copy_(srcold[j][FREQ])
            trace, handle = trace_time(test)
            probes[mode] = step(test, start+j+1, check=True)['check']
            handle.remove(); logs[mode] = trace
        assert p.compare(probes['baseline'], canon[j+1])['pass_all']
        deltas = []
        for a, b in zip(logs['baseline'], logs['source_frequency_only']):
            assert torch.equal(a['t'], b['t'])
            phase_diff = a['t'].double().reshape(-1, 1) @ (b['w']-a['w']).double().T
            deltas.append(dict(input_abs_max=float(a['t'].abs().max()) if a['t'].numel() else 0.,
                               phase_diff_abs_max=float(phase_diff.abs().max()) if phase_diff.numel() else 0.,
                               encoding=metric(b['y'], a['y'])))
        torch.save(dict(checks=probes, traces=logs), out/(label+'_transplant.pt'))
        p.emit('transplant', case=label, source_update_index=start+j, probe_index=start+j+1,
               weight_delta=metric(srcold[j][FREQ], canon[j][FREQ]),
               metrics=brief(probes['source_frequency_only'], probes['baseline']), calls=deltas)
        # FP64 is explicitly a changed precision mode, compared against itself.
        fp = {}
        for mode in ['keeper', 'source']:
            test = p.Trainer(name, bs)
            install_time64(test); test.restore(ck)
            enc = test.model['memory'].time_enc
            assert enc.lin.weight.dtype == torch.float64
            assert test.optimizer.state[enc.lin.weight]['exp_avg'].dtype == torch.float64
            fp[mode] = [step(test, start+j, mode, check=True)['check'] for j in range(8)]
        torch.save(fp, out/(label+'_time64.pt'))
        p.emit('time64_witness', case=label, checks=[dict(index=start+j, **p.compare(fp['source'][j], fp['keeper'][j]),
            metrics=brief(fp['source'][j], fp['keeper'][j])) for j in range(8)])
        p.ref.guard()


def qualify(out, selected):
    for name in ['wikipedia', 'college']:
        for bs in [32, 200]:
            for start in ([128, 256, 512] if bs == 32 else [20, 40, 80]):
                label = f'{name}_b{bs}_s{start}'
                ck = torch.load(ART/(label+'_checkpoint.pt'), weights_only=False)
                original_source = torch.load(ART/(label+'_source_replay_checks.pt'), weights_only=False)
                variants = {}
                for mode in selected.split(','):
                    modes = ['source', 'keeper'] if mode == 'time64' else ['compatible']
                    for arm in modes:
                        tr = p.Trainer(name, bs)
                        if mode == 'time64':
                            install_time64(tr)
                        else:
                            install_compatible(tr)
                        tr.restore(ck)
                        rows = [step(tr, start+j, arm, check=True)['check'] for j in range(8)]
                        variants[mode+'/'+arm] = rows
                        del tr
                    base, other = (variants['time64/source'], variants['time64/keeper']) if mode == 'time64' else (original_source, variants['compatible/compatible'])
                    comparisons = [dict(index=start+j, **p.compare(other[j], base[j]), metrics=brief(other[j], base[j])) for j in range(8)]
                    p.emit('qualification', case=label, mode=mode, checks=comparisons)
                torch.save(variants, out/(label+'_qualification.pt'))
                p.ref.guard()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--task', choices=['witnesses', 'qualify'], default='witnesses')
    ap.add_argument('--selected', default='compatible,time64')
    args = ap.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p.ref.guard(initial=True)
    torch.zeros(1, device='cuda')
    p.ref.guard()
    p.emit('opening', torch=torch.__version__, gpu=torch.cuda.get_device_name(), args=vars(args)|{'output':str(args.output)},
           script_sha256=p.sha(__file__), old_runner_sha256=p.sha(OLD/'src/profile_training.py'))
    if args.task == 'witnesses':
        witnesses(args.output)
    else:
        qualify(args.output, args.selected)
    p.emit('complete', apps=p.ref.guard())


if __name__ == '__main__':
    main()
