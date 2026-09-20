"""Frozen TNCN teacher, conditional flow state distillation, and paid replay."""
import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, '/home/data/wangxuran/tncn_memory_profile_20260909/src')
import instrument as inst
p, torch, np = inst.p, inst.torch, inst.p.np

ROOT = Path(__file__).resolve().parents[1]
TRAIN, VALID, END, BS = 12000, 16000, 20000, 32
SEEDS = [1701, 1702, 1703]
CHUNKS = [128, 512]


def emit(kind, **kw):
    print(json.dumps(dict(kind=kind, **kw)), flush=True)


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(cpu(v) for v in obj)
    return copy.deepcopy(obj)


def load(path, device='cuda'):
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and 'rng' in obj:
        obj['rng'] = obj['rng'].cpu()
        if obj.get('cuda_rng') is not None:
            obj['cuda_rng'] = obj['cuda_rng'].cpu()
    return obj


def metric(logits, labels):
    """Independent tie-aware binary ranking metrics, using NumPy only."""
    scores = np.asarray(logits, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    assert np.isfinite(scores).all() and (y == 1).any() and (y == 0).any()
    order = np.argsort(-scores, kind='stable')
    s, y = scores[order], y[order]
    cuts = np.r_[np.flatnonzero(np.diff(s)), len(s)-1]
    tp = np.cumsum(y)[cuts].astype(float)
    fp = (cuts+1)-tp
    recall = tp/tp[-1]
    ap = np.sum(np.diff(np.r_[0., recall])*(tp/(tp+fp)))
    tpr, fpr = np.r_[0., recall], np.r_[0., fp/fp[-1]]
    auc = np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])/2)
    loss = np.mean(np.logaddexp(0., s)-y*s)
    return dict(ap=float(ap), auc=float(auc), bce=float(loss), examples=len(y))


def make_tr(name, weights=None):
    tr = p.Trainer(name, BS)
    inst.prepare(tr, 'batch_store', 'none')
    if weights is not None:
        tr.model.load_state_dict(weights)
        with torch.no_grad():
            tr.model.eval()
        tr.model['memory'].reset_state()
        tr.neighbor.reset_state()
        tr.model.requires_grad_(False)
    gen = torch.Generator(device='cuda').manual_seed(61711)
    tr.fixed_neg = torch.randint(tr.info['min_dst'], tr.info['max_dst']+1,
                                (END,), device='cuda', generator=gen)
    return tr


@torch.no_grad()
def exact_replay(tr, lo, hi):
    mem = tr.model['memory']
    for pos in range(lo, hi, BS):
        src, dst, t, msg = (a[pos:min(pos+BS, hi)] for a in tr.data)
        mem.update_state(src, dst, t, msg)
        tr.neighbor.insert(src, dst)


@torch.no_grad()
def predict(tr, lo, hi):
    src, dst = tr.data[0][lo:hi], tr.data[1][lo:hi]
    neg = tr.fixed_neg[lo:hi]
    ids = torch.cat([src, dst, neg]).unique()
    ids, _, _ = tr.neighbor(ids)
    ids, edges, eid = tr.neighbor(ids)
    tr.assoc[ids] = torch.arange(len(ids), device='cuda')
    z, last = tr.model['memory'](ids)
    z = tr.model['gnn'](z, last, edges, tr.data[2][eid], tr.data[3][eid])
    q = torch.stack([torch.cat([tr.assoc[src], tr.assoc[src]]),
                     torch.cat([tr.assoc[dst], tr.assoc[neg]])])
    tr.model['pred']._paid_graph = None
    out = torch.cat([tr.model['pred'](z, edges, q[:, :len(src)], 2),
                     tr.model['pred'](z, edges, q[:, len(src):], 2)]).flatten()
    y = torch.cat([torch.ones(len(src), device='cuda'), torch.zeros(len(src), device='cuda')])
    return out, y


@torch.no_grad()
def score_suffix(tr, lo, hi):
    logits, labels = [], []
    for pos in range(lo, hi, BS):
        a, b = predict(tr, pos, min(pos+BS, hi))
        logits.append(a.cpu()); labels.append(b.cpu())
        exact_replay(tr, pos, min(pos+BS, hi))
    a, b = torch.cat(logits), torch.cat(labels)
    return metric(a.numpy(), b.numpy()), dict(logits=a, labels=b)


def teacher_stage(out, names, epochs):
    for name in names:
        begin = time.perf_counter()
        tr = make_tr(name)
        best, rows, state = -1., [], None
        emit('teacher_start', dataset=name, info=tr.info)
        for epoch in range(epochs):
            tr.model.train(); tr.model['memory'].reset_state(); tr.neighbor.reset_state()
            losses = []
            for j in range(TRAIN//BS):
                losses.append(inst.b.step(tr, j, 'direct')['loss'])
                if j % 100 == 0:
                    p.ref.guard()
            with torch.no_grad():
                tr.model.eval()
                stats, raw = score_suffix(tr, TRAIN, VALID)
            row = dict(epoch=epoch+1, train_loss=float(np.mean(losses)), validation=stats)
            rows.append(row); emit('teacher_epoch', dataset=name, **row)
            if stats['ap'] > best:
                best = stats['ap']; state = cpu(tr.model.state_dict())
                torch.save(dict(weights=state, epoch=epoch+1, validation=stats), out/(name+'_teacher.pt'))
            p.ref.guard()
        del tr; gc.collect()
        teacher = make_tr(name, state)
        exact_replay(teacher, 0, VALID)
        test, raw = score_suffix(teacher, VALID, END)
        torch.save(raw, out/(name+'_teacher_full_test.pt'))
        result = dict(epochs=rows, selected_ap=best, frozen_replay_test=test,
                      seconds=time.perf_counter()-begin, info=teacher.info)
        save_json(out/(name+'_teacher.json'), result)
        emit('teacher_done', dataset=name, **result)
        del teacher; gc.collect(); p.ref.guard()


@torch.no_grad()
def encode(tr, lo, hi):
    """No endpoint state or future query is accessible to this encoder."""
    src, dst, t, raw = (a[lo:hi] for a in tr.data)
    mem = tr.model['memory']
    nodes, inverse = torch.cat([src, dst]).unique(return_inverse=True)
    peer = torch.cat([dst, src])
    times = t.repeat(2)
    initial = mem.memory[nodes].clone()
    rel = (times-mem.last_update[torch.cat([src, dst])]).float()
    enc = mem.time_enc
    ix = torch.linspace(0, 99, 16, device='cuda').long()
    phase = rel[:, None]*enc.lin.weight[ix, 0]+enc.lin.bias[ix]
    ordinal = torch.arange(hi-lo, device='cuda').repeat(2)
    fraction = ordinal.float()/max(1, hi-lo-1)
    direction = torch.cat([torch.zeros(hi-lo, device='cuda'), torch.ones(hi-lo, device='cuda')])
    features = torch.cat([mem.memory[peer], raw.repeat(2, 1), phase.cos(),
                          torch.log1p(rel.clamp_min(0))[:, None], fraction[:, None], direction[:, None]], 1)
    bins = (ordinal*4//(hi-lo)).clamp_max(3)
    group = inverse*4+bins
    sums = features.new_zeros((len(nodes)*4, features.shape[1]))
    sums.index_add_(0, group, features)
    counts = torch.bincount(group, minlength=len(nodes)*4).float()
    means = sums/counts.clamp_min(1)[:, None]
    # Break timestamp/event ties by direction consistently, using only observed events.
    order = ordinal*2+direction.long()
    lastkey = torch.full((len(nodes),), -1, device='cuda', dtype=torch.long)
    lastkey.scatter_reduce_(0, inverse, order, reduce='amax', include_self=True)
    lastrow = torch.full_like(lastkey, -1)
    row = torch.arange(len(inverse), device='cuda')
    lastrow.scatter_reduce_(0, inverse, torch.where(order == lastkey[inverse], row, -1),
                            reduce='amax', include_self=True)
    condition = torch.cat([initial, means.reshape(len(nodes), -1),
                           counts.reshape(len(nodes), 4).log1p(), features[lastrow]], 1)
    return nodes, initial, condition


@torch.no_grad()
def metadata(tr, lo, hi):
    """Retain original per-minibatch message cache and neighbor insertion order."""
    mem = tr.model['memory']
    for pos in range(lo, hi, BS):
        src, dst, t, raw = (a[pos:min(pos+BS, hi)] for a in tr.data)
        mem._update_msg_store(src, dst, t, raw, mem.msg_s_store)
        mem._update_msg_store(dst, src, t, raw, mem.msg_d_store)
        tr.neighbor.insert(src, dst)
    src, dst, t = (a[lo:hi] for a in tr.data[:3])
    nodes = torch.cat([src, dst]).unique()
    latest = torch.zeros_like(mem.last_update)
    latest.scatter_reduce_(0, torch.cat([src, dst]), t.repeat(2), reduce='amax', include_self=True)
    return nodes, latest[nodes]


def discrete(tr):
    mem = tr.model['memory']
    result = dict(last=mem.last_update.cpu().clone(), eid=tr.neighbor.e_id.cpu().clone(),
                  neighbor=tr.neighbor.neighbors[tr.neighbor.e_id >= 0].cpu().clone(),
                  cur_e_id=tr.neighbor.cur_e_id)
    for direction in ['s', 'd']:
        store = getattr(mem, 'msg_'+direction+'_store')
        vals = [store[j] for j in range(tr.n)]
        result[direction+'_counts'] = torch.tensor([len(v[0]) for v in vals])
        for k in range(4):
            result[direction+'_'+str(k)] = torch.cat([v[k] for v in vals]).cpu().clone()
    return result


def byte_equal(a, b):
    assert a.keys() == b.keys()
    failed, count = [], 0
    for k in a:
        if torch.is_tensor(a[k]):
            x, y = a[k].contiguous(), b[k].contiguous()
            ok = x.dtype == y.dtype and x.shape == y.shape and torch.equal(x.view(torch.uint8), y.view(torch.uint8))
            count += x.numel()
        else:
            ok = a[k] == b[k]
        if not ok:
            failed.append(k)
    return dict(exact=not failed, failed=failed, elements=count)


def samples_stage(out, names):
    for name in names:
        weights = load(out/(name+'_teacher.pt'))['weights']
        for chunk in CHUNKS:
            begin = time.perf_counter(); tr = make_tr(name, weights)
            records = {}
            for split, lo, hi in [('train', 0, TRAIN), ('val', TRAIN, VALID), ('test', VALID, END)]:
                features, delta, starts, blockids, rows = [], [], [], [], []
                if split == 'test':
                    torch.save(cpu(tr.checkpoint()), out/f'{name}_k{chunk}_test_start.pt')
                for block, pos in enumerate(range(lo, hi-chunk+1, chunk)):
                    nodes, initial, condition = encode(tr, pos, pos+chunk)
                    exact_replay(tr, pos, pos+chunk)
                    target = tr.model['memory'].memory[nodes].clone()
                    features.append(condition.cpu()); delta.append((target-initial).cpu()); starts.append(initial.cpu())
                    blockids.append(torch.full((len(nodes),), block, dtype=torch.long))
                    rows.append(dict(lo=pos, hi=pos+chunk, nodes=len(nodes)))
                used = lo+len(rows)*chunk
                exact_replay(tr, used, hi)
                data = dict(condition=torch.cat(features), delta=torch.cat(delta), initial=torch.cat(starts),
                            block=torch.cat(blockids), windows=rows)
                assert torch.isfinite(data['condition']).all() and torch.isfinite(data['delta']).all()
                torch.save(data, out/f'{name}_k{chunk}_{split}.pt')
                records[split] = dict(rows=len(data['delta']), blocks=len(rows), omitted_tail=hi-used,
                                      condition_dim=data['condition'].shape[1], max_event=hi)
                p.ref.guard()
            save_json(out/f'{name}_k{chunk}_samples.json', dict(splits=records, seconds=time.perf_counter()-begin))
            emit('samples_done', dataset=name, chunk=chunk, splits=records, seconds=time.perf_counter()-begin)
            del tr; gc.collect()


class Net(torch.nn.Module):
    def __init__(self, condition_dim):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(condition_dim+101, 256), torch.nn.SiLU(),
                                          torch.nn.Linear(256, 256), torch.nn.SiLU(), torch.nn.Linear(256, 100))

    def forward(self, cond, z, t):
        return self.layers(torch.cat([cond, z, t], 1))


def integrate(net, cond, mode, steps):
    z = cond.new_zeros((len(cond), 100))
    if mode == 'mlp':
        return net(cond, z, cond.new_zeros((len(cond), 1)))
    for k in range(steps):
        z = z+net(cond, z, cond.new_full((len(cond), 1), k/steps))/steps
    return z


@torch.no_grad()
def validation_loss(net, condition, target, mode):
    error = 0.
    for lo in range(0, len(target), 2048):
        pred = integrate(net, condition[lo:lo+2048], mode, 1 if mode == 'mlp' else 2)
        error += float((pred-target[lo:lo+2048]).square().sum())
    return error/target.numel()


def fit_stage(out, names, steps):
    for name in names:
        for chunk in CHUNKS:
            train, val = [load(out/f'{name}_k{chunk}_{s}.pt') for s in ['train', 'val']]
            mean = train['condition'].mean(0)
            scale = train['condition'].std(0).clamp_min(.01)
            target_scale = train['delta'].square().mean(0).sqrt().clamp_min(.01)
            tx, ty = (train['condition']-mean)/scale, train['delta']/target_scale
            vx, vy = (val['condition']-mean)/scale, val['delta']/target_scale
            torch.save(cpu(dict(mean=mean, scale=scale, target_scale=target_scale)), out/f'{name}_k{chunk}_scaler.pt')
            for seed in SEEDS:
                for mode in ['mlp', 'fm']:
                    torch.manual_seed(seed)
                    net = Net(tx.shape[1]).cuda()
                    optim = torch.optim.Adam(net.parameters(), lr=.001)
                    # Same row sequence across objectives; FM times use a separate generator.
                    rows_rng = torch.Generator(device='cuda').manual_seed(seed+10)
                    time_rng = torch.Generator(device='cuda').manual_seed(seed+20)
                    begin = time.perf_counter(); best = math.inf; curve = []; best_step = 0
                    for step in range(1, steps+1):
                        rows = torch.randint(len(tx), (512,), device='cuda', generator=rows_rng)
                        x, y = tx[rows], ty[rows]
                        tau = torch.rand((len(rows), 1), device='cuda', generator=time_rng)
                        z = tau*y if mode == 'fm' else torch.zeros_like(y)
                        t = tau if mode == 'fm' else torch.zeros_like(tau)
                        loss = (net(x, z, t)-y).square().mean()
                        optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
                        if step % 100 == 0 or step == steps:
                            score = validation_loss(net, vx, vy, mode)
                            assert math.isfinite(score)
                            curve.append(dict(step=step, training_loss=float(loss), validation_mse=score))
                            if score < best:
                                best, best_step = score, step
                                torch.save(cpu(dict(weights=net.state_dict(), condition_dim=tx.shape[1],
                                                    mode=mode, seed=seed, step=step, validation_mse=score)),
                                           out/f'{name}_k{chunk}_{mode}_{seed}.pt')
                            p.ref.guard()
                    stats = dict(mode=mode, seed=seed, steps=steps, selected_step=best_step, validation_mse=best,
                                 parameters=sum(q.numel() for q in net.parameters()), seconds=time.perf_counter()-begin, curve=curve)
                    save_json(out/f'{name}_k{chunk}_{mode}_{seed}_fit.json', stats)
                    emit('fit_done', dataset=name, chunk=chunk, **{k:v for k,v in stats.items() if k != 'curve'})
            del train, val, tx, ty, vx, vy; gc.collect()


def candidates(out, name, chunk, controls=True):
    if controls:
        yield 'exact', None, None, None, None
        yield 'copy', None, None, None, None
        yield 'coarse', None, None, None, None
    scaler = load(out/f'{name}_k{chunk}_scaler.pt')
    for seed in SEEDS:
        for mode in ['mlp', 'fm']:
            data = load(out/f'{name}_k{chunk}_{mode}_{seed}.pt')
            net = Net(data['condition_dim']).cuda().eval()
            net.load_state_dict(data['weights']); net.requires_grad_(False)
            for steps in ([1] if mode == 'mlp' else [1, 2, 4]):
                yield f'{mode}{steps}_s{seed}', net, scaler, mode, steps


@torch.no_grad()
def candidate_replay(tr, lo, hi, variant, net, scaler, mode, steps):
    if variant == 'exact':
        exact_replay(tr, lo, hi); return
    mem = tr.model['memory']
    if net is not None:
        ids, initial, condition = encode(tr, lo, hi)
        normalized = (condition-scaler['mean'])/scaler['scale']
        delta = integrate(net, normalized, mode, steps)*scaler['target_scale']
    nodes, last = metadata(tr, lo, hi)
    if variant == 'coarse':
        mem._update_memory(nodes)
    elif net is not None:
        # encode and metadata use the same sorted union of observed endpoints.
        mem.memory[ids] = initial+delta
    mem.last_update[nodes] = last


def error_stats(actual, expected, initial=None):
    a, e = actual.double(), expected.double()
    sq = float((a-e).square().sum()); den = float(e.square().sum())
    row = dict(sse=sq, target_sq=den, elements=a.numel(), rmse=math.sqrt(sq/max(1,a.numel())),
               nrmse=math.sqrt(sq/max(den,1e-30)), max_abs=float((a-e).abs().max()))
    if initial is not None:
        row['delta_sq'] = float((e-initial.double()).square().sum())
        row['relative_delta_rmse'] = math.sqrt(sq/max(row['delta_sq'],1e-30))
    return row


@torch.no_grad()
def evaluate_stage(out, names):
    for name in names:
        weights = load(out/(name+'_teacher.pt'))['weights']
        for chunk in CHUNKS:
            begin = time.perf_counter()
            start = load(out/f'{name}_k{chunk}_test_start.pt')
            held = load(out/f'{name}_k{chunk}_test.pt')
            nblocks = (END-VALID)//chunk
            # Endpoint test has no feedback; all starts and features came from exact teacher replay.
            isolated = {}
            for variant, net, scaler, mode, steps in candidates(out, name, chunk, controls=False):
                pred = []
                for lo in range(0, len(held['delta']), 2048):
                    x = (held['condition'][lo:lo+2048]-scaler['mean'])/scaler['scale']
                    pred.append(integrate(net, x, mode, steps)*scaler['target_scale'])
                y = torch.cat(pred)
                isolated[variant] = error_stats(held['initial']+y, held['initial']+held['delta'], held['initial'])
            isolated['copy'] = error_stats(held['initial'], held['initial']+held['delta'], held['initial'])
            save_json(out/f'{name}_k{chunk}_isolated.json', isolated)
            references, summary = [], {}
            for variant, net, scaler, mode, steps in candidates(out, name, chunk):
                tr = make_tr(name, weights); tr.restore(start)
                rows, snapshots, raw_logits, raw_labels = [], [], [], []
                for block in range(nblocks):
                    lo, hi = VALID+block*chunk, VALID+(block+1)*chunk
                    candidate_replay(tr, lo, hi, variant, net, scaler, mode, steps)
                    state = discrete(tr)
                    memory = tr.model['memory'].memory.detach().cpu().clone()
                    if variant == 'exact':
                        references.append(dict(memory=memory, discrete=state))
                    ref = references[block]
                    check = byte_equal(state, ref['discrete'])
                    assert check['exact'], (name, chunk, variant, block, check)
                    active = (tr.neighbor.e_id >= 0).any(dim=1).cpu()
                    drift = error_stats(memory[active], ref['memory'][active])
                    logits = labels = None
                    if hi+BS <= END:
                        logits, labels = predict(tr, hi, hi+BS)
                        raw_logits.append(logits.cpu()); raw_labels.append(labels.cpu())
                    rows.append(dict(block=block+1, history_end=hi, drift=drift, discrete=check,
                                     logits=None if logits is None else logits.cpu(),
                                     labels=None if labels is None else labels.cpu()))
                    if block+1 in [1, 4, 16, nblocks]:
                        snapshots.append(dict(block=block+1, memory=memory, reference=ref['memory'],
                                              active=active, discrete=state, reference_discrete=ref['discrete']))
                logits, labels = torch.cat(raw_logits), torch.cat(raw_labels)
                quality = metric(logits.numpy(), labels.numpy())
                row = dict(quality=quality, final_drift=rows[-1]['drift'], discrete_exact=True,
                           blocks=nblocks, replayed_events=nblocks*chunk, queried_events=len(labels)//2)
                summary[variant] = row
                torch.save(dict(rows=rows, snapshots=snapshots, logits=logits, labels=labels),
                           out/f'{name}_k{chunk}_{variant}_rollout.pt')
                save_json(out/f'{name}_k{chunk}_{variant}_rollout.json', row)
                emit('rollout_done', dataset=name, chunk=chunk, variant=variant, **row)
                del tr; gc.collect(); p.ref.guard()
            save_json(out/f'{name}_k{chunk}_evaluation.json', dict(isolated=isolated, rollout=summary,
                                                                  seconds=time.perf_counter()-begin))


@torch.no_grad()
def timing_stage(out, names, rounds):
    rng = random.Random(20260909)
    for name in names:
        weights = load(out/(name+'_teacher.pt'))['weights']
        for chunk in CHUNKS:
            evaluation = json.loads((out/f'{name}_k{chunk}_evaluation.json').read_text())
            assert all(x['discrete_exact'] for x in evaluation['rollout'].values())
            start = load(out/f'{name}_k{chunk}_test_start.pt')
            end = VALID+(END-VALID)//chunk*chunk
            variants = list(candidates(out, name, chunk))
            tr = make_tr(name, weights)
            # Warm every path, including feature encoder and all solver lengths.
            for variant, net, scaler, mode, steps in variants:
                tr.restore(start)
                for lo in range(VALID, min(end, VALID+2*chunk), chunk):
                    candidate_replay(tr, lo, lo+chunk, variant, net, scaler, mode, steps)
            records = []
            for rnd in range(rounds):
                order = list(range(len(variants))); rng.shuffle(order)
                for k in order:
                    variant, net, scaler, mode, steps = variants[k]
                    tr.restore(start); gc.collect(); torch.cuda.synchronize()
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin = time.perf_counter_ns(); a.record()
                    for lo in range(VALID, end, chunk):
                        candidate_replay(tr, lo, lo+chunk, variant, net, scaler, mode, steps)
                    b.record(); torch.cuda.synchronize()
                    elapsed = (time.perf_counter_ns()-begin)/1e6
                    row = dict(round=rnd, variant=variant, wall_ms=elapsed, cuda_ms=a.elapsed_time(b),
                               events=end-VALID, order=[variants[j][0] for j in order])
                    records.append(row)
                    # Endpoint after timing must reproduce that path's quality run.
                    raw = load(out/f'{name}_k{chunk}_{variant}_rollout.pt', device='cpu')
                    assert torch.equal(tr.model['memory'].memory.cpu(), raw['snapshots'][-1]['memory'])
                emit('timing_round', dataset=name, chunk=chunk, round=rnd)
                p.ref.guard()
            save_json(out/f'{name}_k{chunk}_timing.json', records)
            del tr, variants; gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=['teacher','samples','fit','evaluate','timing'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--dataset', default='both', choices=['both','wikipedia','college'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--rounds', type=int, default=7)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20260909)
    names = ['wikipedia','college'] if args.dataset == 'both' else [args.dataset]
    emit('start', args={k:str(v) for k,v in vars(args).items()}, torch=torch.__version__,
         cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), sha256=p.sha(__file__))
    fn = dict(teacher=teacher_stage, samples=samples_stage, fit=fit_stage,
              evaluate=evaluate_stage, timing=timing_stage)[args.stage]
    extra = dict(teacher=[args.epochs], fit=[args.steps], timing=[args.rounds]).get(args.stage, [])
    fn(args.out, names, *extra)
    emit('complete', stage=args.stage)


if __name__ == '__main__':
    main()
