"""CPU/NumPy replay of archived numeric findings; initializes no CUDA context."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/home/data/wangxuran/tncn_numeric_diagnosis_20260908')
OLD = Path('/home/data/wangxuran/tncn_training_profile_20260908/output/full_v5/artifacts')


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def comp(a, b):
    assert a.keys() == b.keys()
    failed, different, total = [], [], 0
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, torch.Tensor):
            assert isinstance(y, torch.Tensor) and x.shape == y.shape
            x, y = x.numpy(), y.numpy()
            total += x.size
            if np.issubdtype(x.dtype, np.floating):
                d = np.abs(x.astype(np.float64)-y.astype(np.float64))
                ok = np.isfinite(x) & np.isfinite(y) & (d <= 2e-4+2e-4*np.abs(y.astype(np.float64)))
            else:
                ok = x == y
            if x.dtype != y.dtype or x.tobytes() != y.tobytes():
                different.append(k)
            if not np.all(ok):
                failed.append(k)
        elif x != y:
            different.append(k); failed.append(k)
    return dict(pass_all=not failed, bitwise_equal=not different,
                failures=failed, different_fields=different, elements=total)


def reference(raw):
    x, c, q = raw['x'].numpy().astype(np.float64), raw['C'].numpy().astype(np.float64), raw['q'].numpy()
    b, n, d = q.shape[1], len(x), x.shape[1]
    y = (c.reshape(-1, n) @ x).reshape(7, b, d).transpose(1, 0, 2).reshape(b, -1)
    weights = {k: v.numpy().astype(np.float64) for k, v in raw['predictor_before'].items()}
    w0, b0, w2, b2 = [weights['pred.xsmlp.'+s] for s in ['0.weight', '0.bias', '2.weight', '2.bias']]
    features = np.concatenate([x[q[0]]*x[q[1]], y], axis=1)
    h = features @ w0.T + b0
    relu = np.maximum(h, 0)
    logits = relu @ w2.T + b2
    half = b//2
    target = np.zeros_like(logits); target[:half] = 1
    loss = np.sum(np.logaddexp(0, logits)-target*logits)/half
    dl = (1/(1+np.exp(-logits))-target)/half
    dh = (dl @ w2)*(h > 0)
    df = dh @ w0
    dy = df[:, d:]
    dx = c.reshape(-1, n).T @ dy.reshape(b, 7, d).transpose(1, 0, 2).reshape(-1, d)
    np.add.at(dx, q[0], df[:, :d]*x[q[1]])
    np.add.at(dx, q[1], df[:, :d]*x[q[0]])
    common = raw['common_dy'].numpy().astype(np.float64).reshape(b, 7, d).transpose(1, 0, 2).reshape(-1, d)
    expected = dict(y=y, logits=logits, loss=np.asarray(loss), dx=dx, dy=dy,
        common_vjp=c.reshape(-1, n).T@common)
    expected.update({'grad/xsmlp.0.weight': dh.T@features, 'grad/xsmlp.0.bias': dh.sum(0),
                     'grad/xsmlp.2.weight': dl.T@relu, 'grad/xsmlp.2.bias': dl.sum(0)})
    results = {}
    for k, a in expected.items():
        b = raw['dense64'][k].numpy()
        delta = np.abs(a-b)
        passed = bool(np.all(delta <= 1e-10+1e-12*np.abs(a)))
        results[k] = dict(pass_all=passed, max_abs=float(delta.max()), elements=a.size)
        assert passed, (k, results[k])
    return results


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); args = ap.parse_args()
    assert not torch.cuda.is_initialized()
    rows = [json.loads(s) for s in (ROOT/'output/witnesses_v1/stdout.jsonl').read_text().splitlines()]
    output = dict(witness=[], qualification=[], independent_fp64=[], hashes={})
    for label in ['wikipedia_b200_s20', 'college_b200_s20']:
        keeper = load(OLD/(label+'_keeper_checks.pt'))
        source = load(ROOT/'output/witnesses_v1/artifacts'/(label+'_source_checks.pt'))
        for mode in ['source', 'source_anchor_frequency', 'source_anchor_mlp', 'compatible']:
            actual = load(ROOT/'output/witnesses_v1/artifacts'/(label+'_'+mode+'_checks.pt'))
            logged = next(r for r in rows if r['kind']=='witness_trajectory' and r['case']==label and r['variant']==mode)
            checks = [comp(a, b) for a, b in zip(actual, keeper)]
            assert [c['pass_all'] for c in checks] == [c['pass_all'] for c in logged['checks']]
            entry = dict(case=label, mode=mode, versus_keeper=checks)
            if mode == 'compatible':
                entry['versus_source'] = [comp(a, b) for a, b in zip(actual, source)]
            output['witness'].append(entry)
        fp = load(ROOT/'output/witnesses_v1/artifacts'/(label+'_time64.pt'))
        checks = [comp(a, b) for a, b in zip(fp['source'], fp['keeper'])]
        logged = next(r for r in rows if r['kind']=='time64_witness' and r['case']==label)
        assert [c['pass_all'] for c in checks] == [c['pass_all'] for c in logged['checks']]
        output['witness'].append(dict(case=label, mode='time64', versus_keeper=checks))
    rows = [json.loads(s) for s in (ROOT/'output/qualify_v1/stdout.jsonl').read_text().splitlines()]
    for file in sorted((ROOT/'output/qualify_v1/artifacts').glob('*_qualification.pt')):
        label = file.name.removesuffix('_qualification.pt')
        source = load(OLD/(label+'_source_replay_checks.pt'))
        actual = load(file)['compatible/compatible']
        checks = [comp(a, b) for a, b in zip(actual, source)]
        logged = next(r for r in rows if r['kind']=='qualification' and r['case']==label)
        assert [c['pass_all'] for c in checks] == [c['pass_all'] for c in logged['checks']]
        output['qualification'].append(dict(case=label, checks=checks))
    for file in sorted((ROOT/'output/witnesses_v1/artifacts').glob('*_isolated.pt')):
        output['independent_fp64'].append(dict(case=file.stem, checks=reference(load(file))))
    for file in sorted(ROOT.glob('output/*/artifacts/*.pt')):
        h = hashlib.sha256()
        with file.open('rb') as f:
            for chunk in iter(lambda: f.read(1<<20), b''): h.update(chunk)
        output['hashes'][str(file.relative_to(ROOT))] = dict(bytes=file.stat().st_size, sha256=h.hexdigest())
    assert not torch.cuda.is_initialized()
    output['cuda_initialized'] = False
    output['summary'] = dict(qualification_pass=sum(c['pass_all'] for r in output['qualification'] for c in r['checks']),
        qualification_bitwise=sum(c['bitwise_equal'] for r in output['qualification'] for c in r['checks']),
        qualification_total=sum(len(r['checks']) for r in output['qualification']),
        comparison_elements=sum(c['elements'] for r in output['qualification'] for c in r['checks']))
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output['summary']), flush=True)


if __name__ == '__main__': main()
