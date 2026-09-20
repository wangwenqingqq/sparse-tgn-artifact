"""CPU/NumPy verification of stored predictions, state snapshots and split rules."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def binary_metrics(scores, labels):
    s, y = np.asarray(scores, np.float64).ravel(), np.asarray(labels, np.int64).ravel()
    # AUC from ascending average ranks (different computation from the GPU runner).
    order = np.argsort(s, kind='stable')
    _, first, counts = np.unique(s[order], return_index=True, return_counts=True)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.repeat(first+(counts+1)/2, counts)
    pos, neg = int(y.sum()), int((1-y).sum())
    auc = (ranks[y == 1].sum()-pos*(pos+1)/2)/(pos*neg)
    # AP directly counts items above each positive score threshold, including ties.
    thresholds, positives = np.unique(s[y == 1], return_counts=True)
    ap = sum(n/pos*float(y[s >= threshold].mean()) for threshold,n in zip(thresholds,positives))
    bce = np.mean(np.maximum(s, 0)-y*s+np.log1p(np.exp(-np.abs(s))))
    return dict(ap=float(ap), auc=float(auc), bce=float(bce), examples=len(s))


def same(a, b):
    assert a.keys() == b.keys()
    count = 0
    for key in a:
        if torch.is_tensor(a[key]):
            x, y = a[key].numpy(), b[key].numpy()
            assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), key
            count += x.size
        else:
            assert a[key] == b[key], key
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(1)
    rows, elements, snapshots, predictions = [], 0, 0, 0
    for path in sorted(args.input.glob('*_rollout.pt')):
        raw = torch.load(path, map_location='cpu', weights_only=False)
        stats = json.loads(path.with_suffix('.json').read_text())
        scores = raw['logits'].numpy(); labels = raw['labels'].numpy()
        assert np.isfinite(scores).all()
        quality = binary_metrics(scores, labels)
        for key in ['ap','auc','bce']:
            assert abs(quality[key]-stats['quality'][key]) < 1e-11, (path,key)
        for snap in raw['snapshots']:
            elements += same(snap['discrete'], snap['reference_discrete'])
            mask = snap['active'].numpy()
            actual, ref = snap['memory'].numpy()[mask].astype(float), snap['reference'].numpy()[mask].astype(float)
            assert np.isfinite(actual).all()
            sse = float(np.square(actual-ref).sum())
            expected = raw['rows'][snap['block']-1]['drift']
            assert np.isclose(sse, expected['sse'], rtol=1e-11, atol=1e-12), path
            assert np.isclose(np.sqrt(np.mean(np.square(actual-ref))), expected['rmse'], rtol=1e-11, atol=1e-12)
            elements += actual.size; snapshots += 1
        predictions += len(scores)
        rows.append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), quality=quality,
                         blocks=len(raw['rows']), snapshots=len(raw['snapshots'])))
    splits = []
    for path in sorted(args.input.glob('*_samples.json')):
        stem = path.name.removesuffix('_samples.json')
        previous = 0
        train = None
        for split,lo,hi in [('train',0,12000),('val',12000,16000),('test',16000,20000)]:
            data = torch.load(args.input/(stem+'_'+split+'.pt'), map_location='cpu', weights_only=False)
            windows = data['windows']
            assert windows[0]['lo'] == lo and windows[-1]['hi'] <= hi
            assert all(w['lo'] >= lo and w['hi'] <= hi for w in windows)
            assert all(a['hi'] == b['lo'] for a,b in zip(windows,windows[1:]))
            assert np.array_equal(data['condition'][:, :100].numpy(), data['initial'].numpy())
            assert len(data['condition']) == sum(w['nodes'] for w in windows)
            if split == 'train':
                train = data
            splits.append(dict(case=stem, split=split, blocks=len(windows), samples=len(data['delta']), lo=lo, hi=hi))
        scaler = torch.load(args.input/(stem+'_scaler.pt'), map_location='cpu', weights_only=False)
        cond = train['condition'].numpy().astype(float)
        assert np.allclose(cond.mean(0), scaler['mean'].numpy(), rtol=1e-4, atol=1e-5)
        assert np.allclose(np.maximum(cond.std(0,ddof=1),.01), scaler['scale'].numpy(), rtol=1e-4, atol=1e-5)
        delta = train['delta'].numpy().astype(float)
        assert np.allclose(np.maximum(np.sqrt(np.square(delta).mean(0)),.01), scaler['target_scale'].numpy(), rtol=1e-5, atol=1e-7)
    report = dict(pass_all=True, rollout_archives=len(rows), snapshots=snapshots, checked_elements=elements,
                  predictions=predictions, rollout=rows, splits=splits,
                  boundary='All stored query logits and selected raw state snapshots; intermediate discrete checks are recorded by the GPU runner.')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['rollout','splits']}))


if __name__ == '__main__':
    main()
