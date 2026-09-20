"""CPU-only validation of real-source integration and checkpoint restoration."""
import os
os.environ['TRAIN_DEVICE'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import profile_training as p
import torch

torch.set_num_threads(1)
assert not torch.cuda.is_initialized()

def formula(n, edges, q):
    e = np.concatenate([edges, edges[::-1]], axis=1)
    a = sp.coo_matrix((np.ones(e.shape[1], np.int64), (e[0], e[1])), shape=(n, n)).tocsr()
    support = (a != 0).astype(np.int64)
    two = (a @ a != 0).astype(np.int64)
    u, v = q
    ar, br = support[u], support[v]
    aa, bb, h, k = [z.toarray() for z in (ar, br, two[u], two[v])]
    weights = np.asarray(a[u, v]).reshape(-1, 1)
    s = (ar.multiply(br) @ a).toarray()
    c = np.zeros((7, len(u), n), np.int64)
    row = np.arange(len(u))
    c[0, row, u] = bb[row, u]
    c[1, row, v] = aa[row, v]
    c[2] = aa * bb
    c[3] = aa * (k-weights)
    c[4] = bb * (h-weights)
    c[5] = h*k+s
    c[5][(weights > 0) & ((aa != 0) | (bb != 0))] = 0
    for j in (3, 4, 5):
        c[j, row, u] = c[j, row, v] = 0
    c[6] = s
    return c

for dataset in ['wikipedia', 'college']:
    for bs in [32, 200]:
        tr = p.Trainer(dataset, bs)
        for i in range(4):
            r = tr.step(i, 'source')
        ck = tr.checkpoint()
        first = [tr.step(i, 'source', check=True)['check'] for i in [4, 5]]
        tr.restore(ck)
        again = [tr.step(i, 'source', check=True)['check'] for i in [4, 5]]
        compares = [p.compare(x, y) for x, y in zip(first, again)]
        assert all(c['pass_all'] for c in compares)
        p.emit('cpu_training_restore', dataset=dataset, batch=bs, checks=compares,
               trained_history_steps=4, replay_steps=2, last_loss=float(first[-1]['loss']))

base = p.ROOT/'experiments/temporal_tc_20260905_tncn_mode2_gate0/output/census1'
cases = json.loads((base/'census.json').read_text())
for case in cases:
    state = next(s for s in case['states'] if s['state'] == 'before')
    with np.load(base/state['fixture']) as z:
        n, edges, q = len(z['ids']), z['edges'].copy(), z['targets'].copy()
    expected = formula(n, edges, q)
    actual = p.COEFFICIENTS(None, torch.zeros((n, 1)), torch.from_numpy(edges),
                            torch.from_numpy(q), 2).numpy()
    assert np.array_equal(actual, expected), case['case']
    p.emit('cpu_real_coefficient', case=case['case'], n=n, b=q.shape[1],
           cells=int(expected.size), S_nnz=int(np.count_nonzero(expected[6])), exact=True)
assert not torch.cuda.is_initialized()
p.emit('cpu_complete', cuda_initialized=torch.cuda.is_initialized(), coefficient_cases=len(cases))
