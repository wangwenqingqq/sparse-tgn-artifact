"""Independent NumPy comparisons of saved real-training tensors (CPU only)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('--input', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
assert not args.output.exists()
log = [json.loads(s) for s in (args.input.parent/'stdout.jsonl').read_text().splitlines()]
expected = {r['case']: r for r in log if r['kind'] == 'training_checks'}
results = []
files = {}
elements = 0

def read(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    files[path.name] = dict(bytes=path.stat().st_size, sha256=h.hexdigest())
    return torch.load(path, map_location='cpu', weights_only=True)

for path in sorted(args.input.glob('*_keeper_checks.pt')):
    name = path.name.removesuffix('_keeper_checks.pt')
    anchor = read(path)
    assert len(anchor) == 8
    for variant in ['keeper', 'source', 'free_graph', 'free_relations']:
        values = read(args.input/(name+'_'+variant+'_replay_checks.pt'))
        assert len(values) == len(anchor)
        for j, (a, b) in enumerate(zip(values, anchor)):
            assert a.keys() == b.keys()
            failures = []
            max_abs = 0.
            for key in a:
                x, y = a[key], b[key]
                if x is None or y is None:
                    assert x is None and y is None, (name, variant, j, key)
                    continue
                if torch.is_tensor(x):
                    x, y = x.numpy(), y.numpy()
                    assert x.dtype == y.dtype and x.shape == y.shape
                    elements += x.size
                    # FP64 lifting makes tolerance arithmetic independent of Torch.
                    if np.issubdtype(x.dtype, np.floating):
                        xx, yy = x.astype(np.float64), y.astype(np.float64)
                        diff = np.abs(xx-yy)
                        ok = np.isfinite(xx) & np.isfinite(yy) & (diff <= 2e-4+2e-4*np.abs(yy))
                        err = float(diff.max()) if diff.size else 0.
                        max_abs = max(max_abs, err)
                    else:
                        ok = x == y
                    if not np.all(ok):
                        at = np.unravel_index(np.argmax(~ok), x.shape) if x.shape else ()
                        failures.append(dict(field=key, count=int(np.count_nonzero(~ok)),
                            first_coordinate=[int(v) for v in at], actual=float(x[at]), reference=float(y[at])))
                else:
                    assert x == y
            index = int(name.rsplit('_s', 1)[1])+j
            predicted = next(c for c in expected[name]['checks'] if c['variant'] == variant and c['index'] == index)
            assert bool(not failures) == predicted['pass_all'], ('runtime/audit disagreement', name, variant, index)
            results.append(dict(case=name, variant=variant, index=index, pass_all=not failures,
                                fields=len(a), max_abs=max_abs, failures=failures))
    del anchor, values
    print(json.dumps(dict(case=name, audit_complete=True)), flush=True)
assert len(expected) == 12 and len(results) == 384
result = dict(status='complete', comparisons=len(results), tensor_element_comparisons=elements,
              cases=12, all_runtime_flags_reproduced=True, cuda_initialized=torch.cuda.is_initialized(),
              passed=sum(r['pass_all'] for r in results), failed=sum(not r['pass_all'] for r in results),
              by_variant={v: dict(passed=sum(r['pass_all'] for r in results if r['variant'] == v),
                                  failed=sum(not r['pass_all'] for r in results if r['variant'] == v))
                          for v in ['keeper', 'source', 'free_graph', 'free_relations']},
              files=files, results=results)
args.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('files', 'results')}), flush=True)
