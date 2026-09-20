import json
from pathlib import Path
import torch
import audit as a

root = a.ROOT/'output/timing_v1'
rows = [json.loads(s) for s in (root/'stdout.jsonl').read_text().splitlines()]
results = []
for file in sorted((root/'artifacts').glob('*_oracle_checks.pt')):
    label = file.name.removesuffix('_oracle_checks.pt')
    actual = a.load(file)
    reference = a.load(a.OLD/(label+'_source_replay_checks.pt'))
    checks = [a.comp(x, y) for x, y in zip(actual, reference)]
    logged = next(r for r in rows if r['kind']=='oracle_check' and r['case']==label)
    assert [c['pass_all'] for c in checks] == [c['pass_all'] for c in logged['checks']]
    results.append(dict(case=label, checks=checks))
assert len(results)==12
assert not torch.cuda.is_initialized()
result = dict(cases=results, cuda_initialized=False,
    passed=sum(c['pass_all'] for r in results for c in r['checks']),
    bitwise=sum(c['bitwise_equal'] for r in results for c in r['checks']),
    total=sum(len(r['checks']) for r in results),
    elements=sum(c['elements'] for r in results for c in r['checks']))
(root/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
