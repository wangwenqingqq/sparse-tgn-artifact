import csv
import json
import math
from pathlib import Path
import statistics as st
import numpy as np

root = Path(__file__).resolve().parents[1]
rows = [json.loads(s) for s in (root/'remote_evidence/output/timing_recheck_v5/stdout.jsonl').read_text().splitlines()]
assert rows[-1]['kind'] == 'complete'
assert all(r['pass_all'] for r in rows if r['kind'] == 'checkpoint_replay')
rng = np.random.default_rng(20260908)
data = []
for case in sorted({r['case'] for r in rows if r['kind'] == 'timing'}):
    variants = {v: {r['round']: r['wall_ms']/r['steps'] for r in rows
                   if r['kind'] == 'timing' and r['case'] == case and r['variant'] == v}
                for v in ['keeper', 'free_graph', 'free_relations']}
    assert all(set(d) == set(range(9)) for d in variants.values())
    record = dict(case=case, keeper_ms=st.median(variants['keeper'].values()))
    for v in ['free_graph', 'free_relations']:
        ratios = np.array([variants['keeper'][i]/variants[v][i] for i in range(9)])
        medians = np.median(ratios[rng.integers(0, 9, size=(20000, 9))], axis=1)
        lo, hi = np.quantile(medians, [.025, .975])
        ratio = float(np.median(ratios))
        record[v] = dict(median_ratio=ratio, paired_ratios=ratios.tolist(),
                         bootstrap95_median=[float(lo), float(hi)],
                         removable_fraction=1-1/ratio,
                         implied_2x_stage_full_ratio=1/(1-(1-1/ratio)/2))
    data.append(record)
result = dict(cases=data, intervals=324, timed_steps=2592, rounds_per_case_variant=9,
              checkpoint_replay_steps=96, checkpoint_checks_pass=True,
              geomean_free_relations=math.exp(st.mean(math.log(r['free_relations']['median_ratio']) for r in data)),
              note='Bootstrap of nine paired rounds within this shared-host session; not a cross-session or hardware guarantee. Original full_v5 source failures remain unchanged.')
(root/'analysis/recheck_summary.json').write_text(json.dumps(result, indent=2)+'\n')
lines = ['| 窗口 | keeper ms/步 | 免费建图 | 免费建图与全部关系 | 95% bootstrap 区间 |',
         '|---|---:|---:|---:|---:|']
for r in data:
    f = r['free_relations']
    lines.append(f"| {r['case']} | {r['keeper_ms']:.3f} | {r['free_graph']['median_ratio']:.3f}× | {f['median_ratio']:.3f}× | {f['bootstrap95_median'][0]:.3f}–{f['bootstrap95_median'][1]:.3f} |")
(root/'analysis/recheck_table.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(geomean=result['geomean_free_relations'],
                     ratios=[r['free_relations']['median_ratio'] for r in data],
                     largest_bootstrap_upper=max(r['free_relations']['bootstrap95_median'][1] for r in data))))
