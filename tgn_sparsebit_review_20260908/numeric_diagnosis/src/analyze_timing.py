import json
import math
from pathlib import Path
import statistics as st
import numpy as np

root = Path(__file__).resolve().parents[1]
rows = [json.loads(s) for s in (root/'evidence/timing_v1/stdout.jsonl').read_text().splitlines()]
out = root/'analysis'; out.mkdir(exist_ok=True)
cases = sorted({r['case'] for r in rows if r['kind']=='timing'})
assert len(cases)==12 and any(r['kind']=='complete' for r in rows)
rng = np.random.default_rng(20260908)
results = []
for case in cases:
    times = {v: {r['round']:r['timeline_ms']/r['steps'] for r in rows if r['kind']=='timing' and r['case']==case and r['variant']==v}
             for v in ['source','keeper','compatible','compatible_free_relations']}
    assert all(set(t)==set(range(9)) for t in times.values())
    ratios = {}
    for a,b in [('source','compatible'),('keeper','compatible'),('compatible','compatible_free_relations')]:
        values = np.array([times[a][i]/times[b][i] for i in range(9)])
        boot = np.median(values[rng.integers(0,9,size=(20000,9))],axis=1)
        ratios[a+'/'+b] = dict(median=float(np.median(values)), values=values.tolist(),
            min=float(values.min()), max=float(values.max()), bootstrap_95=np.quantile(boot,[.025,.975]).tolist())
    results.append(dict(case=case, median_ms={v:st.median(t.values()) for v,t in times.items()}, ratios=ratios))
aggregate = {k:dict(geomean=math.exp(st.mean(math.log(r['ratios'][k]['median']) for r in results)),
    min=min(r['ratios'][k]['median'] for r in results),max=max(r['ratios'][k]['median'] for r in results))
    for k in results[0]['ratios']}
result=dict(protocol='nine randomized paired rounds; eight restored complete training steps per interval; gc.collect outside timing',
    intervals=12*9*4, timed_steps=12*9*4*8, cases=results, aggregate=aggregate,
    caveat='Within-session descriptive bootstrap only. Keeper is not source-qualified in two windows. Free relations also remove format conversion.')
(out/'timing_summary.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['| 窗口 | 原版 ms/步 | keeper ms/步 | 兼容 ms/步 | 免费关系兼容 ms/步 | 原版/兼容 | keeper/兼容 | 兼容/免费关系 |',
       '|---|---:|---:|---:|---:|---:|---:|---:|']
for r in results:
    t=r['median_ms']; v=r['ratios']
    lines.append(f"| {r['case']} | {t['source']:.3f} | {t['keeper']:.3f} | {t['compatible']:.3f} | {t['compatible_free_relations']:.3f} | {v['source/compatible']['median']:.3f} | {v['keeper/compatible']['median']:.3f} | {v['compatible/compatible_free_relations']['median']:.3f} |")
(out/'timing_table.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(aggregate,indent=2))
