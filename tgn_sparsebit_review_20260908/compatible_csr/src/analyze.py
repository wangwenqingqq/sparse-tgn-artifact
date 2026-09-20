import json
import math
from pathlib import Path
import statistics as st
import numpy as np

root=Path(__file__).resolve().parents[1]
rows=[json.loads(s) for s in (root/'evidence/timing_v1/stdout.jsonl').read_text().splitlines()]
assert any(r['kind']=='complete' for r in rows)
out=root/'analysis';out.mkdir(exist_ok=True)
cases=sorted({r['case'] for r in rows if r['kind']=='timing'})
assert len(cases)==12
rng=np.random.default_rng(20260908);result=[]
pairs=[('compatible','direct'),('compatible','fused_bounds'),('keeper','fused_bounds'),('keeper','direct'),('direct','fused_bounds')]
for case in cases:
    times={v:{r['round']:r['timeline_ms']/8 for r in rows if r['kind']=='timing' and r['case']==case and r['variant']==v}
           for v in ['compatible','keeper','direct','fused_bounds']}
    assert all(set(t)==set(range(9)) for t in times.values())
    ratios={}
    for a,b in pairs:
        v=np.array([times[a][i]/times[b][i] for i in range(9)])
        boot=np.median(v[rng.integers(0,9,size=(20000,9))],axis=1)
        ratios[a+'/'+b]=dict(median=float(np.median(v)),values=v.tolist(),
            min=float(v.min()),max=float(v.max()),bootstrap_95=np.quantile(boot,[.025,.975]).tolist())
    phases={r['variant']:{k:dict(timeline_ms_per_step=v['timeline_ms']/8,calls_per_step=v['calls']/8)
            for k,v in r['phases'].items()} for r in rows if r['kind']=='phases' and r['case']==case}
    result.append(dict(case=case,median_ms={v:st.median(t.values()) for v,t in times.items()},ratios=ratios,phases=phases))
summary=dict(cases=result,intervals=432,timed_steps=3456,paired_rounds=9,gpu_index=3,
    aggregate={a+'/'+b:dict(geomean=math.exp(st.mean(math.log(r['ratios'][a+'/'+b]['median']) for r in result)),
        min=min(r['ratios'][a+'/'+b]['median'] for r in result),max=max(r['ratios'][a+'/'+b]['median'] for r in result)) for a,b in pairs},
    caveats=['Paired-ratio medians and geomean across 12 windows; no cross-session pooling.',
        'Bootstrap is descriptive within this shared-host GPU3 session.',
        'Keeper remains source-unqualified in two windows.',
        'Instrumented phases are separate replays and include CPU launch gaps.',
        'Direct CSR defers row-index construction to torch_sparse SpMM, so conversion and aggregation phases must be read together.'])
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['| 窗口 | 稠密兼容 ms/步 | 直接 CSR ms/步 | 合并边界读取 ms/步 | 旧 keeper ms/步 | 稠密/直接 | keeper/直接 |',
       '|---|---:|---:|---:|---:|---:|---:|']
for r in result:
    t=r['median_ms'];v=r['ratios']
    lines.append(f"| {r['case']} | {t['compatible']:.3f} | {t['direct']:.3f} | {t['fused_bounds']:.3f} | {t['keeper']:.3f} | {v['compatible/direct']['median']:.3f} | {v['keeper/direct']['median']:.3f} |")
(out/'table.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary['aggregate'],indent=2))
