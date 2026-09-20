import json
import math
from pathlib import Path
import statistics as st
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'analysis';OUT.mkdir(exist_ok=True)
def logs(run):return [json.loads(s) for s in (ROOT/'evidence'/run/'stdout.jsonl').read_text().splitlines()]
rows=logs('timing_v1');profiles=logs('profiles_v1')
assert all(any(r['kind']=='complete' for r in group) for group in [rows,profiles])
cases=sorted({r['case'] for r in rows if r['kind']=='timing'});assert len(cases)==12
variants=['direct','direct_nograd','batch_store','batch_nograd']
pairs=[('direct','direct_nograd'),('batch_store','batch_nograd'),('direct','batch_store'),('direct','batch_nograd'),('direct_nograd','batch_nograd')]
rng=np.random.default_rng(202609091);result=[]
for case in cases:
    times={v:{r['round']:r['timeline_ms']/8 for r in rows if r['kind']=='timing' and r['case']==case and r['variant']==v} for v in variants}
    assert all(set(t)==set(range(9)) for t in times.values());ratios={}
    for a,b in pairs:
        x=np.array([times[a][j]/times[b][j] for j in range(9)])
        boot=np.median(x[rng.integers(0,9,size=(20000,9))],axis=1)
        ratios[a+'/'+b]=dict(median=float(np.median(x)),values=x.tolist(),bootstrap_95=np.quantile(boot,[.025,.975]).tolist())
    phases={r['variant']:{k:dict(timeline_ms_per_step=v['timeline_ms']/8,host_ms_per_step=v['host_ms']/8,calls_per_step=v['calls']/8) for k,v in r['phases'].items()} for r in profiles if r['kind']=='phases' and r['case']==case}
    result.append(dict(case=case,median_ms={v:st.median(t.values()) for v,t in times.items()},ratios=ratios,phases=phases))
summary=dict(cases=result,variants=variants,intervals=432,timed_steps=3456,paired_rounds=9,
    aggregate={a+'/'+b:dict(geomean=math.exp(st.mean(math.log(r['ratios'][a+'/'+b]['median']) for r in result)),
        min=min(r['ratios'][a+'/'+b]['median'] for r in result),max=max(r['ratios'][a+'/'+b]['median'] for r in result),
        lower_ci_above_one=sum(r['ratios'][a+'/'+b]['bootstrap_95'][0]>1 for r in result),
        upper_ci_below_one=sum(r['ratios'][a+'/'+b]['bootstrap_95'][1]<1 for r in result)) for a,b in pairs},
    mechanism=[r for r in profiles if r['kind']=='mechanism'],
    caveats=['Nine paired-ratio medians per fixed window; descriptive 20k bootstrap within one GPU3 session.',
        'No cross-session multiplication of speedups. Combined speed does not isolate the no-grad effect.',
        'Detailed phases are separate instrumented replays, including CPU submission gaps.',
        'Batch-store view retention is inherited and not solved here. Only eight-step windows are qualified.'])
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['| 窗口 | direct ms/步 | direct no-grad | batch store | batch no-grad | no-grad 单独收益 | no-grad 叠加收益 | 组合总收益 |',
       '|---|---:|---:|---:|---:|---:|---:|---:|']
for r in result:
    t=r['median_ms'];v=r['ratios']
    lines.append(f"| {r['case']} | {t['direct']:.3f} | {t['direct_nograd']:.3f} | {t['batch_store']:.3f} | {t['batch_nograd']:.3f} | {v['direct/direct_nograd']['median']:.3f} | {v['batch_store/batch_nograd']['median']:.3f} | {v['direct/batch_nograd']['median']:.3f} |")
(OUT/'table.md').write_text('\n'.join(lines)+'\n')
lines=['| 窗口 | direct / direct no-grad，95% CI | batch / batch no-grad，95% CI |','|---|---:|---:|']
for r in result:
    cols=[]
    for pair in ['direct/direct_nograd','batch_store/batch_nograd']:
        lo,hi=r['ratios'][pair]['bootstrap_95'];cols.append(f'[{lo:.4f}, {hi:.4f}]')
    lines.append('| '+r['case']+' | '+' | '.join(cols)+' |')
(OUT/'intervals.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary['aggregate'],indent=2))
