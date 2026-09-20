import json
import math
from pathlib import Path
import statistics as st
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis';OUT.mkdir(exist_ok=True)
def read(name):return [json.loads(s) for s in (ROOT/'evidence'/name/'stdout.jsonl').read_text().splitlines()]
rows=read('timing_v1');profiles=read('profiles_v1');qual=read('qualify_v1')
assert all(any(r['kind']=='complete' for r in a) for a in [rows,profiles,qual])
cases=sorted({r['case'] for r in rows if r['kind']=='timing'});assert len(cases)==12
rng=np.random.default_rng(20260909);result=[]
for case in cases:
    times={v:{r['round']:r['timeline_ms']/8 for r in rows if r['kind']=='timing' and r['case']==case and r['variant']==v} for v in ['direct','batch_store']}
    assert all(set(t)==set(range(9)) for t in times.values())
    ratios=np.array([times['direct'][j]/times['batch_store'][j] for j in range(9)])
    boot=np.median(ratios[rng.integers(0,9,size=(20000,9))],axis=1)
    phases={r['variant']:{k:{'timeline_ms_per_step':v['timeline_ms']/8,'host_ms_per_step':v['host_ms']/8,'calls_per_step':v['calls']/8} for k,v in r['phases'].items()} for r in profiles if r['kind']=='phases' and r['case']==case}
    storage={r['variant']:[dict(index=c['index'],logical_bytes=sum(x['logical_bytes'] for x in c['storage'].values()),
        unique_storage_bytes=sum(x['unique_storage_bytes'] for x in c['storage'].values()),unique_storages=sum(x['unique_storages'] for x in c['storage'].values())) for c in r['checks']] for r in qual if r['kind']=='qualification' and r['case']==case}
    for s in storage.values():
        for x in s:x['retention_ratio']=x['unique_storage_bytes']/x['logical_bytes']
    result.append(dict(case=case,median_ms={v:st.median(t.values()) for v,t in times.items()},
        speedup=dict(median=float(np.median(ratios)),values=ratios.tolist(),bootstrap_95=np.quantile(boot,[.025,.975]).tolist()),
        phases=phases,storage=storage))
summary=dict(cases=result,intervals=216,timed_steps=1728,paired_rounds=9,
    speedup=dict(geomean=math.exp(st.mean(math.log(r['speedup']['median']) for r in result)),
        min=min(r['speedup']['median'] for r in result),max=max(r['speedup']['median'] for r in result)),
    storage=dict(max_retention_ratio=max(s['retention_ratio'] for r in result for s in r['storage']['batch_store']),
        max_storage_bytes=max(s['unique_storage_bytes'] for r in result for s in r['storage']['batch_store'])),
    caveats=['Paired medians, descriptive 20k bootstrap within one GPU3 session.',
        'Detailed profiles are separate replays with overhead. CUDA-event phase timelines include CPU submission gaps.',
        'Only 8 steps from each of 12 frozen checkpoints; message-store retention is not a full-epoch bound.'])
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['| 窗口 | direct ms/步 | batch store ms/步 | 加速比 | 配对比值 95% 区间 | 末步缓存存储/逻辑字节 |','|---|---:|---:|---:|---:|---:|']
for r in result:
    t=r['median_ms'];s=r['speedup'];lo,hi=s['bootstrap_95']
    lines.append(f"| {r['case']} | {t['direct']:.3f} | {t['batch_store']:.3f} | {s['median']:.3f} | [{lo:.3f}, {hi:.3f}] | {r['storage']['batch_store'][-1]['retention_ratio']:.3f} |")
(OUT/'table.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:summary[k] for k in ['speedup','storage']},indent=2))
