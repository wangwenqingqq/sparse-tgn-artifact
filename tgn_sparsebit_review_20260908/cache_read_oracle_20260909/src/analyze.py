import json
import math
from pathlib import Path
import statistics as st
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'analysis';OUT.mkdir(exist_ok=True)
def logs(run):return [json.loads(s) for s in (ROOT/'evidence'/run/'stdout.jsonl').read_text().splitlines()]
rows=logs('timing_v1');profiles=logs('profiles_v1');qualification=logs('qualify_v2')
assert all(any(r['kind']=='complete' for r in group) for group in [rows,profiles,qualification])
cases=sorted({r['case'] for r in rows if r['kind']=='timing'});assert len(cases)==12
variants=['batch_store','free_cache_reads'];rng=np.random.default_rng(202609092);result=[];bootstrap_windows=[]
for case in cases:
    times={v:{r['round']:r['timeline_ms']/8 for r in rows if r['kind']=='timing' and r['case']==case and r['variant']==v} for v in variants}
    assert all(set(t)==set(range(9)) for t in times.values())
    ratio=np.array([times['batch_store'][j]/times['free_cache_reads'][j] for j in range(9)])
    boot=np.median(ratio[rng.integers(0,9,size=(20000,9))],axis=1)
    bootstrap_windows.append(boot)
    phases={r['variant']:{k:dict(timeline_ms_per_step=v['timeline_ms']/8,host_ms_per_step=v['host_ms']/8,calls_per_step=v['calls']/8) for k,v in r['phases'].items()} for r in profiles if r['kind']=='phases' and r['case']==case}
    prefix={}
    for v in variants:
        prefix[v]=dict(timeline_ms_per_step=sum(x['timeline_ms_per_step'] for k,x in phases[v].items() if k.split('/')[-1] in ['message.lookup','message.unpack','message.concat','message.cached_read']),
            calls_per_step={leaf:sum(x['calls_per_step'] for k,x in phases[v].items() if k.endswith('/'+leaf)) for leaf in ['message.lookup','message.unpack','message.concat','message.cached_read']})
    tape=next(r for r in qualification if r['kind']=='tape' and r['case']==case)
    _,b,start=case.split('_');display=case.split('_')[0]+' / batch_size='+b[1:]+' / step'+start[1:]
    result.append(dict(case=case,display=display,median_ms={v:st.median(t.values()) for v,t in times.items()},
        speedup=dict(median=float(np.median(ratio)),values=ratio.tolist(),bootstrap_95=np.quantile(boot,[.025,.975]).tolist()),
        phases=phases,prefix=prefix,tape=tape))
geo=math.exp(st.mean(math.log(r['speedup']['median']) for r in result))
aggregate_bootstrap=np.exp(np.mean(np.log(np.stack(bootstrap_windows)),axis=0))
summary=dict(cases=result,variants=variants,intervals=216,timed_steps=1728,paired_rounds=9,
    speedup=dict(geomean=geo,descriptive_bootstrap_95=np.quantile(aggregate_bootstrap,[.025,.975]).tolist(),min=min(r['speedup']['median'] for r in result),max=max(r['speedup']['median'] for r in result),
        lower_ci_above_one=sum(r['speedup']['bootstrap_95'][0]>1 for r in result),
        lower_ci_above_1_10=sum(r['speedup']['bootstrap_95'][0]>1.10 for r in result),
        upper_ci_below_1_10=sum(r['speedup']['bootstrap_95'][1]<1.10 for r in result)),
    sensitivity_leave_one_window_out={r['case']:math.exp(st.mean(math.log(x['speedup']['median']) for x in result if x['case']!=r['case'])) for r in result},
    by_batch={str(bs):dict(geomean=math.exp(st.mean(math.log(r['speedup']['median']) for r in result if '_b'+str(bs)+'_' in r['case'])),windows=6) for bs in [32,200]},
    decision=dict(threshold=1.10,metric='Geomean across 12 paired-ratio medians',
        passes_point_estimate=geo>=1.10,action='Admit further investigation, accounting for uncertainty and implementation cost' if geo>=1.10 else 'Do not build a complex cache-read optimization to chase overall 1.10x on this workload'),
    tape=dict(records=sum(r['tape']['records'] for r in result),
        logical_bytes=sum(r['tape']['logical_bytes'] for r in result),
        max_window_resident_bytes=max(r['tape']['unique_storage_bytes'] for r in result)),
    caveats=['Diagnostic oracle: capture/load is outside timing; same records are resident in both arms.',
        'Source time encoding, current-memory gathers, aggregation, GRU, cache writes, backward and Adam are paid.',
        'Oracle cursor cost remains. Reused buffers, allocator and cache effects prevent a mathematical hard-bound claim.',
        'Aggregate bootstrap and leave-one-window-out sensitivity are post-run descriptive checks; the predeclared point-estimate gate is unchanged.',
        'Descriptive bootstrap within one GPU3 session; no full-epoch, long-run retention or Tensor Core claim.'])
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['| 窗口 | 批量缓存基线 ms/步 | 预缓存读取对照 ms/步 | 配对比值 | 95% CI |','|---|---:|---:|---:|---:|']
for r in result:
    t=r['median_ms'];v=r['speedup'];lo,hi=v['bootstrap_95']
    lines.append(f"| {r['display']} | {t['batch_store']:.3f} | {t['free_cache_reads']:.3f} | {v['median']:.3f} | [{lo:.3f}, {hi:.3f}] |")
(OUT/'table.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:summary[k] for k in ['speedup','decision','tape']},indent=2))
