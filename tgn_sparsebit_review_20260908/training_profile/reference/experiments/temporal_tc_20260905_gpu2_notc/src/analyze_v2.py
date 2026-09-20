"""Frozen paired estimators; all shapes, regressions and orders retained."""
import csv,hashlib,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/analysis2';OUT.mkdir(parents=True,exist_ok=False)
V=['L-separate','B-separate','L-shared','B-shared','L-shared-library']
raw=[]
for p in range(5):
    rows=json.loads((ROOT/f'output/gpu2/bench{p}/samples.json').read_text());assert len(rows)==900
    assert all(r['process']==p and r['order']==V[p:]+V[:p] and r['eligible'] for r in rows);raw+=rows
keep=[r for r in raw if not r['warmup']];assert len(keep)==3150
with (OUT/'samples.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
def stats(a):
    return dict(zip(['p10','median','p90'],map(float,np.percentile(a,[10,50,90]))))
def paired(logs):
    logs=np.array(logs);rng=np.random.RandomState(20260905)
    boot=np.exp(logs[rng.randint(0,len(logs),(2000,len(logs)))].mean(1))
    lo,hi=map(float,np.percentile(boot,[2.5,97.5]));ratio=float(np.exp(logs.mean()))
    return dict(paired_geomean_ratio=ratio,bootstrap95=[lo,hi],process_wins=int((logs>0).sum()),ordinary_prototype_gate=bool(lo>1.02 and (logs>0).sum()>=4))
comparisons=[('L-sharing','L-separate','L-shared'),('B-sharing','B-separate','B-shared'),('native-vs-library','L-shared-library','L-shared'),('B-vs-L-shared','L-shared','B-shared')]
shapes=[];latency=[]
for case in sorted(set(r['case'] for r in keep)):
    rows=[r for r in keep if r['case']==case];med={(p,v):np.median([r['wall_us'] for r in rows if r['process']==p and r['variant']==v]) for p in range(5) for v in V}
    ev={(p,v):np.median([r['event_us'] for r in rows if r['process']==p and r['variant']==v]) for p in range(5) for v in V}
    for v in V:
        latency.append(dict(case=case,variant=v,wall_us=stats([r['wall_us'] for r in rows if r['variant']==v]),event_us=stats([r['event_us'] for r in rows if r['variant']==v]),process_wall_medians_us=[float(med[p,v]) for p in range(5)]))
    for name,a,b in comparisons:
        logs=[math.log(med[p,a]/med[p,b]) for p in range(5)];pairs=[]
        for p in range(5):
            order=V[p:]+V[:p]
            pairs.append(dict(process=p,order=order,numerator=a,denominator=b,ratio=float(med[p,a]/med[p,b]),event_ratio=float(ev[p,a]/ev[p,b]),numerator_first=order.index(a)<order.index(b)))
        splits={}
        for first in [True,False]:
            x=[r['ratio'] for r in pairs if r['numerator_first']==first];splits[str(first)]=dict(processes=len(x),geomean=float(np.exp(np.log(x).mean())))
        item=dict(case=case,comparison=name,**paired(logs),process_pairs=pairs,order_split=splits,
                  marginal_median_ratio=float(np.median([r['wall_us'] for r in rows if r['variant']==a])/np.median([r['wall_us'] for r in rows if r['variant']==b])),
                  arithmetic_mean_process_ratio=float(np.mean(np.exp(logs))))
        if name not in ['L-sharing','B-sharing']:item['ordinary_prototype_gate']=None # Secondary comparisons have no predeclared promotion gate.
        shapes.append(item)
aggregate=[]
for name,a,b in comparisons:
    rs=[r for r in shapes if r['comparison']==name]
    # Aggregate across shapes inside each process, then paired process estimator/CI.
    logs=[float(np.mean([math.log(r['process_pairs'][p]['ratio']) for r in rs])) for p in range(5)]
    item=dict(comparison=name,**paired(logs),per_process_case_geomean_ratio=list(map(float,np.exp(logs))),
              shape_ratio_range=[min(r['paired_geomean_ratio'] for r in rs),max(r['paired_geomean_ratio'] for r in rs)],
              per_shape_gate_pass=sum(r['ordinary_prototype_gate'] is True for r in rs),
              regressing_shape_means=[r['case'] for r in rs if r['paired_geomean_ratio']<1],shapes=18)
    if name not in ['L-sharing','B-sharing']:item['ordinary_prototype_gate']=None
    aggregate.append(item)
result=dict(scope='native ordinary controls only; synchronized initialized complete decoder-pair step; not full model/task/TC',retained=3150,warmups=1350,
            primary_comparisons=['L-sharing','B-sharing'],aggregate=aggregate,shapes=shapes,latencies=latency,
            caveats=['5 process bootstrap is a small-sample prototype screen','No learned-state trajectory or quality','No NCU or sustained workload promotion','Cyclic Latin order positions balanced; individual pair directions may be asymmetric'])
(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(retained=3150,warmups=1350,aggregate=aggregate),indent=2))
