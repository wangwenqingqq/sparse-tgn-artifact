"""Retain every benchmark cell and summarize medians, with paired-round ratios."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

p=argparse.ArgumentParser()
p.add_argument('input',type=Path)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
with args.input.open() as f: data=list(csv.DictReader(f))
groups={}
for r in data:
    key=(r['class'],int(r['k']),int(r['dim']),int(r['tiles']),int(r['packing']))
    mode=r['mode']; round_id=int(r['round']); value=float(r['us_per_call'])
    assert math.isfinite(value) and value>0
    runs=groups.setdefault(key,{}).setdefault(mode,{})
    assert round_id not in runs
    runs[round_id]=value
expected={'natural_sparse','aligned_sparse','natural_dense','simt_sparse','aligned_split'}
cells=[]
for key,modes in sorted(groups.items()):
    assert set(modes)==expected
    assert all(set(v)==set(range(7)) for v in modes.values())
    medians={name:statistics.median(v.values()) for name,v in modes.items()}
    candidate=medians['aligned_sparse']
    competitors=['natural_sparse','natural_dense','simt_sparse']
    best=min(competitors,key=lambda n:medians[n])
    ratio={name:medians[name]/candidate for name in expected if name!='aligned_sparse'}
    paired={name:[modes[name][i]/modes['aligned_sparse'][i] for i in range(7)] for name in competitors}
    cells.append(dict(topology=key[0],k=key[1],dim=key[2],tiles=key[3],packing=key[4],
                      median_us=medians,baseline_over_aligned=ratio,
                      best_independent=best,best_over_aligned=medians[best]/candidate,
                      paired_round_ratios=paired))
assert len(cells)==72 and len(data)==2520
geo=lambda xs:math.exp(statistics.mean(math.log(x) for x in xs))
summaries=[]
for pack in [0,1]:
    for topology in ['all','two_of_four','dense']:
        subset=[c for c in cells if c['packing']==pack and (topology=='all' or c['topology']==topology)]
        row=dict(packing=pack,topology=topology,cells=len(subset))
        for name in ['natural_sparse','natural_dense','simt_sparse','aligned_split']:
            vals=[c['baseline_over_aligned'][name] for c in subset]
            row[name]=dict(geomean_ratio=geo(vals),minimum=min(vals),maximum=max(vals),
                           aligned_faster_cells=sum(v>1 for v in vals),
                           aligned_faster_by_5pct_cells=sum(v>1.05 for v in vals))
        vals=[c['best_over_aligned'] for c in subset]
        row['best_independent']=dict(geomean_ratio=geo(vals),minimum=min(vals),maximum=max(vals),
                                     aligned_faster_cells=sum(v>1 for v in vals),
                                     aligned_faster_by_5pct_cells=sum(v>1.05 for v in vals))
        summaries.append(row)
result=dict(scope='One Blackwell GPU run; B1 lowering must be labeled; synthetic operator chain',
            ratio_definition='baseline time / aligned_sparse time; values above 1 favor aligned_sparse',
            rounds=7,event_intervals=len(data),calls_per_interval=20,
            cells=cells,summaries=summaries,
            uncertainty='Descriptive medians and paired rounds only. No independent-session CI or model-quality inference.')
args.output.write_text(json.dumps(result,indent=2)+'\n')
table=args.output.with_suffix('.csv')
with table.open('w',newline='') as f:
    writer=csv.writer(f)
    writer.writerow(['topology','k','dim','tiles','packing',*sorted(expected),'best_independent','best_over_aligned'])
    for c in cells:writer.writerow([c[n] for n in ['topology','k','dim','tiles','packing']]+[c['median_us'][n] for n in sorted(expected)]+[c['best_independent'],c['best_over_aligned']])
print(json.dumps(summaries,indent=2))
