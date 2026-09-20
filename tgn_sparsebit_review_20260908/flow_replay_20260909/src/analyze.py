"""Aggregate paired replay timings and clustered quality uncertainty."""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def quality(scores, labels):
    order = np.argsort(-scores, kind='stable')
    s, y = scores[order], labels[order]
    cuts = np.r_[np.flatnonzero(np.diff(s)), len(s)-1]
    tp = np.cumsum(y)[cuts].astype(float); fp = cuts+1-tp
    recall = tp/tp[-1]
    ap = float(np.sum(np.diff(np.r_[0.,recall])*tp/(tp+fp)))
    tpr, fpr = np.r_[0.,recall], np.r_[0.,fp/fp[-1]]
    auc = float(np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])/2))
    return np.array([ap, auc])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260909)
    rows, cases, preparation = [], [], {}
    for name in ['wikipedia','college']:
        preparation[name] = dict(teacher=json.loads((args.input/(name+'_teacher.json')).read_text()))
        for chunk in [128,512]:
            case = f'{name}_k{chunk}'
            ev = json.loads((args.input/(case+'_evaluation.json')).read_text())
            timing = json.loads((args.input/(case+'_timing.json')).read_text())
            raw = np.load(args.input/(case+'_queries.npz'))
            exact = raw['exact']; labels = raw['labels']
            ids = raw['block']
            blocks = np.unique(ids)
            sample_ids = rng.integers(0,len(blocks),(1500,len(blocks)))
            exact_q = quality(exact, labels)
            times = {v:np.array([x['wall_ms'] for x in sorted(timing,key=lambda r:r['round']) if x['variant']==v])
                     for v in ev['rollout']}
            base = times['exact']
            timing_ids = rng.integers(0,len(base),(5000,len(base)))
            for variant, perf in ev['rollout'].items():
                ratio = base/times[variant]
                speed = float(np.median(ratio))
                ci = np.quantile(np.median(ratio[timing_ids],1),[.025,.975]).tolist()
                delta = quality(raw[variant],labels)-exact_q
                if variant == 'exact':
                    quality_ci = [[0.,0.],[0.,0.]]
                else:
                    replicates = []
                    for sampled in sample_ids:
                        selected = np.concatenate([np.flatnonzero(ids == blocks[b]) for b in sampled])
                        replicates.append(quality(raw[variant][selected],labels[selected])-quality(exact[selected],labels[selected]))
                    quality_ci = np.quantile(replicates,[.025,.975],axis=0).T.tolist()
                seed = int(variant.rsplit('_s',1)[1]) if '_s' in variant else None
                family = variant.split('_')[0]
                iso = ev['isolated'].get(variant,{})
                row = dict(dataset=name, chunk=chunk, variant=variant, family=family, seed=seed,
                           wall_ms=float(np.median(times[variant])), speed=speed, speed_ci=ci,
                           ap=perf['quality']['ap'], auc=perf['quality']['auc'], ap_delta=float(delta[0]),
                           auc_delta=float(delta[1]), ap_delta_ci=quality_ci[0], auc_delta_ci=quality_ci[1],
                           final_nrmse=perf['final_drift']['nrmse'], isolated_nrmse=iso.get('nrmse'),
                           isolated_delta_error=iso.get('relative_delta_rmse'),
                           passes_point_gate=bool(speed>=1.10 and min(delta)>=-.01),
                           passes_ci_gate=bool(ci[0]>=1.10 and min(q[0] for q in quality_ci)>=-.01),
                           queried_events=perf['queried_events'], replayed_events=perf['replayed_events'])
                if seed is not None:
                    mlp = f'mlp1_s{seed}'
                    row['speed_vs_mlp'] = float(np.median(times[mlp]/times[variant]))
                    row['ap_vs_mlp'] = row['ap']-ev['rollout'][mlp]['quality']['ap']
                    row['auc_vs_mlp'] = row['auc']-ev['rollout'][mlp]['quality']['auc']
                rows.append(row)
            case_rows = [r for r in rows if r['dataset']==name and r['chunk']==chunk]
            families = {}
            for family in sorted({r['family'] for r in case_rows}):
                selected = [r for r in case_rows if r['family']==family]
                families[family] = dict(count=len(selected), point_passes=sum(r['passes_point_gate'] for r in selected),
                    speed_median=float(np.median([r['speed'] for r in selected])),
                    speed_range=[min(r['speed'] for r in selected),max(r['speed'] for r in selected)],
                    ap_delta_range=[min(r['ap_delta'] for r in selected),max(r['ap_delta'] for r in selected)],
                    auc_delta_range=[min(r['auc_delta'] for r in selected),max(r['auc_delta'] for r in selected)],
                    final_nrmse_range=[min(r['final_nrmse'] for r in selected),max(r['final_nrmse'] for r in selected)])
            fit = [json.loads(p.read_text()) for p in args.input.glob(case+'_*_fit.json')]
            sample = json.loads((args.input/(case+'_samples.json')).read_text())
            # Per-model preparation cost includes that teacher, all target generation for this chunk,
            # and the selected objective/seed's full fixed-budget fit (including validation).
            for row in case_rows:
                if row['seed'] is None:
                    continue
                objective = 'mlp' if row['family'].startswith('mlp') else 'fm'
                train = next(x for x in fit if x['mode']==objective and x['seed']==row['seed'])
                prep = preparation[name]['teacher']['seconds']+sample['seconds']+train['seconds']
                saving = (float(np.median(base))-row['wall_ms'])/1000
                row['preparation_seconds'] = prep
                row['break_even_suffix_replays'] = math.ceil(prep/saving) if saving>0 else None
            cases.append(dict(case=case, teacher_boundary_quality=ev['rollout']['exact']['quality'],
                              families=families, samples=sample, fitting=fit))
    summary = dict(rows=rows,cases=cases,preparation=preparation,
                   uncertainty='Paired timing bootstrap over seven same-session rounds; paired query bootstrap over boundary blocks. Sequential blocks are correlated: these intervals are descriptive, not an independent-sequence generalization guarantee.')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines = ['| Dataset | Block | Variant | Replay speed | AP change (pp) | AUC change (pp) | Final memory NRMSE | Point gate |',
             '|---|---:|---|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['chunk']} | {r['variant']} | {r['speed']:.3f}x | {100*r['ap_delta']:+.3f} | {100*r['auc_delta']:+.3f} | {r['final_nrmse']:.3f} | {'pass' if r['passes_point_gate'] else 'fail'} |")
    (args.output/'table.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps([dict(case=c['case'],families=c['families']) for c in cases],indent=2))


if __name__ == '__main__':
    main()
