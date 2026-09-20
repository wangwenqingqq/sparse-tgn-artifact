from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def read(name,file):return json.loads((ROOT/'evidence'/name/file).read_text())

def main():
    meta=read('capture_v1','artifacts/capture.json');q1=read('qualify_v1','artifacts/qualification.json');q2=read('qualify_keep_empty_v2','artifacts/qualification.json');noise=read('noise_v2','artifacts/noise.json')
    for name in ['qualify_v1','qualify_keep_empty_v2','noise_v2']:assert read(name,'audit.json')['all_logged_comparisons_match']
    assert not q1['all_pass'] and q2['all_pass'];assert read('qualify_keep_empty_v2','audit.json')['all_pass']
    timing=read('timing_keep_empty_v1','artifacts/timing.json')['intervals'];assert len(timing)==216
    empty={x['label']:x for x in json.loads((ROOT/'analysis/empty_queries.json').read_text())['cases']}
    rng=np.random.default_rng(20260911);rows=[]
    for port in meta['ports']:
        obs=[x for x in timing if x['label']==port['label']];assert len(obs)==18
        paired={i:{x['arm']:x for x in obs if x['rep']==i} for i in range(9)}
        ratios=np.array([v['source']['cuda_ms']/v['candidate']['cuda_ms'] for v in paired.values()]);boot=np.median(ratios[rng.integers(0,9,size=(20000,9))],axis=1)
        wallratios=np.array([v['source']['wall_ms']/v['candidate']['wall_ms'] for v in paired.values()])
        row=dict(port=port,empty_queries=empty[port['label']]['empty_queries'],retained_dense_rows=empty[port['label']]['retained_projection_rows'],
            source_ms=float(np.median([v['source']['cuda_ms']/5 for v in paired.values()])),candidate_ms=float(np.median([v['candidate']['cuda_ms']/5 for v in paired.values()])),
            ratio=float(np.median(ratios)),ratio_ci95=np.quantile(boot,[.025,.975]).tolist(),paired_ratios=ratios.tolist(),wall_ratio=float(np.median(wallratios)))
        rows.append(row)
    groups={}
    for name,selection in [('all12',rows),('source_dense',[x for x in rows if not x['port']['source_packed']]),('source_packed_identity',[x for x in rows if x['port']['source_packed']])]:
        matrix=np.array([x['paired_ratios'] for x in selection]);draw=rng.integers(0,9,size=(20000,len(selection),9));values=np.take_along_axis(np.broadcast_to(matrix,(20000,*matrix.shape)),draw,axis=2)
        boot=np.exp(np.mean(np.log(np.median(values,axis=2)),axis=1));ratios=[x['ratio'] for x in selection]
        groups[name]=dict(n=len(selection),geomean=float(np.exp(np.mean(np.log(ratios)))),descriptive_ci95=np.quantile(boot,[.025,.975]).tolist(),min=min(ratios),max=max(ratios),cases_ci_lower_gt_one=sum(x['ratio_ci95'][0]>1 for x in selection),cases_ci_upper_lt_one=sum(x['ratio_ci95'][1]<1 for x in selection))
    nchecks=[v for x in noise['cases'] for v in x['checks']]
    nfields=[v for x in nchecks for v in x['fields'].values()]
    nr=dict(calls=len(nchecks),within_fixed_gate=sum(x['pass_all'] for x in nchecks),bitwise=sum(x['bitwise'] for x in nchecks),max_abs=max(x['max_abs'] for x in nfields),max_scaled=max(x['max_scaled_error'] for x in nfields),input_sets=len(noise['cases']),input_unchanged=all(x['input_unchanged'] for x in noise['cases']))
    qual={}
    for name,q in [('initial',q1),('keep_empty',q2)]:
        qual[name]=dict(all_pass=q['all_pass'],cases=len(q['cases']),candidate_pass=sum(x['checks']['candidate']['pass_all'] for x in q['cases']),candidate_byte=sum(x['checks']['candidate']['bitwise'] for x in q['cases']),
            source_byte=sum(x['checks']['source_replay']['bitwise'] for x in q['cases']),capture_forward_byte=sum(x['checks']['capture_forward']['bitwise'] for x in q['cases']),
            candidate_forward_byte=sum(x['checks']['candidate']['fields']['output']['bitwise'] for x in q['cases']),candidate_max_abs=max(v['max_abs'] for x in q['cases'] for v in x['checks']['candidate']['fields'].values()),candidate_max_scaled=max(v['max_scaled_error'] for x in q['cases'] for v in x['checks']['candidate']['fields'].values()),elements=sum(x['checks']['candidate']['elements'] for x in q['cases']))
    result=dict(rows=rows,groups=groups,noise=nr,qualification=qual,local_only=True,whole_training_qualified=False,dense_local_1_10_gate=groups['source_dense']['geomean']>=1.1,intervals=len(timing),calls=sum(x['calls'] for x in timing))
    (ROOT/'analysis/summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['|batch_size|层数|位置/attention 层|源路径|投影保留行比例|源毫秒|候选毫秒|配对比值 [95%区间]|','|---:|---:|---|---|---:|---:|---:|---|']
    for x in rows:
        c=x['port'];keep=x['retained_dense_rows']/c['rows'] if not c['source_packed'] else c['valid_ratio'];ci=x['ratio_ci95']
        lines.append(f"|{c['case']['bs']}|{c['case']['layers']}|{c['position']}/{c['module']}|{'packed' if c['source_packed'] else 'dense'}|{100*keep:.1f}%|{x['source_ms']:.3f}|{x['candidate_ms']:.3f}|{x['ratio']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]|")
    (ROOT/'analysis/table.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:result[k] for k in ['groups','noise','qualification','dense_local_1_10_gate']}))

if __name__=='__main__':main()
