"""Independent CPU audit: self replay, profiling replay and temporal sampling."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/home/data/wangxuran/factor_tgn_sptc_20260902/project/experiments/factor_tgn_20260904_anchored_time_quality_wikipedia_full/data_root/data/wiki')


def compare(a,b):
    assert set(a)==set(b)
    different=[];failed=[];elements=0
    for key in sorted(a):
        x,y=a[key],b[key]
        if torch.is_tensor(x):
            x=x.numpy();y=y.numpy()
            assert x.shape==y.shape and x.dtype==y.dtype
            elements+=x.size
            exact=memoryview(np.ascontiguousarray(x)).tobytes()==memoryview(np.ascontiguousarray(y)).tobytes()
            if not exact:different.append(key)
            if x.dtype.kind=='f':
                accepted=np.isfinite(x).all() and np.isfinite(y).all() and np.allclose(x.astype('float64'),y.astype('float64'),rtol=2e-4,atol=2e-4)
            else:accepted=exact
            if not accepted:failed.append(key)
        elif x!=y:different.append(key);failed.append(key)
    return dict(pass_all=not failed,bitwise_equal=not different,fields=len(a),elements=elements,
        different_fields=different,failed_fields=failed)


def sample_check(records):
    ind,nbr,eid,ets=[np.load(DATA/('edges.undirected_tcsr.'+name+'.npy')) for name in ['ind','nbr','eid','ets']]
    result=[]
    for call,row in enumerate(records):
        nodes=row['nodes'].numpy();times=row['times'].numpy();k=row['k']
        wants=[np.full((len(nodes),k),-1,dtype=np.int32),np.full((len(nodes),k),-1,dtype=np.int32),np.zeros((len(nodes),k),dtype=np.float32)]
        for j,(node,t) in enumerate(zip(nodes,times)):
            begin,end=ind[int(node):int(node)+2]
            stop=begin+np.searchsorted(ets[begin:end],t,side='left')
            start=max(begin,stop-k);take=stop-start
            wants[0][j,:take]=nbr[start:stop];wants[1][j,:take]=eid[start:stop];wants[2][j,:take]=ets[start:stop]
        fields=['neighbors','events','timestamps']
        bad=[name for name,want in zip(fields,wants) if row[name].numpy().dtype!=want.dtype or row[name].numpy().tobytes()!=want.tobytes()]
        result.append(dict(call=call,queries=len(nodes),total_call_queries=row['total_queries'],k=k,pass_all=not bad,different=bad))
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    assert not torch.cuda.is_initialized()
    root=ROOT/'output'/args.run;art=root/'artifacts';q=json.loads((art/'qualification.json').read_text())
    result=dict(cases=[],hashes={},cuda_initialized=False)
    for case in q['cases']:
        label=case['case']['label']
        source=torch.load(art/(label+'_source_a.pt'),map_location='cpu',weights_only=False)
        row=dict(case=case['case'],comparisons={})
        for arm in ['source_b','instrumented']:
            actual=torch.load(art/(label+'_'+arm+'.pt'),map_location='cpu',weights_only=False)
            assert len(actual)==len(source)==4
            checks=[compare(a,b) for a,b in zip(actual,source)]
            for logged,check in zip(case['comparisons'][arm],checks):
                for key in ['pass_all','bitwise_equal','fields','elements']:assert logged[key]==check[key],(label,arm,key)
                assert sorted(logged['different_fields'])==check['different_fields']
                assert sorted(x['field'] for x in logged['failures'])==check['failed_fields']
            row['comparisons'][arm]=checks
        records=torch.load(art/(label+'_sampler.pt'),map_location='cpu',weights_only=False)
        row['sampler']=sample_check(records);result['cases'].append(row)
    for f in art.glob('*.pt'):
        h=hashlib.sha256()
        with f.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
        result['hashes'][f.name]=dict(bytes=f.stat().st_size,sha256=h.hexdigest())
    checks=[c for r in result['cases'] for rows in r['comparisons'].values() for c in rows]
    samples=[c for r in result['cases'] for c in r['sampler']]
    result['all_pass']=all(c['pass_all'] for c in checks)
    result['comparisons']=dict(steps=len(checks),passed=sum(c['pass_all'] for c in checks),bitwise=sum(c['bitwise_equal'] for c in checks))
    result['sampler']=dict(calls=len(samples),queries=sum(c['queries'] for c in samples),all_pass=all(c['pass_all'] for c in samples))
    assert not torch.cuda.is_initialized()
    (root/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['all_pass','comparisons','sampler']}))


if __name__=='__main__':main()
