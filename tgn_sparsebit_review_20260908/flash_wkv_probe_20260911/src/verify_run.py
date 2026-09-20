"""Independent CPU archive audit. No FlashTGN imports or CUDA initialization."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]


def check(actual,reference,logged):
    assert actual.keys()==reference.keys()==logged['fields'].keys()
    for name,x in actual.items():
        y=reference[name];l=logged['fields'][name];assert x.shape==y.shape and x.dtype==y.dtype
        xx=x.contiguous().numpy().reshape(-1);yy=y.contiguous().numpy().reshape(-1);bad=0;exact=True;different=0;finite=True;mx=0.;scaled=0.
        # Deliberately different partitioning and np.isclose implementation.
        for start in range(0,xx.size,700001):
            a=xx[start:start+700001];b=yy[start:start+700001]
            exact &= memoryview(a).tobytes()==memoryview(b).tobytes();different+=int(np.sum(a!=b))
            if a.dtype.kind=='f':
                aa=a.astype('float64');bb=b.astype('float64');good=np.isfinite(aa)&np.isfinite(bb);finite &= bool(good.all())
                bad+=int(np.sum(~good|~np.isclose(aa,bb,atol=.0002,rtol=.0002)))
                if len(aa):mx=max(mx,float(np.max(np.abs(aa-bb))));scaled=max(scaled,float(np.max(np.abs(aa-bb)/(.0002+.0002*np.abs(bb)))))
            else:bad+=int(np.sum(a!=b))
        assert bad==l['failed_elements'] and exact==l['bitwise'] and different==l['different'] and finite==l['finite']
        assert mx==l['max_abs'] and scaled==l['max_scaled_error']
    assert logged['pass_all']==all(v['pass_all'] for v in logged['fields'].values())
    return logged['pass_all']


def sha(f):
    h=hashlib.sha256()
    with f.open('rb') as stream:
        for x in iter(lambda:stream.read(1<<20),b''):h.update(x)
    return dict(bytes=f.stat().st_size,sha256=h.hexdigest())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--stage',choices=['qualify','noise'],required=True);args=ap.parse_args();assert not torch.cuda.is_initialized()
    root=ROOT/'output'/args.run;art=root/'artifacts';meta=json.loads((ROOT/'output/capture_v1/artifacts/capture.json').read_text());hashes={};checked=0;passes=0
    if args.stage=='qualify':
        q=json.loads((art/'qualification.json').read_text())
        for row in q['cases']:
            port=row['port'];raw=torch.load(ROOT/'output/capture_v1/artifacts'/port['input_file'],map_location='cpu',weights_only=False)
            mask=(raw['nbr'].numpy()>=0).ravel();assert int(mask.sum())==port['valid'] and len(mask)==port['rows']
            rawshape=raw['z'].shape;assert rawshape[0]==len(mask)
            ref=torch.load(art/(port['label']+'_source_a.pt'),map_location='cpu',weights_only=False)
            passes+=check({'output':ref['output']},{'output':raw['source_output']},row['checks']['capture_forward']);checked+=1
            for arm,key in [('source_b','source_replay'),('candidate','candidate')]:
                actual=torch.load(art/(port['label']+'_'+arm+'.pt'),map_location='cpu',weights_only=False)
                passes+=check(actual,ref,row['checks'][key]);checked+=1
        assert q['all_pass']==(passes==checked)
    else:
        q=json.loads((art/'noise.json').read_text())
        for row in q['cases']:
            info=row['info'];raw=torch.load(ROOT/'output/capture_v1/artifacts'/info['input_file'],map_location='cpu',weights_only=False)
            actual=torch.load(art/(info['label']+'_noise_outputs.pt'),map_location='cpu',weights_only=False)
            assert len(actual)==16
            ref={f'output_{i}':x for i,x in enumerate(raw['source_outputs'])}
            for observation,logged in zip(actual,row['checks']):passes+=check({f'output_{i}':x for i,x in enumerate(observation)},ref,logged);checked+=1
    for f in art.glob('*.pt'):hashes[f.name]=sha(f)
    result=dict(stage=args.stage,all_logged_comparisons_match=True,all_pass=passes==checked,comparisons=checked,passed=passes,hashes=hashes,cuda_initialized=False)
    assert not torch.cuda.is_initialized();(root/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['stage','all_logged_comparisons_match','all_pass','comparisons','passed']}))

if __name__=='__main__':main()
