"""CPU-only independent byte and CSR reconstruction checks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path('/home/data/wangxuran/tncn_compatible_csr_20260908')
OLD=Path('/home/data/wangxuran/tncn_training_profile_20260908/output/full_v5/artifacts')
sys.path.insert(0,'/home/data/wangxuran/tncn_numeric_diagnosis_20260908/src')
import audit as prior


def verify_csr(record):
    n,b=record['n'],record['b']
    expected=np.zeros((7,b,n),dtype=np.int64)
    offset=0
    for piece in record['input_csr']:
        pb=piece['b'];rp,col,iv=[piece[k].numpy() for k in ['rp','col','iv']]
        assert len(rp)==7*pb+1 and rp[0]==0 and rp[-1]==len(col)==len(iv)
        assert np.all(np.diff(rp)>=0) and np.all((col>=0)&(col<n)) and np.all(iv!=0)
        for c in range(7):
            for row in range(pb):
                lo,hi=rp[c*pb+row:c*pb+row+2]
                assert np.all(np.diff(col[lo:hi])>0)
                expected[c,offset+row,col[lo:hi]]=iv[lo:hi]
        offset+=pb
    assert offset==b
    actual=np.zeros((7,b,n),dtype=np.float32)
    elements=0
    for channel,tensors in enumerate(record['csr']):
        rp,col,val=[t.numpy() for t in tensors]
        assert len(rp)==b+1 and rp[0]==0 and rp[-1]==len(col)==len(val)
        assert np.all(np.diff(rp)>=0) and np.all((col>=0)&(col<n)) and np.all(val!=0)
        for row in range(b):
            lo,hi=rp[row:row+2]
            assert np.all(np.diff(col[lo:hi])>0)
            actual[channel,row,col[lo:hi]]=val[lo:hi]
        elements+=rp.size+col.size+val.size
    assert np.array_equal(actual.astype(np.float64),expected.astype(np.float64))
    return dict(exact=True,dense_positions=expected.size,csr_elements=elements,
                nnz=int(np.count_nonzero(expected)),
                old_dense_float_bytes=expected.size*4,n=n,b=b)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='qualify_v1');args=ap.parse_args()
    root=ROOT/'output'/args.run
    rows=[json.loads(s) for s in (root/'stdout.jsonl').read_text().splitlines()]
    result=dict(training=[],csr=[],hashes={},cuda_initialized=False)
    assert not torch.cuda.is_initialized()
    for file in sorted((root/'artifacts').glob('*_checks.pt')):
        stem=file.name.removesuffix('_checks.pt')
        variant=next(v for v in ['fused_bounds','compatible','direct','keeper'] if stem.endswith('_'+v))
        label=stem.removesuffix('_'+variant)
        actual=prior.load(file);reference=prior.load(OLD/(label+'_source_replay_checks.pt'))
        checks=[prior.comp(a,b) for a,b in zip(actual,reference)]
        logged=next(r for r in rows if r['kind']=='qualification' and r['variant']==variant and r['case']==label)
        assert [c['bitwise_equal'] for c in checks]==[c['bitwise_equal'] for c in logged['checks']]
        assert [c['pass_all'] for c in checks]==[c['pass_all'] for c in logged['checks']]
        result['training'].append(dict(case=label,variant=variant,checks=checks))
    for file in sorted((root/'artifacts').glob('*_csr.pt')):
        records=prior.load(file)
        assert len(records)==16
        result['csr'].append(dict(file=file.name,checks=[verify_csr(r) for r in records]))
    for file in sorted((root/'artifacts').glob('*.pt')):
        h=hashlib.sha256()
        with file.open('rb') as f:
            for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
        result['hashes'][file.name]=dict(bytes=file.stat().st_size,sha256=h.hexdigest())
    summary={}
    for variant in ['compatible','direct','fused_bounds','keeper']:
        checks=[c for r in result['training'] if r['variant']==variant for c in r['checks']]
        summary[variant]=dict(total=len(checks),bitwise=sum(c['bitwise_equal'] for c in checks),
            tolerance_pass=sum(c['pass_all'] for c in checks),elements=sum(c['elements'] for c in checks))
    summary['csr']=dict(calls=sum(len(r['checks']) for r in result['csr']),
        dense_positions=sum(c['dense_positions'] for r in result['csr'] for c in r['checks']))
    result['summary']=summary
    assert not torch.cuda.is_initialized()
    (root/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
