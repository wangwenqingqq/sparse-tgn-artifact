"""CPU/NumPy state audit and independent reconstruction of the oracle prefixes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,'/home/data/wangxuran/tncn_numeric_diagnosis_20260908/src')
import audit as prior
ROOT=Path(__file__).resolve().parents[1]
ART=Path('/home/data/wangxuran/tncn_memory_profile_20260909/output/qualify_v1/artifacts')
SOURCE=Path('/home/data/wangxuran/tncn_training_profile_20260908/output/full_v5/artifacts')
VARIANTS=['free_cache_reads','batch_store','capture']
FIELDS=[('src','src'),('dst','dst'),('t','time'),('raw_msg','message')]


def prefix_check(records,checkpoint,reference):
    assert len(records)==32;result=[]
    for pos,row in enumerate(records):
        step=pos//4;direction=row['direction'];ids=row['n_id'].numpy()
        assert direction==['s','d','s','d'][pos%4] and ids.dtype==np.int64
        if step==0:
            store=checkpoint['memory'][2 if direction=='s' else 3]
            expected={field:np.concatenate([store[int(node)][j].numpy() for node in ids],axis=0) for j,(field,_) in enumerate(FIELDS)}
        else:
            previous=reference[step-1];prefix='cache/'+direction+'/'
            lengths=previous[prefix+'lengths'].numpy();offsets=np.concatenate([np.zeros(1,np.int64),np.cumsum(lengths,dtype=np.int64)])
            assert np.all((ids>=0)&(ids<len(lengths)))
            expected={field:np.concatenate([previous[prefix+key].numpy()[offsets[node]:offsets[node+1]] for node in ids],axis=0) for field,key in FIELDS}
        different=[];elements=0
        for field,_ in FIELDS:
            assert not row[field].requires_grad
            actual=row[field].numpy();want=expected[field];elements+=actual.size
            if actual.shape!=want.shape or actual.dtype!=want.dtype or actual.tobytes()!=want.tobytes():different.append(field)
        result.append(dict(call=pos,step_offset=step,direction=direction,nodes=len(ids),message_rows=len(row['src']),
            exact=not different,different_fields=different,elements=elements))
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    assert not torch.cuda.is_initialized();root=ROOT/'output'/args.run
    logs=[json.loads(s) for s in (root/'stdout.jsonl').read_text().splitlines()]
    result=dict(training=[],prefix=[],hashes={},cuda_initialized=False)
    for file in sorted((root/'artifacts').glob('*_tape.pt')):
        label=file.name.removesuffix('_tape.pt')
        reference=prior.load(ART/(label+'_direct_checks.pt'));source=prior.load(SOURCE/(label+'_source_replay_checks.pt'))
        checkpoint=prior.load(SOURCE/(label+'_checkpoint.pt'))
        result['prefix'].append(dict(case=label,checks=prefix_check(prior.load(file),checkpoint,reference)))
        for variant in VARIANTS:
            actual=prior.load(root/'artifacts'/(label+'_'+variant+'_checks.pt'))
            assert len(actual)==len(reference)==len(source)==8
            logged=next(r for r in logs if r['kind']=='qualification' and r['case']==label and r['variant']==variant)
            for j,(a,b,s) in enumerate(zip(actual,reference,source)):
                assert len(a)==len(b)==119 and len(s)==107
                cmp=prior.comp(a,b);original=prior.comp({k:a[k] for k in s},s)
                assert cmp==logged['checks'][j]['comparison'] and original==logged['checks'][j]['source']
                result['training'].append(dict(case=label,variant=variant,step_offset=j,comparison=cmp,source=original))
    for file in sorted((root/'artifacts').glob('*.pt')):
        h=hashlib.sha256()
        with file.open('rb') as f:
            for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
        result['hashes'][file.name]=dict(bytes=file.stat().st_size,sha256=h.hexdigest())
    result['summary']={}
    for variant in VARIANTS:
        checks=[r for r in result['training'] if r['variant']==variant]
        result['summary'][variant]=dict(steps=len(checks),bitwise=sum(r['comparison']['bitwise_equal'] for r in checks),
            source_bitwise=sum(r['source']['bitwise_equal'] for r in checks),elements=sum(r['comparison']['elements'] for r in checks))
    checks=[c for r in result['prefix'] for c in r['checks']]
    result['summary']['prefix']=dict(calls=len(checks),exact=all(c['exact'] for c in checks),elements=sum(c['elements'] for c in checks))
    assert not torch.cuda.is_initialized()
    (root/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary']))
    assert len(checks)==384


if __name__=='__main__':main()
