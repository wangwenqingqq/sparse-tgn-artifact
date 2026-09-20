"""Independent CPU/NumPy byte checks; no training imports or CUDA context."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,'/home/data/wangxuran/tncn_numeric_diagnosis_20260908/src')
import audit as prior
ROOT=Path(__file__).resolve().parents[1]
ART=Path('/home/data/wangxuran/tncn_memory_profile_20260909/output/qualify_v1/artifacts')
SOURCE=Path('/home/data/wangxuran/tncn_training_profile_20260908/output/full_v5/artifacts')
VARIANTS=['direct_nograd','batch_nograd','batch_store','direct']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    assert not torch.cuda.is_initialized();root=ROOT/'output'/args.run
    logs=[json.loads(s) for s in (root/'stdout.jsonl').read_text().splitlines()]
    result=dict(checks=[],hashes={},cuda_initialized=False)
    for file in sorted((root/'artifacts').glob('*_checks.pt')):
        variant=next(v for v in VARIANTS if file.name.endswith('_'+v+'_checks.pt'))
        label=file.name.removesuffix('_'+variant+'_checks.pt')
        actual=prior.load(file);reference=prior.load(ART/(label+'_direct_checks.pt'))
        source=prior.load(SOURCE/(label+'_source_replay_checks.pt'))
        assert len(actual)==len(reference)==len(source)==8
        logged=next(r for r in logs if r['kind']=='qualification' and r['case']==label and r['variant']==variant)
        for j,(a,b,s) in enumerate(zip(actual,reference,source)):
            assert len(a)==len(b)==119 and len(s)==107
            cmp=prior.comp(a,b);original=prior.comp({k:a[k] for k in s},s)
            assert cmp==logged['checks'][j]['comparison'] and original==logged['checks'][j]['source']
            result['checks'].append(dict(case=label,variant=variant,step_offset=j,comparison=cmp,source=original))
        h=hashlib.sha256()
        with file.open('rb') as f:
            for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
        result['hashes'][file.name]=dict(bytes=file.stat().st_size,sha256=h.hexdigest())
    result['summary']={}
    for variant in VARIANTS:
        checks=[r for r in result['checks'] if r['variant']==variant]
        result['summary'][variant]=dict(steps=len(checks),bitwise=sum(r['comparison']['bitwise_equal'] for r in checks),
            source_bitwise=sum(r['source']['bitwise_equal'] for r in checks),elements=sum(r['comparison']['elements'] for r in checks))
    assert not torch.cuda.is_initialized()
    (root/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary']))
    assert all(v['steps']==96 for v in result['summary'].values())


if __name__=='__main__':main()
