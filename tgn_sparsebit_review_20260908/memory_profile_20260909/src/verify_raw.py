"""CPU/NumPy byte audit of complete logical states, including message caches and RNG."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,'/home/data/wangxuran/tncn_numeric_diagnosis_20260908/src')
import audit as prior
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/home/data/wangxuran/tncn_training_profile_20260908/output/full_v5/artifacts')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    assert not torch.cuda.is_initialized();root=ROOT/'output'/args.run
    result=dict(checks=[],hashes={},cuda_initialized=False)
    for file in sorted((root/'artifacts').glob('*_direct_checks.pt')):
        label=file.name.removesuffix('_direct_checks.pt')
        base=prior.load(file);candidate=prior.load(root/'artifacts'/(label+'_batch_store_checks.pt'))
        source=prior.load(OLD/(label+'_source_replay_checks.pt'))
        assert len(base)==len(candidate)==len(source)==8
        for j,(a,b,s) in enumerate(zip(base,candidate,source)):
            assert len(a)==len(b)==119 and len(s)==107
            result['checks'].append(dict(case=label,step_offset=j,baseline_source=prior.comp({k:a[k] for k in s},s),
                candidate=prior.comp(b,a),candidate_source=prior.comp({k:b[k] for k in s},s)))
    for file in sorted((root/'artifacts').glob('*.pt')):
        h=hashlib.sha256()
        with file.open('rb') as f:
            for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
        result['hashes'][file.name]=dict(bytes=file.stat().st_size,sha256=h.hexdigest())
    result['summary']={}
    for key in ['baseline_source','candidate','candidate_source']:
        checks=[r[key] for r in result['checks']]
        result['summary'][key]=dict(steps=len(checks),bitwise=sum(c['bitwise_equal'] for c in checks),
            elements=sum(c['elements'] for c in checks))
    assert not torch.cuda.is_initialized()
    (root/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary']))
    assert all(v['bitwise']==96 for v in result['summary'].values())


if __name__=='__main__':main()
