"""CPU checks of complete diagnostic training archives, not quality parity."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]

def main():
    assert not torch.cuda.is_initialized();rows=[];hashes={}
    for name in ['diagnostic_timing_v1','diagnostic_timing_v2','diagnostic_timing_v3','diagnostic_profiles_v1']:
        root=ROOT/'output'/name
        assert json.loads((root/'run.json').read_text())['exit_code']==0
        log=[json.loads(line) for line in (root/'stdout.jsonl').read_text().splitlines() if line.startswith('{')]
        complete=[x for x in log if x['kind']=='training_complete'];epochs=[x for x in log if x['kind']=='epoch']
        assert len(complete)==8 and len(epochs)==(16 if 'profiles' in name else 32)
        for c in complete:
            label=c['case']['label'];f=root/'artifacts'/(label+'_final.pt');raw=torch.load(f,map_location='cpu',weights_only=False)
            assert len(raw)==c['fields'];finite=True;elements=0
            for key,v in raw.items():
                if torch.is_tensor(v):
                    a=v.numpy();elements+=a.size
                    if a.dtype.kind=='f':finite &= bool(np.isfinite(a).all())
            assert finite==c['finite']==True and not c['cuda_errors']
            h=hashlib.sha256()
            with f.open('rb') as stream:
                for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
            hashes[str(f.relative_to(ROOT))]=dict(bytes=f.stat().st_size,sha256=h.hexdigest())
            rows.append(dict(run=name,case=c['case'],fields=len(raw),elements=elements,finite=finite))
    result=dict(all_finite=all(x['finite'] for x in rows),archives=len(rows),rows=rows,hashes=hashes,cuda_initialized=False,accuracy_or_quality_parity=False)
    assert not torch.cuda.is_initialized();(ROOT/'analysis/training_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['all_finite','archives','accuracy_or_quality_parity']}))

if __name__=='__main__':main()
