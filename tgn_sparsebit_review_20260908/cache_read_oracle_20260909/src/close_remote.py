import datetime
import hashlib
import json
from pathlib import Path
import subprocess

root=Path('/home/data/wangxuran/tncn_cache_read_oracle_20260909')
nograd=Path('/home/data/wangxuran/tncn_state_nograd_20260909')
memory=Path('/home/data/wangxuran/tncn_memory_profile_20260909')
csr=Path('/home/data/wangxuran/tncn_compatible_csr_20260908')
profile=Path('/home/data/wangxuran/tncn_training_profile_20260908')
diagnosis=Path('/home/data/wangxuran/tncn_numeric_diagnosis_20260908')
project=Path('/home/data/wangxuran/factor_tgn_sptc_20260902/project')


def sha(file):
    h=hashlib.sha256()
    with file.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


checked=0;mismatches=[]
for name,base in [('previous_profile.json',profile),('previous_diagnosis.json',diagnosis),('previous_csr.json',csr),('previous_memory.json',memory),('previous_nograd.json',nograd)]:
    prior=json.loads((root/name).read_text())
    for category,where in [('files',base),('runtime_sources',project)]:
        for rel,expected in prior.get(category,{}).items():
            file=where/rel
            value=expected['sha256'] if isinstance(expected,dict) else expected
            if not file.exists() or sha(file)!=value:mismatches.append(str(file))
            checked+=1
result=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            previous_check=dict(files=checked,mismatches=mismatches),files={})
for file in sorted(root.rglob('*')):
    if file.is_file() and file.name!='remote_manifest.json' and '__pycache__' not in file.parts:
        result['files'][str(file.relative_to(root))]=dict(bytes=file.stat().st_size,sha256=sha(file))
result['gpu_final']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
result['processes_final']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True)
assert 'GPU-a149f5af-55ab-ce33-8d3d-371a7ae61dd2' not in result['processes_final']
(root/'remote_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(previous_files=checked,mismatches=mismatches,new_files=len(result['files']),new_bytes=sum(x['bytes'] for x in result['files'].values()))))
assert not mismatches
