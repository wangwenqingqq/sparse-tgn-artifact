"""Read-only checks of imported sources, then index this experiment's artifacts."""
import hashlib
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[1]
run=json.loads((root/'runs/pipeline_v1/run.json').read_text())
assert run['exit_code']==0 and all(x['exit_code']==0 for x in run['stages'])
checks={}
for path,expected in run['import_hashes'].items():
    actual=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert actual==expected,path
    checks[path]=actual
for stage in ['samples','fit','evaluate','timing']:
    first=json.loads((root/f'runs/pipeline_v1/{stage}_stdout.jsonl').read_text().splitlines()[0])
    assert first['sha256']==run['sources']['src/experiment.py']
assert json.loads((root/'runs/pipeline_v1/audit.json').read_text())['pass_all']
g=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
a=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True)
own=[]
for line in a.splitlines():
    fields=[s.strip() for s in line.split(',')]
    if fields[0]==run['gpu']:
        pid=fields[1]
        cmd=Path('/proc')/pid/'cmdline'
        if cmd.exists() and 'tncn_flow_replay_20260909' in cmd.read_bytes().decode(errors='replace'):
            own.append(line)
assert not own,own
closure=dict(imports_unchanged=checks,runner_hash=run['sources']['src/experiment.py'],
             all_stages_exit_zero=True,audit_pass=True,own_gpu_processes=own,gpus=g,processes=a)
(root/'closure.json').write_text(json.dumps(closure,indent=2)+'\n')
files=[]
for path in sorted(root.rglob('*')):
    if not path.is_file() or '__pycache__' in path.parts or path.name in ['remote_manifest.json','MANIFEST.sha256']:
        continue
    h=hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda:f.read(1<<20),b''):h.update(buf)
    files.append(dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=h.hexdigest()))
manifest=dict(root=str(root),files=files,total_bytes=sum(x['bytes'] for x in files))
(root/'remote_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(files=len(files),bytes=manifest['total_bytes'],imports_unchanged=True,own_gpu_processes=own)))
