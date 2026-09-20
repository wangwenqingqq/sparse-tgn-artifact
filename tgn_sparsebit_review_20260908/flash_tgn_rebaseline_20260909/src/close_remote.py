import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]

def sha(file):
    h=hashlib.sha256()
    with file.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()

m=json.loads((ROOT/'input_manifest.json').read_text());bad=[]
for path,info in m['files'].items():
    if sha(Path(path))!=info['sha256']:bad.append(path)
assert not bad,bad
result=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),input_check=dict(files=len(m['files']),mismatches=bad),files={})
for f in sorted(ROOT.rglob('*')):
    if not f.is_file() or '__pycache__' in f.parts or f.name=='remote_manifest.json':continue
    result['files'][str(f.relative_to(ROOT))]=dict(bytes=f.stat().st_size,sha256=sha(f))
result['gpu_final']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
result['processes_final']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True)
assert 'GPU-9bd364a9-9d88-0b78-ef55-caac6f73d6b1' not in result['processes_final']
with open('/home/data/wangxuran/.locks/flash_rebaseline_gpu0.lock','a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);result['gpu0_lock_free']=True
(ROOT/'remote_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(input_files=len(m['files']),mismatches=bad,files=len(result['files']),bytes=sum(x['bytes'] for x in result['files'].values()),gpu0_lock_free=True)))
