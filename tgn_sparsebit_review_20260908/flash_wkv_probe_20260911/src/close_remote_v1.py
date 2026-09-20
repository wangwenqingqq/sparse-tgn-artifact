import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
PREV=Path('/home/data/wangxuran/flash_tgn_rebaseline_20260909')

def sha(f):
    h=hashlib.sha256()
    with f.open('rb') as stream:
        for part in iter(lambda:stream.read(1<<20),b''):h.update(part)
    return h.hexdigest()

source=json.loads((PREV/'input_manifest.json').read_text());bad=[]
for path,info in source['files'].items():
    if sha(Path(path))!=info['sha256']:bad.append(path)
assert not bad,bad
m=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_input_files=len(source['files']),source_mismatches=bad,files={})
for f in sorted(ROOT.rglob('*')):
    if f.is_file() and '__pycache__' not in f.parts and f.name!='remote_manifest.json':m['files'][str(f.relative_to(ROOT))]=dict(bytes=f.stat().st_size,sha256=sha(f))
m['gpu_final']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
m['processes_final']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True)
assert 'GPU-865ae1f0-780e-d04c-5ec3-4deccea65f82' not in m['processes_final']
with open('/home/data/wangxuran/.locks/flash_wkv_gpu6.lock','a+') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);m['gpu6_lock_free']=True
(ROOT/'remote_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print(json.dumps(dict(files=len(m['files']),bytes=sum(x['bytes'] for x in m['files'].values()),source_files=len(source['files']),source_mismatches=bad,gpu6_lock_free=True)))
