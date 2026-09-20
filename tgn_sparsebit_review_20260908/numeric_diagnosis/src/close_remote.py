import datetime
import hashlib
import json
from pathlib import Path
import subprocess

root = Path('/home/data/wangxuran/tncn_numeric_diagnosis_20260908')
old = Path('/home/data/wangxuran/tncn_training_profile_20260908')
project = Path('/home/data/wangxuran/factor_tgn_sptc_20260902/project')


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''): h.update(chunk)
    return h.hexdigest()


prior = json.loads((root/'prior_manifest.json').read_text())
checked, mismatches = 0, []
for category, base in [('files',old),('runtime_sources',project)]:
    for name, expected in prior[category].items():
        file = base/name
        value = expected['sha256'] if isinstance(expected,dict) else expected
        if not file.exists() or sha(file)!=value:
            mismatches.append(str(file))
        checked += 1
result = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(), files={},
              previous_evidence_check=dict(files=checked, mismatches=mismatches),
              prior_manifest_sha256=sha(root/'prior_manifest.json'))
for file in sorted(root.rglob('*')):
    if file.is_file() and file.name!='remote_manifest.json' and '__pycache__' not in file.parts:
        result['files'][str(file.relative_to(root))] = dict(bytes=file.stat().st_size,sha256=sha(file))
result['gpu_final'] = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
result['processes_final'] = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True)
assert 'GPU-16f27f5a-dfcd-48e0-bb39-bebbe4009245' not in result['processes_final']
(root/'remote_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(previous_checked=checked,mismatches=mismatches,new_files=len(result['files']),
                     new_bytes=sum(r['bytes'] for r in result['files'].values()))))
assert not mismatches
