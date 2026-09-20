"""Run each stage under the existing GPU2 lock and record exact provenance."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
GPU = 'GPU-16f27f5a-dfcd-48e0-bb39-bebbe4009245'
parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
parser.add_argument('--stage', required=True)
parser.add_argument('--dataset', default='both')
parser.add_argument('--artifact-dir', default='main')
parser.add_argument('--epochs', default='5')
parser.add_argument('--steps', default='1500')
parser.add_argument('--rounds', default='7')
args = parser.parse_args()
out = ROOT/'runs'/args.run
out.mkdir(parents=True, exist_ok=False)
(out/'sources').mkdir()
for source in (ROOT/'src').glob('*.py'):
    shutil.copy2(source, out/'sources'/source.name)
shutil.copy2(ROOT/'CONTRACT.md', out/'CONTRACT.md')
env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, PYTHONDONTWRITEBYTECODE='1',
           PYTHONPATH='/home/data/wangxuran/tncn_training_profile_20260908/deps',
           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')


def snap(name):
    g = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'], text=True)
    a = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'], text=True)
    (out/(name+'.txt')).write_text(g+'\nPROCESSES\n'+a)
    return g, a


record = dict(start=datetime.datetime.now(datetime.timezone.utc).isoformat(), gpu=GPU, args=vars(args),
              sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')},
              contract_sha256=hashlib.sha256((ROOT/'CONTRACT.md').read_bytes()).hexdigest())
imports = [
    '/home/data/wangxuran/tncn_memory_profile_20260909/src/instrument.py',
    '/home/data/wangxuran/tncn_compatible_csr_20260908/src/bridge.py',
    '/home/data/wangxuran/tncn_numeric_diagnosis_20260908/src/diagnose.py',
    '/home/data/wangxuran/tncn_training_profile_20260908/src/profile_training.py',
]
record['import_hashes'] = {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in imports}
with open('/home/data/wangxuran/.locks/tgn_sptc_gpu2.lock', 'a+') as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        g, a = snap('before')
        row = next(r for r in g.splitlines() if GPU in r).split(',')
        assert int(row[-2]) <= 64 and int(row[-1]) == 0 and GPU not in a, 'GPU not idle'
        cmd = ['/home/data/wangxuran/isaacsim6/env/bin/python',str(ROOT/'src/experiment.py'),
               '--stage',args.stage,'--dataset',args.dataset,'--out',str(ROOT/'artifacts'/args.artifact_dir),
               '--epochs',args.epochs,'--steps',args.steps,'--rounds',args.rounds]
        record['command'] = cmd
        with open(out/'stdout.jsonl','w') as stdout, open(out/'stderr.log','w') as stderr:
            record['exit_code'] = subprocess.run(cmd, env=env,
                cwd='/home/data/wangxuran/factor_tgn_sptc_20260902/project', stdout=stdout, stderr=stderr,
                timeout=7200).returncode
    except Exception as exc:
        record['error'] = repr(exc); record['exit_code'] = -1
    finally:
        snap('after'); record['end'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (out/'run.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
sys.exit(record['exit_code'])
