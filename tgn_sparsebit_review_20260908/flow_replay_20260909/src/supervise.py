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
GPUS = {2:'GPU-16f27f5a-dfcd-48e0-bb39-bebbe4009245',
        3:'GPU-a149f5af-55ab-ce33-8d3d-371a7ae61dd2',
        4:'GPU-863c06a5-9f33-0265-b098-013fa840d5db'}
parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
parser.add_argument('--stage', required=True)
parser.add_argument('--dataset', default='both')
parser.add_argument('--artifact-dir', default='main')
parser.add_argument('--epochs', default='5')
parser.add_argument('--steps', default='1500')
parser.add_argument('--rounds', default='7')
parser.add_argument('--gpu', type=int, choices=[2,3,4], default=2)
args = parser.parse_args()
GPU = GPUS[args.gpu]
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
with open(f'/home/data/wangxuran/.locks/tgn_sptc_gpu{args.gpu}.lock', 'a+') as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        g, a = snap('before')
        row = next(r for r in g.splitlines() if GPU in r).split(',')
        assert int(row[-2]) <= 64 and int(row[-1]) == 0 and GPU not in a, 'GPU not idle'
        stages = ['samples','fit','evaluate','audit','timing'] if args.stage == 'pipeline' else [args.stage]
        record['stages'] = []
        for stage in stages:
            current_env = env
            if stage == 'audit':
                cmd = ['/home/data/wangxuran/isaacsim6/env/bin/python', str(ROOT/'src/audit.py'),
                       '--input',str(ROOT/'artifacts'/args.artifact_dir),'--output',str(out/'audit.json')]
                current_env = dict(env, CUDA_VISIBLE_DEVICES='')
            else:
                cmd = ['/home/data/wangxuran/isaacsim6/env/bin/python',str(ROOT/'src/experiment.py'),
                       '--stage',stage,'--dataset',args.dataset,'--out',str(ROOT/'artifacts'/args.artifact_dir),
                       '--epochs',args.epochs,'--steps',args.steps,'--rounds',args.rounds]
            prefix = stage+'_' if args.stage == 'pipeline' else ''
            with open(out/(prefix+'stdout.jsonl'),'w') as stdout, open(out/(prefix+'stderr.log'),'w') as stderr:
                record['exit_code'] = subprocess.run(cmd, env=current_env,
                    cwd='/home/data/wangxuran/factor_tgn_sptc_20260902/project', stdout=stdout, stderr=stderr,
                    timeout=7200).returncode
            record['stages'].append(dict(stage=stage,command=cmd,exit_code=record['exit_code']))
            (out/'progress.json').write_text(json.dumps(record,indent=2)+'\n')
            if record['exit_code'] != 0:
                break
    except Exception as exc:
        record['error'] = repr(exc); record['exit_code'] = -1
    finally:
        snap('after'); record['end'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (out/'run.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
sys.exit(record['exit_code'])
