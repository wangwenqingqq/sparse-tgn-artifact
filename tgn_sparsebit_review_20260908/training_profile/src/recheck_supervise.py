import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
p.add_argument('--smoke', action='store_true')
p.add_argument('--dataset', default='both')
p.add_argument('--batch', default='both')
args = p.parse_args()
root = Path(__file__).resolve().parents[1]
out = root/'output'/args.run
out.mkdir(parents=True, exist_ok=False)
gpu = 'GPU-16f27f5a-dfcd-48e0-bb39-bebbe4009245'
env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONDONTWRITEBYTECODE='1',
           PYTHONPATH=str(root/'deps'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           CUBLAS_WORKSPACE_CONFIG=':4096:8')

def snap(name):
    g = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    a = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv,noheader,nounits'], text=True)
    (out/(name+'.txt')).write_text(g+'\nPROCESSES\n'+a)
    return g, a

record = dict(start=datetime.datetime.now(datetime.timezone.utc).isoformat(), gpu=gpu, args=vars(args))
with open('/home/data/wangxuran/.locks/tgn_sptc_gpu2.lock', 'a+') as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        g, a = snap('before')
        row = [r for r in g.splitlines() if gpu in r][0].split(',')
        assert int(row[-2]) <= 64 and int(row[-1]) == 0 and gpu not in a, 'GPU not idle'
        cmd = ['/home/data/wangxuran/isaacsim6/env/bin/python', str(root/'src/timing_recheck.py'),
               '--output', str(out/'artifacts'), '--dataset', args.dataset, '--batch', args.batch]
        if args.smoke:
            cmd.append('--smoke')
        record['command'] = cmd
        with open(out/'stdout.jsonl', 'w') as stdout, open(out/'stderr.log', 'w') as stderr:
            record['exit_code'] = subprocess.run(cmd, env=env,
                cwd='/home/data/wangxuran/factor_tgn_sptc_20260902/project',
                stdout=stdout, stderr=stderr, timeout=3600).returncode
    except Exception as exc:
        record['error'] = repr(exc)
        record['exit_code'] = -1
    finally:
        snap('after')
        record['end'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (out/'run.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record), flush=True)
sys.exit(record['exit_code'])
