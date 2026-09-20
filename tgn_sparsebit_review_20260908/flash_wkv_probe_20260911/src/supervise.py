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
p.add_argument('--stage', default='qualify')
p.add_argument('--qualification')
p.add_argument('--script',required=True)
p.add_argument('--rep',type=int,default=0)
args = p.parse_args()
root = Path(__file__).resolve().parents[1]
out = root/'output'/args.run
out.mkdir(parents=True, exist_ok=False)
gpu = 'GPU-9bd364a9-9d88-0b78-ef55-caac6f73d6b1'
env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONDONTWRITEBYTECODE='1',
           PYTHONPATH='/home/data/wangxuran/flash_tgn_rebaseline_20260909/deps'+':/home/data/wangxuran/factor_tgn_sptc_20260902/vendor/flash-tgn-artifact/python', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           CUBLAS_WORKSPACE_CONFIG=':4096:8')
for key in list(env):
    if key.startswith('FLASH_TGN_'):env.pop(key)
env.update(FLASH_TGN_STRICT_FUSED='1',FLASH_TGN_ATTN_LOG='0' if args.stage=='diagnostic_timing' else '1',FLASH_TGN_PROFILE='0')

def snap(name):
    g = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    a = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv,noheader,nounits'], text=True)
    (out/(name+'.txt')).write_text(g+'\nPROCESSES\n'+a)
    return g, a

record = dict(start=datetime.datetime.now(datetime.timezone.utc).isoformat(), gpu=gpu, args=vars(args))
with open('/home/data/wangxuran/.locks/flash_rebaseline_gpu0.lock', 'a+') as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        g, a = snap('before')
        row = [r for r in g.splitlines() if gpu in r][0].split(',')
        assert int(row[-2]) <= 64 and int(row[-1]) == 0 and gpu not in a, 'GPU not idle'
        cmd = ['/home/data/wangxuran/isaacsim6/env/bin/python', str(root/'src'/args.script),
               '--output', str(out/'artifacts'), '--stage', args.stage]
        if args.qualification:
            cmd += ['--qualification', args.qualification]
        record['command'] = cmd
        with open(out/'stdout.jsonl', 'w') as stdout, open(out/'stderr.log', 'w') as stderr:
            record['exit_code'] = subprocess.run(cmd, env=env,
                cwd='/home/data/wangxuran/factor_tgn_sptc_20260902/vendor/flash-tgn-artifact',
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
