"""Run only this experiment's executable under the existing per-GPU flock."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
p.add_argument('--stage', choices=['small', 'full'], required=True)
p.add_argument('--run', required=True)
p.add_argument('--binary', default='compile/probe_sm120a')
args = p.parse_args()
root = Path(args.root).resolve()
out = root / 'output' / args.run
out.mkdir(exist_ok=False)
gpu = 'GPU-16f27f5a-dfcd-48e0-bb39-bebbe4009245'
lock = Path('/home/data/wangxuran/.locks/tgn_sptc_gpu2.lock')
env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
exe = root / args.binary
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
record = dict(start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              gpu=gpu, lock=str(lock), stage=args.stage,
              files={str(f.relative_to(root)): sha(f) for f in
                     [root / 'src/layout_probe.cu', root / 'src/CONTRACT.md', exe]},
              steps=[])


def save():
    (out / 'run.json').write_text(json.dumps(record, indent=2) + '\n')


def snapshot(label):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
    processes = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
    (out / (label + '.txt')).write_text(result + '\nPROCESSES\n' + processes)
    return result, processes


def run(label, command, seconds, archive=False):
    runenv = dict(env)
    if archive:
        raw = out / (label + '_raw')
        raw.mkdir()
        runenv['LAYOUT_ARCHIVE_DIR'] = str(raw)
    step = dict(label=label, command=command, timeout=seconds,
                start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    record['steps'].append(step)
    save()
    with (out / (label + '.stdout')).open('w') as stdout, (out / (label + '.stderr')).open('w') as stderr:
        try:
            result = subprocess.run(command, cwd=root, env=runenv, stdout=stdout, stderr=stderr, timeout=seconds)
            step['exit_code'] = result.returncode
        except subprocess.TimeoutExpired:
            step['exit_code'] = -999
    step['end_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
    print(json.dumps(step), flush=True)
    if step['exit_code'] != 0:
        raise RuntimeError('Failed stage: ' + label)


save()
with lock.open('a+') as held:
    try:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, processes = snapshot('before')
        target = [r for r in state.splitlines() if gpu in r]
        if len(target) != 1:
            raise RuntimeError('GPU UUID missing/ambiguous')
        fields = [s.strip() for s in target[0].split(',')]
        if int(fields[-2]) > 64 or int(fields[-1]) != 0 or gpu in processes:
            raise RuntimeError('GPU not idle')
        if args.stage == 'small':
            run('small', [str(exe), 'small'], 90, archive=True)
        else:
            run('correctness', [str(exe), 'check'], 180, archive=True)
            for tool in ['memcheck', 'racecheck', 'initcheck', 'synccheck']:
                run(tool, ['/usr/local/cuda/bin/compute-sanitizer', '--tool', tool,
                           '--error-exitcode', '91', str(exe), 'small'], 180)
            run('benchmark', [str(exe), 'bench'], 360)
        record['status'] = 'complete'
    except Exception as exc:
        record['status'] = 'failed'
        record['error'] = repr(exc)
        print(repr(exc), file=sys.stderr)
    finally:
        snapshot('after')
        record['end_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
sys.exit(0 if record['status'] == 'complete' else 1)
