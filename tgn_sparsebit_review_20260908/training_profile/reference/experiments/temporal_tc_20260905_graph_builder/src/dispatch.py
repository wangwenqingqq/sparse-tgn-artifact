"""Bounded new-directory transport and serialized GPU2 build/gates/timing."""
import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

EXP=Path(__file__).resolve().parents[1];ROOT=EXP.parent.parent
REL=str(EXP.relative_to(ROOT))
REMOTE='/home/data/wangxuran/factor_tgn_sptc_20260902/project'
PY='/home/data/wangxuran/isaacsim6/env/bin/python'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','pro6000-8-wxr']
PREFIX=f'''cd {REMOTE} && exec 9</home/data/wangxuran/.locks/tgn_sptc_gpu2.lock && flock -n -x 9 && export PATH=/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=GPU-16f27f5a-dfcd-48e0-bb39-bebbe4009245 PYTHONDONTWRITEBYTECODE=1 CUDA_CACHE_DISABLE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 GPU_NT_LIB=$PWD/experiments/temporal_tc_20260905_gpu2_notc/output/gpu2/native_gpu.so GB_LIB=$PWD/{REL}/output/build1/graph_builder.so && '''


def execute(label,command,data=None,timeout=1200):
    dest=EXP/'output'/label;dest.mkdir(exist_ok=False)
    (dest/'command.json').write_text(json.dumps(dict(command=command,input_sha256=hashlib.sha256(data).hexdigest() if data else None),indent=2)+'\n')
    print('START',label,flush=True)
    with (dest/'stdout').open('wb') as out,(dest/'stderr').open('wb') as err:
        p=subprocess.run(SSH+[command],input=data,stdout=out,stderr=err,timeout=timeout)
    (dest/'exit').write_text(str(p.returncode)+'\n')
    assert p.returncode==0,(label,(dest/'stderr').read_text()[-3000:])
    print('PASS',label,flush=True)
    return dest


def prepare():
    paths=[p for p in (EXP/'src').iterdir() if p.is_file()]+[EXP/'CONTRACT.md',EXP/'DESIGN.md']
    assert all(not p.is_symlink() for p in paths)
    prev=ROOT/'experiments/temporal_tc_20260905_tncn_mode2_gate0'
    dependencies=list((prev/'output/census1').glob('*.npz'))+[prev/'output/census1/census.json']
    dependencies += [prev/'src'/name for name in ['common.py','check_small.py','oracle.py']]
    old=ROOT/'experiments/temporal_tc_20260905_gpu2_notc'
    dependencies += [old/'src/backend_v2.py',old/'src/kernels.cu',old/'output/gpu2/native_gpu.so']
    author=ROOT/'experiments/temporal_tc_20260905_cooccurrence_gate0/sources/TNCN'
    dependencies += [author/'modules/NCNDecoder/NCNPred.py',author/'modules/neighbor_loader.py']
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+dependencies}
    (EXP/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    paths.append(EXP/'source_manifest.json')
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w') as tf:
        for p in paths:tf.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    payload=buf.getvalue();(EXP/'output/source_bundle.tar').write_bytes(payload)
    execute('upload1',f'cd {REMOTE} && test ! -e {REL} && mkdir {REL} && tar -xf -',payload)
    cmd=PREFIX+f'''set -e; test $(df -Pk . | tail -1 | awk '{{print $4}}') -ge 5242880; mkdir -p {REL}/output; mkdir {REL}/output/build1; cd {REL}/output/build1; nvcc --version > toolchain.txt; compute-sanitizer --version >> toolchain.txt; nvcc -std=c++17 -O3 --fmad=false -lineinfo -gencode arch=compute_120a,code=sm_120a -Xcompiler=-fPIC -shared -Xptxas=-v ../../src/builder.cu -o graph_builder.so > build.stdout 2> build.stderr; cuobjdump --dump-resource-usage graph_builder.so > resources.txt; cuobjdump --dump-sass graph_builder.so > sass.txt; sha256sum graph_builder.so > binary.sha256; {PY} ../../src/static.py .'''
    execute('build1_receipt',cmd)
    cmd=f'cd {REMOTE}/{REL}/output && tar -cf - build1'
    dest=EXP/'output/build1'
    assert not dest.exists()
    raw=subprocess.check_output(SSH+[cmd])
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        for item in tf.getmembers():
            assert item.name.startswith('build1/') or item.name=='build1'
            assert '..' not in Path(item.name).parts and not item.issym() and not item.islnk()
        tf.extractall(EXP/'output',filter='data')
    assert json.loads((dest/'static.json').read_text())['all_pass']


def run_phase(label,mode,process=0,tool=None):
    prefix=f'compute-sanitizer --tool {tool} --error-exitcode 99 ' if tool else ''
    dest=execute(label,PREFIX+prefix+f'{PY} -u {REL}/src/run.py --mode {mode} --process {process}')
    # Sanitizers mix their diagnostics into stdout; the runner's JSON records are disjoint.
    lines=dest.joinpath('stdout').read_text().splitlines()
    records=[json.loads(line) for line in lines if line.startswith('{')]
    assert records[-1]['kind']=='closing' and records[-1]['all_pass']
    assert any(r['kind']=='summary' and r['all_pass'] for r in records)
    if tool:
        text=dest.joinpath('stdout').read_text()+dest.joinpath('stderr').read_text()
        assert 'ERROR SUMMARY: 0 errors' in text or (tool=='racecheck' and 'RACECHECK SUMMARY: 0 hazards' in text),text[-1500:]
    (dest/'records.json').write_text(json.dumps(records,indent=2)+'\n')
    (dest/'receipt.json').write_text(json.dumps(dict(all_pass=True,mode=mode,tool=tool,process=process),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','gates','bench'],required=True);args=p.parse_args()
    if args.phase=='prepare':prepare();return
    assert json.loads((EXP/'output/build1/static.json').read_text())['all_pass']
    if args.phase=='gates':
        run_phase('correct1','correct')
        for tool in ['memcheck','synccheck','initcheck','racecheck']:run_phase(tool+'1','sanitizer',tool=tool)
        run_phase('stress1','stress')
    else:
        for name in ['correct1','memcheck1','synccheck1','initcheck1','racecheck1','stress1']:
            assert json.loads((EXP/'output'/name/'receipt.json').read_text())['all_pass']
        for process in range(6):run_phase('bench'+str(process),'bench',process)
        for process in range(4):run_phase('sustained'+str(process),'sustained',process)


if __name__=='__main__':main()
