"""Create-only transport and GPU3 serialization."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
EXP=Path(__file__).resolve().parents[1];ROOT=EXP.parent.parent;REL=str(EXP.relative_to(ROOT))
REMOTE='/home/data/wangxuran/factor_tgn_sptc_20260902/project';PY='/home/data/wangxuran/isaacsim6/env/bin/python'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','pro6000-8-wxr']
PREFIX=f'''cd {REMOTE} && exec 9</home/data/wangxuran/.locks/tgn_sptc_gpu3.lock && flock -n -x 9 && export PATH=/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=GPU-a149f5af-55ab-ce33-8d3d-371a7ae61dd2 PYTHONDONTWRITEBYTECODE=1 CUDA_CACHE_DISABLE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 GPU_NT_LIB=$PWD/experiments/temporal_tc_20260905_gpu2_notc/output/gpu2/native_gpu.so GB_LIB=$PWD/experiments/temporal_tc_20260905_graph_builder/output/build1/graph_builder.so CF_LIB=$PWD/experiments/temporal_tc_20260907_cfree_gpu_screen/output/build1/cfree.so TMPDIR=$PWD/{REL}/tmp && '''
def execute(label,cmd,data=None,timeout=1800):
    dest=EXP/'output'/label;dest.mkdir(exist_ok=False);(dest/'command.json').write_text(json.dumps(dict(command=cmd,input_sha256=hashlib.sha256(data).hexdigest() if data else None),indent=2)+'\n');print('START',label,flush=True)
    with (dest/'stdout').open('wb') as out,(dest/'stderr').open('wb') as err:
        p=subprocess.run(SSH+[cmd],input=data,stdout=out,stderr=err,timeout=timeout)
    (dest/'exit').write_text(str(p.returncode)+'\n');assert p.returncode==0,(label,(dest/'stderr').read_text()[-3000:]);print('PASS',label,flush=True);return dest

def bundle(paths):
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w') as tf:
        for p in paths:tf.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    return buf.getvalue()
def fetch(name):
    assert not (EXP/'output'/name).exists()
    raw=subprocess.check_output(SSH+[f'cd {REMOTE}/{REL}/output && tar -cf - {name}']);(EXP/'output'/(name+'_download.tar')).write_bytes(raw)
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        for item in tf.getmembers():
            pp=Path(item.name);assert pp.parts[0]==name and not pp.is_absolute() and '..' not in pp.parts
            dest=EXP/'output'/pp
            if item.isdir():dest.mkdir(parents=True,exist_ok=True)
            else:
                assert item.isfile() and not item.issym() and not item.islnk()
                with dest.open('xb') as f:f.write(tf.extractfile(item).read())
