import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PREV=Path('/home/data/wangxuran/flash_tgn_rebaseline_20260909')
sys.path.insert(0,str(PREV/'src'))
import core as c
# Only the factory's train-only output directory changes, to this experiment.
c.ROOT=ROOT
p=c.torch
import flash_tgn.layers as lm
import flash_tgn.layers2 as l2


def cases():return [x for x in c.cases() if x['k']==50]


def check_inputs():
    frozen=c.json.loads((PREV/'input_manifest.json').read_text())
    for path,info in frozen['files'].items():assert c.sha(path)==info['sha256'],path
    return len(frozen['files'])


def opening(stage,files):
    count=check_inputs();p.set_num_threads(1);p.set_float32_matmul_precision('high');c.guard();p.zeros(1,device='cuda');c.guard()
    c.emit('opening',stage=stage,input_files=count,gpu=p.cuda.get_device_name(),gpu_uuid=c.os.environ['CUDA_VISIBLE_DEVICES'],torch=p.__version__,cuda=p.version.cuda,precision=p.get_float32_matmul_precision(),tf32=p.backends.cuda.matmul.allow_tf32,
        hashes={str(x):c.sha(x) for x in [ROOT/'CONTRACT.md',Path(__file__),PREV/'src/core.py',PREV/'input_manifest.json',*files]})


def cpu(value):
    if p.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,(tuple,list)):return type(value)(cpu(x) for x in value)
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    return value


def gpu(value):
    if p.is_tensor(value):return value.cuda()
    if isinstance(value,(tuple,list)):return type(value)(gpu(x) for x in value)
    if isinstance(value,dict):return {k:gpu(v) for k,v in value.items()}
    return value
