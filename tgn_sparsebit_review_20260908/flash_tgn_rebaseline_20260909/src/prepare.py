"""CPU-only source/data/dependency freeze before any GPU campaign."""
import datetime
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/home/data/wangxuran/factor_tgn_sptc_20260902/vendor/flash-tgn-artifact')
DATA=Path('/home/data/wangxuran/factor_tgn_sptc_20260902/project/experiments/factor_tgn_20260904_anchored_time_quality_wikipedia_full/data_root/data/wiki')
sys.path.insert(0,str(ROOT/'deps'))
import numpy as np
import torch
import sklearn


def sha(file):
    h=hashlib.sha256()
    with file.open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()


assert not torch.cuda.is_initialized()
assert sklearn.__version__=='1.7.2' and str(ROOT/'deps') in sklearn.__file__
manifest=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),files={},
    baseline=str(BASE),git_metadata_present=(BASE/'.git').exists(),torch=torch.__version__,sklearn=sklearn.__version__)
def add(path,role,expected=None):
    actual=sha(path)
    if expected is not None:assert actual==expected,str(path)
    manifest['files'][str(path)]=dict(role=role,bytes=path.stat().st_size,sha256=actual)
for file in sorted((BASE/'python/flash_tgn').iterdir()):
    if file.is_file() and file.suffix in ['.py','.so']:add(file,'baseline')
for file in sorted((BASE/'csrc_flash_tgn').iterdir()):
    if file.is_file():add(file,'baseline')
for name in ['train_flash_tgn.py','setup.py','README.md','docs/flash_tgn_clean_code.md']:add(BASE/name,'baseline')
historical=json.loads((DATA/'full_quality_manifest.json').read_text())
for name,info in historical['structural'].items():add(DATA/name,'data',info['sha256'])
for name in ['edge_features','node_features','raw_csv']:
    info=historical[name];add(Path(info['path']),'data',info['sha256'])
add(DATA/'full_quality_manifest.json','data_manifest')
for file in sorted((ROOT/'deps').rglob('*')):
    if file.is_file() and (file.suffix=='.py' or file.name=='METADATA' or file.suffix=='.so'):
        add(file,'dependency')
src,dst,ts=[np.load(DATA/name) for name in ['src.npy','dst.npy','ts.npy']]
ind,nbr,eid,ets=[np.load(DATA/('edges.undirected_tcsr.'+name+'.npy')) for name in ['ind','nbr','eid','ets']]
assert len(src)==len(dst)==len(ts)==157474 and np.all(ts[1:]>=ts[:-1])
assert len(ind)==9228 and ind[0]==0 and ind[-1]==len(nbr)==len(eid)==len(ets)==2*len(src)
assert np.all(np.diff(ind)>=0) and np.all((eid>=0)&(eid<len(src)))
owners=np.repeat(np.arange(len(ind)-1),np.diff(ind))
assert np.all(((owners==src[eid])&(nbr==dst[eid]))|((owners==dst[eid])&(nbr==src[eid])))
assert np.array_equal(ets,ts[eid]) and np.all(np.bincount(eid,minlength=len(src))==2)
assert all(np.all(ets[a+1:b]>=ets[a:b-1]) for a,b in zip(ind[:-1],ind[1:]) if b-a>1)
edge=torch.load(DATA/'edge_features.pt',map_location='cpu',weights_only=True)
node=torch.load(DATA/'node_features.pt',map_location='cpu',weights_only=True)
assert edge.shape==(157474,172) and edge.dtype==torch.float32 and torch.isfinite(edge).all()
assert node.shape==(9227,256) and node.dtype==torch.float32 and torch.count_nonzero(node)==0
manifest['data_check']=dict(events=len(src),nodes=9227,edge_features=list(edge.shape),node_features=list(node.shape),
    chronological=True,tcsr_endpoints=True,tcsr_times=True,two_directed_entries_per_event=True)
assert not torch.cuda.is_initialized()
(ROOT/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(files=len(manifest['files']),bytes=sum(v['bytes'] for v in manifest['files'].values()),data_check=manifest['data_check'])))
