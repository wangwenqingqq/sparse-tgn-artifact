"""Frozen explicit-algorithm replay, retaining every native observation."""
import hashlib,json,os,platform,sys
from pathlib import Path
ROOT=Path.cwd();EXP=Path(__file__).resolve().parents[1]
for name,h in json.loads((EXP/'source_manifest.json').read_text()).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
sys.path.insert(0,str(ROOT/'experiments/temporal_tc_20260907_cfree_gpu_screen/src'))
import run as r
import backend as cf
import torch,numpy as np
from explicit import identity,transpose,run
ref=cf.ref

def init(label):
    ref.guard(True);torch.set_num_threads(1);torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.zeros(1,device='cuda');torch.cuda.synchronize()
    r.emit('opening',pid=os.getpid(),apps=ref.guard(),label=label,torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__,python=platform.python_version(),library=identity())
    out=EXP/'output'/label;out.mkdir(exist_ok=False);return out

def snapshot(p,tr):
    return {prefix+k:t.cpu().numpy().copy() for prefix,ts in [('',(p.rp,p.col,p.iv)),('T_',tr)] for k,t in zip(['rp','col','iv'],ts)}

def main(process):
    out=init(f'replay_payload{process}');case,st,n,raw,qr=list(r.allstates())[24]
    e=r.upload(raw);q=r.upload(qr);rng=np.random.RandomState(720260907+1024)
    x=torch.tensor(rng.randn(n,100)*.1,dtype=torch.float32,device='cuda');target=torch.tensor(rng.randn(64,700)*.1,dtype=torch.float32,device='cuda')
    fixed=ref.from_graph(n,q,cf.builder.graph(n,e,checked=True));tr=transpose(fixed);cc=ref.dense(fixed)
    assert hashlib.sha256(cc.tobytes()).hexdigest()==st['coefficient_sha256']
    assert np.array_equal(cc,r.coefficients(raw,qr,n))
    arrays=dict(X=x.cpu().numpy(),target=target.cpu().numpy(),C=cc,edges=raw,targets=qr,**snapshot(fixed,tr));rows=[]
    algorithms=[6,12] if process==0 else [12,6];modes=['cached','fresh'] if process==0 else ['fresh','cached']
    for alg in algorithms:
        for mode in modes:
            for j in range(16):
                p=fixed if mode=='cached' else ref.from_graph(n,q,cf.builder.graph(n,e,checked=True))
                tt=tr if mode=='cached' else transpose(p)
                for t,u in zip((p.rp,p.col,p.iv,*tt),(fixed.rp,fixed.col,fixed.iv,*tr)):assert torch.equal(t,u)
                values,cost=run(p,tt,x.clone(),target,alg);key=f'a{alg}_{mode}_{j}'
                for name,t in values.items():arrays[key+'_'+name]=t.detach().cpu().numpy().copy()
                row=dict(key=key,algorithm=alg,mode=mode,iteration=j,process=process,cost=cost);rows.append(row);r.emit('sample',**row);ref.guard()
    # The original K and unchanged G/E are diagnostics only, not new pass criteria.
    for v in ['K','G','E']:
        xx=x.clone().requires_grad_();y=cf.consume(xx,e,q,v);g=2*(y.detach()-target);(y*g).sum().backward();torch.cuda.synchronize()
        for name,t in dict(Y=y,g=g,dx=xx.grad).items():arrays[v+'_'+name]=t.detach().cpu().numpy().copy()
    np.savez_compressed(out/'arrays.npz',**arrays)
    summary=dict(process=process,algorithms=algorithms,modes=modes,rows=rows,arrays=len(arrays),array_sha256=ref.sha(out/'arrays.npz'),fixture=st['fixture'],C_sha256=st['coefficient_sha256'],library=identity(),diagnostic_only=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');r.emit('summary',**summary);r.emit('closing',pid=os.getpid(),apps=ref.guard(),all_pass=True)
if __name__=='__main__':main(int(sys.argv[1]))
