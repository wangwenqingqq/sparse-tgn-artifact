"""Predeclared R3 boundary expansion, stopping at the first failed case."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path.cwd();EXP=Path(__file__).resolve().parents[1]
for name,h in json.loads((EXP/'expansion_manifest.json').read_text()).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
from replay_v2 import r,cf,ref,init,snapshot
from explicit_v2 import transpose,run
import torch,numpy as np
sys.path.insert(0,str(EXP/'analysis'))
from verify_replay import exact_c,fromcsr

def metric(a,b,dtype):
    aa=a.astype(np.float64);bb=b.astype(np.float64);assert np.isfinite(aa).all() and np.isfinite(bb).all()
    at,rt=(1e-9,1e-8) if dtype==torch.float64 else (2e-4,2e-4)
    er=np.abs(aa-bb);sc=er/(at+rt*np.abs(bb));ix=np.unravel_index(sc.argmax(),sc.shape)
    return dict(failed=int((sc>1).sum()),max_scaled=float(sc[ix]),max_abs=float(er.max()),worst=list(map(int,ix)),a=float(aa[ix]),b=float(bb[ix]))

def check(out,name,n,raw,qr,seed,d,dtype):
    rng=np.random.RandomState(720260907+seed);e=r.upload(raw);q=r.upload(qr)
    x=torch.tensor(rng.randn(n,d)*.1,dtype=dtype,device='cuda');target=torch.tensor(rng.randn(q.shape[1],7*d)*.1,dtype=dtype,device='cuda')
    p=ref.from_graph(n,q,cf.builder.graph(n,e,checked=True));tt=transpose(p);C=ref.dense(p)
    assert np.array_equal(C,exact_c(raw,qr,n))
    arrays=dict(C=C,X=x.cpu().numpy(),target=target.cpu().numpy(),edges=raw,targets=qr,**snapshot(p,tt));costs=[]
    for j in range(4):
        # Rebuild every second trial to test exact structural and allocation replay.
        if j%2:
            pp=ref.from_graph(n,q,cf.builder.graph(n,e,checked=True));tr=transpose(pp)
        else:pp,tr=p,tt
        for a,b in zip((p.rp,p.col,p.iv,*tt),(pp.rp,pp.col,pp.iv,*tr)):assert torch.equal(a,b)
        values,cost=run(pp,tr,x.clone(),target,12);costs.append(cost)
        for field,t in values.items():arrays[f'r{j}_{field}']=t.detach().cpu().numpy().copy()
    path=out/(name+'.npz');np.savez_compressed(path,**arrays)
    cm=C.reshape(7*q.shape[1],n)
    assert np.array_equal(cm,fromcsr(arrays['rp'],arrays['col'],arrays['iv'],len(cm),n))
    assert np.array_equal(cm.T,fromcsr(arrays['T_rp'],arrays['T_col'],arrays['T_iv'],n,len(cm)))
    yexact=np.einsum('cbn,nd->bcd',C.astype(np.float64),arrays['X'].astype(np.float64),optimize=False).reshape(q.shape[1],7*d)
    gexact=2*(yexact-arrays['target'].astype(np.float64));dxexact=np.einsum('cbn,bcd->nd',C.astype(np.float64),gexact.reshape(q.shape[1],7,d),optimize=False)
    checks=[];repeat=True
    for j in range(4):
        assert np.array_equal(arrays[f'r{j}_g'],2*(arrays[f'r{j}_Y']-arrays['target']))
        repeat=repeat and all(np.array_equal(arrays[f'r{j}_'+f],arrays['r0_'+f]) for f in ['Y','g','dx'])
        own=np.einsum('cbn,bcd->nd',C.astype(np.float64),arrays[f'r{j}_g'].astype(np.float64).reshape(q.shape[1],7,d),optimize=False)
        for field,expected in [('Y',yexact),('dx_own_upstream',own),('dx_composed',dxexact)]:
            checks.append(dict(iteration=j,field=field,**metric(arrays[f'r{j}_'+('Y' if field=='Y' else 'dx')],expected,dtype)))
    good=repeat and all(c['failed']==0 for c in checks if c['field']!='dx_composed')
    row=dict(name=name,n=n,E=raw.shape[1],B=q.shape[1],D=d,dtype=str(dtype),source_seed=seed,arrays=len(arrays),C_cells=C.size,C_sha256=hashlib.sha256(C.tobytes()).hexdigest(),array_sha256=ref.sha(path),bitwise_repeat=repeat,checks=checks,costs=costs,all_pass=good)
    (out/(name+'.json')).write_text(json.dumps(row,indent=2)+'\n');r.emit('case',**row);ref.guard()
    assert good,('R3_EXPANSION_GATE',name,repeat,[c for c in checks if c['failed'] and c['field']!='dx_composed'])
    return row

def main(mode):
    out=init('expansion_payload1' if mode=='correct' else f'{mode}_payload1');rows=[]
    if mode=='correct':
        for k in range(128):
            n,e,q=r.generated(k);dtype=torch.float64 if k%2 else torch.float32
            rows.append(check(out,f'generated_{k}',n,e,q,k,[7,33,100,128][(k//2)%4],dtype))
        for i,(case,st,n,e,q) in enumerate(r.allstates()):
            for dtype in [torch.float32,torch.float64]:rows.append(check(out,f'real_{i}_'+('f64' if dtype==torch.float64 else 'f32'),n,e,q,1000+i,100,dtype))
    else:
        for j,k in enumerate([0,1,2,8,9,14,15,16,17,18,20,21]):
            n,e,q=r.generated(k);rows.append(check(out,f'probe_{j}',n,e,q,4000+j,[7,33,100,128][j%4],torch.float32 if j%2==0 else torch.float64))
    summary=dict(mode=mode,cases=len(rows),native_replays=4*len(rows),all_pass=True,composed_oracle_failing_cases=sum(any(c['failed'] for c in row['checks'] if c['field']=='dx_composed') for row in rows))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');r.emit('summary',**summary);r.emit('closing',pid=__import__('os').getpid(),apps=ref.guard(),all_pass=True)
if __name__=='__main__':main(sys.argv[1])
