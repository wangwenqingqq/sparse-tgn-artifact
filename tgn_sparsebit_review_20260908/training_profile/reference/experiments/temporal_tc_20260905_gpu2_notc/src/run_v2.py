"""Native GPU gates and balanced complete-decoder ordinary-control screen."""
import argparse,hashlib,importlib.util,json,os,platform,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from backend_v2 import ROOT,LIB,Plan,Consume,decode,pair,validate
REF=ROOT.parent/'temporal_tc_20260905_native_notc_control';PREV=ROOT.parent/'temporal_tc_20260905_tncn_mode2_gate0'
sys.path.insert(0,str(REF/'src'))
import native as cpu
spec=importlib.util.spec_from_file_location('cpu_checks',REF/'src/check.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
sys.path.insert(0,str(PREV/'src'))
from common import source,SRC,sha
from check_small import snap
VARIANTS=['L-separate','B-separate','L-shared','B-shared','L-shared-library']
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def coeff(n,e,q):return cpu.Plan(n,e,q,'list').dense()
def dense_decode(m,x,q,c,decay,info):
    cc=torch.tensor(c.reshape(7*q.shape[1],len(x)),dtype=x.dtype,device=x.device)
    if decay:
        factors=torch.exp(-(info[1][:,None]-info[0][None,:])/10000)
        cc=cc.reshape(7,q.shape[1],len(x))*torch.cat([factors[None].expand(6,-1,-1),torch.ones_like(factors)[None]],0)
        cc=cc.reshape(7*q.shape[1],len(x))
    y=(cc@x).reshape(7,q.shape[1],x.shape[1]).permute(1,0,2).reshape(q.shape[1],-1)
    return m.xsmlp(torch.cat([x[q[0]]*x[q[1]],y],-1))
def compare(a,b,dtype):return old.compare(a,b,dtype)
def gpu_apps():
    txt=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True)
    return [r for r in txt.splitlines() if os.environ['CUDA_VISIBLE_DEVICES'] in r]
def gpu_guard():
    apps=gpu_apps()
    if len(apps)!=1:raise RuntimeError('GPU isolation failed: '+repr(apps))
    return apps
def structure(n,e,q,c,mode):
    p=Plan(n,e,q,mode);actual=p.dense();assert torch.equal(actual.cpu(),torch.tensor(c)),mode
    rp,col,tp,tr,perm,row=[t.cpu().numpy() for t in [p.rp,p.col,p.tp,p.tr,p.perm,p.row]]
    assert np.array_equal(np.sort(perm),np.arange(p.nnz))
    assert np.array_equal(tr,row[perm]);assert tp[-1]==rp[-1]==p.nnz
    for z in range(n):
        ids=perm[tp[z]:tp[z+1]];assert np.all(col[ids]==z)
        rr=tr[tp[z]:tp[z+1]];assert np.all(rr[1:]>rr[:-1])
    return dict(mode=mode,all7_exact=True,transpose_exact=True,nnz=p.nnz,coefficient_sha256=hashlib.sha256(c.tobytes()).hexdigest())
def decoder_checks(ns,n,e,q,c,x,up,ti,seed,width=7):
    checks=[]
    for dtype in [torch.float64,torch.float32]:
        torch.set_default_dtype(dtype)
        for decay in [False,True]:
            info=tuple(torch.tensor(a,dtype=dtype,device='cuda') for a in ti);snaps=[]
            for kind in ['oracle','list','bitset']:
                torch.manual_seed(seed);m=ns['NCNPredictor'](width,11 if width==7 else 256,3 if width==7 else 1,2).cuda()
                xx=torch.tensor(x,dtype=dtype,device='cuda',requires_grad=True);opt=torch.optim.Adam(m.parameters(),lr=1e-4)
                y=dense_decode(m,xx,q,c,decay,info) if kind=='oracle' else decode(m,xx,e,q,kind,decay,info)
                (y*torch.tensor(up,dtype=dtype,device='cuda')).sum().backward();opt.step();snaps.append(snap(m,xx,y,opt))
            for kind,s in zip(['list','bitset'],snaps[1:]):checks.append(dict(mode=kind,width=width,dtype=str(dtype),decay=decay,**compare(snaps[0],s,dtype)))
    return checks
def source_interpreter_checks(ns):
    rows=[]
    for name,seed,n,e,q,x,up,ti,state,before in old.cases(False):
        if seed>=8:break
        torch.set_default_dtype(torch.float64);ed=torch.tensor(e,device='cuda');qu=torch.tensor(q,device='cuda');c=coeff(n,e,q)
        for decay in [False,True]:
            torch.manual_seed(seed);m=ns['NCNPredictor'](7,11,3,2).cuda();xx=torch.tensor(x,device='cuda');info=tuple(torch.tensor(z,device='cuda') for z in ti)
            a=m(xx,ed,qu,2,decay,info);b=dense_decode(m,xx,qu,c,decay,info)
            torch.testing.assert_close(a,b,atol=1e-9,rtol=1e-8)
            rows.append(dict(case=name,decay=decay,max_abs=float((a-b).abs().max())))
    return rows
def safety():
    e=torch.tensor([[0,1],[1,2]],device='cuda');q=torch.tensor([[0],[2]],device='cuda');p=Plan(3,e,q)
    x=torch.randn(3,14,device='cuda',dtype=torch.float64)[:,::2];v=p.values(x.dtype)
    torch.testing.assert_close(p.mm(v,x),p.mm(v,x.contiguous()),atol=0,rtol=0)
    rejected=0
    for n,ed,qu in [(2049,e,q),(3,e.float(),q),(3,e.cpu(),q),(3,e,torch.tensor([[3],[0]],device='cuda'))]:
        try:Plan(n,ed,qu)
        except (TypeError,ValueError):rejected+=1
        else:raise AssertionError('invalid inputs accepted')
    return dict(noncontiguous_features=True,invalid_cases_rejected=rejected,graph='unsupported dynamic nnz sync')
def run_correct(out):
    ns=source('interpreter');dump(out/'source_interpreter_checks.json',source_interpreter_checks(ns));rows=[]
    for name,seed,n,e,q,x,up,ti,state,before in old.cases(True):
        c=coeff(n,e,q)
        if state is not None:assert hashlib.sha256(c.tobytes()).hexdigest()==state['coefficient_sha256']
        ed=torch.tensor(e,device='cuda');qu=torch.tensor(q,device='cuda')
        r=dict(case=name,structures=[structure(n,ed,qu,c,m) for m in ['list','bitset']],checks=decoder_checks(ns,n,ed,qu,c,x,up,ti,seed))
        if before:
            rng=np.random.RandomState(seed+100);r['width100']=decoder_checks(ns,n,ed,qu,c,rng.randn(n,100)*.1,rng.randn(qu.shape[1],1)*.1,ti,seed,100)
        rows.append(r);dump(out/'checks.json',rows)
        if len(rows)%16==0 or before:print('PASS',name,flush=True)
    allc=[v for r in rows for v in r['checks']+r.get('width100',[])]
    summary=dict(cases=len(rows),coefficient_checks=2*len(rows),decoder_comparisons=len(allc),max_abs_by_dtype={str(d):max(r['max_abs'] for r in allc if r['dtype']==str(d)) for d in [torch.float32,torch.float64]},safety=safety(),all_pass=True)
    dump(out/'summary.json',summary);print(json.dumps(summary),flush=True)
def native_probe(out,stress=False):
    rows=[];count=200 if stress else 24;streams=[torch.cuda.Stream(),torch.cuda.Stream()]
    for k in range(count):
        n=([31,32,33,63,64,65,127,128,129,257,1059,2048][k%12]);rng=np.random.RandomState(20260905+k)
        e=rng.randint(0,n,(2,3*n)).astype(np.int64) if k%12 else np.empty((2,0),np.int64)
        q=rng.randint(0,n,(2,1+k%12)).astype(np.int64);q[:,0]=n-1;c=coeff(n,e,q)
        with torch.cuda.stream(streams[k%2]):
            ed=torch.tensor(e,device='cuda');qu=torch.tensor(q,device='cuda');p=Plan(n,ed,qu,'list' if (k+k//12)%2==0 else 'bitset')
            assert torch.equal(p.dense().cpu(),torch.tensor(c))
            dtype=torch.float64 if k%3==0 else torch.float32;x=torch.tensor(rng.randn(n,33)*.1,dtype=dtype,device='cuda',requires_grad=True)
            v=p.values(dtype);a=Consume.apply(x,p,v);up=torch.tensor(rng.randn(p.m,33)*.1,dtype=dtype,device='cuda');(a*up).sum().backward()
            cc=torch.tensor(c.reshape(p.m,n),dtype=dtype,device='cuda');tol=1e-9 if dtype==torch.float64 else 2e-4
            torch.testing.assert_close(a,cc@x.detach(),atol=tol,rtol=tol);torch.testing.assert_close(x.grad,cc.T@up,atol=tol,rtol=tol)
        streams[k%2].synchronize();rows.append(dict(case=k,n=n,pass_all=True))
    dump(out/'checks.json',rows);dump(out/'summary.json',dict(cases=count,all_pass=True,alternate_streams=True,pointer_churn=True));print('PASS',count,flush=True)
def setup(ns,n,seed):
    torch.set_default_dtype(torch.float32);torch.manual_seed(seed);m=ns['NCNPredictor'](100,256,1,2).cuda()
    rng=np.random.RandomState(seed);x=torch.tensor(rng.randn(n,100)*.1,dtype=torch.float32,device='cuda',requires_grad=True)
    return m,x,torch.optim.Adam(m.parameters(),lr=1e-4)
def step(m,x,opt,e,q,variant,c=None):
    opt.zero_grad(set_to_none=True);x.grad=None
    if variant=='oracle':
        pos=dense_decode(m,x,q[:,:32],c[:,:32],False,None);neg=dense_decode(m,x,q[:,32:],c[:,32:],False,None)
    else:pos,neg=pair(m,x,e,q,variant)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(pos,torch.ones_like(pos))+torch.nn.functional.binary_cross_entropy_with_logits(neg,torch.zeros_like(neg))
    loss.backward();opt.step();return pos,neg,loss
def real_states():
    for case in old.cases(True):
        if case[-1]:yield case
def pairs_or_bench(out,process,check_only=False):
    ns=source('interpreter');rows=[];checks=[];order=VARIANTS[process:]+VARIANTS[:process]
    for name,seed,n,e,q,x,up,ti,state,before in real_states():
        gpu_guard();ed=torch.tensor(e,device='cuda');qu=torch.tensor(q,device='cuda');validate(n,ed,qu)
        if check_only:
            c=coeff(n,e,q);snaps=[]
            for kind in ['oracle']+VARIANTS:
                m,xx,opt=setup(ns,n,20260905+seed);pos,neg,loss=step(m,xx,opt,ed,qu,kind,c)
                s=snap(m,xx,torch.cat([pos,neg]),opt);s['loss']=loss.detach().cpu();snaps.append(s)
            checks.append(dict(case=name,checks=[dict(variant=v,**compare(snaps[0],s,torch.float32)) for v,s in zip(VARIANTS,snaps[1:])]))
            dump(out/'checks.json',checks);print('PAIR',name,flush=True);continue
        for kind in order:
            m,xx,opt=setup(ns,n,20260905+seed);torch.cuda.synchronize();start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True)
            for k in range(10):
                torch.cuda.synchronize();t=time.perf_counter_ns();start_event.record()
                step(m,xx,opt,ed,qu,kind);end_event.record();torch.cuda.synchronize();us=(time.perf_counter_ns()-t)/1000
                rows.append(dict(case=name,process=process,order=order,variant=kind,observation=k-3,warmup=k<3,wall_us=us,event_us=start_event.elapsed_time(end_event)*1000,eligible=True))
            dump(out/'samples.json',rows)
        gpu_guard();print('BENCH',name,flush=True)
    if check_only:summary=dict(comparisons=sum(len(r['checks']) for r in checks),max_abs=max(c['max_abs'] for r in checks for c in r['checks']),all_pass=True)
    else:summary=dict(retained=sum(not r['warmup'] for r in rows),warmups=sum(r['warmup'] for r in rows),all_complete=True,order=order)
    dump(out/'summary.json',summary);print(json.dumps(summary),flush=True)
def main(args):
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.zeros(1,device='cuda');torch.cuda.synchronize();apps=gpu_guard()
    dump(out/'opening.json',dict(pid=os.getpid(),apps=apps,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability()))
    if args.mode=='correct':run_correct(out)
    elif args.mode in ['sanitizer','stress']:native_probe(out,args.mode=='stress')
    else:pairs_or_bench(out,args.process,args.mode=='pair-check')
    gpu_guard()
    dump(out/'provenance.json',dict(torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],binary_sha256=sha(LIB),reference_binary_sha256=sha(cpu.LIB),
         contract_sha256=sha(ROOT/'CONTRACT.md'),amendment_sha256=sha(ROOT/'COMPARATOR_REPAIR.md'),files={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src').glob('*') if p.is_file()},
         source_hashes={str(p.relative_to(SRC)):sha(p) for p in [SRC/'modules/NCNDecoder/NCNPred.py',SRC/'modules/neighbor_loader.py']},
         reference_files={str(p.relative_to(REF)):sha(p) for p in [REF/'src/native.cpp',REF/'src/native.py',REF/'src/check.py']},
         fixtures={p.name:sha(p) for p in (PREV/'output/census1').glob('*.npz')}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['correct','sanitizer','stress','pair-check','bench'],required=True);p.add_argument('--process',type=int,default=0);p.add_argument('--output',required=True);main(p.parse_args())
