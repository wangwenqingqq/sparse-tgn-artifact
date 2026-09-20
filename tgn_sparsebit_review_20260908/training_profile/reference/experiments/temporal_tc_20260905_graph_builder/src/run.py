"""Frozen builder checks, full-decoder A/B, and sustained fixed-state replay."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
ROOT = Path.cwd()
EXP = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((EXP/'source_manifest.json').read_text())
for name, h in MANIFEST.items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == h, name
import numpy as np
import torch
import builder
import reference as ref
from check_small import fixtures

SIZES = [1,2,3,7,31,32,33,63,64,65,127,128,129,255,256,257,1023,1024,1025,1788,2047,2048]


def emit(kind, **data):
    print(json.dumps(dict(kind=kind, **data)), flush=True)


def b1(n, e, q):
    return ref.from_graph(n, q, builder.graph(n, e, checked=True))


def generated(k):
    n = SIZES[k % len(SIZES)]
    rng = np.random.RandomState(20260905+k)
    m = [0, 2*n, 8*n, min(n*n,16*n)][k%4]
    e = rng.randint(0,n,(2,m)).astype(np.int64)
    if k % 7 == 0 and m:
        e[:, :m//2] = 0
    return n, e


def structure(n, raw, noncontig=False):
    e = torch.tensor(raw.T, device='cuda').T if noncontig else torch.tensor(raw,device='cuda')
    gg = builder.graph(n,e)
    old = ref.graph(n,e)
    a = np.zeros((n,n),np.int64)
    np.add.at(a,(raw[0],raw[1]),1)
    np.add.at(a,(raw[1],raw[0]),1)
    rr, cc = np.nonzero(a)
    rp = np.concatenate(([0],np.cumsum(np.bincount(rr,minlength=n)))).astype(np.int64)
    nnz = len(cc)
    pad = np.zeros((n,((n+31)//32)*32),np.uint8)
    pad[:,:n] = a != 0
    bits = np.packbits(pad,axis=1,bitorder='little').view(np.int32)
    for graph in [gg,old]:
        pp, ci, av, pb = graph
        assert np.array_equal(pp.cpu().numpy(),rp)
        assert np.array_equal(ci[:nnz].cpu().numpy(),cc)
        assert np.array_equal(av[:nnz].cpu().numpy(),a[rr,cc])
        assert np.array_equal(pb.cpu().numpy(),bits)
    return e, gg, dict(n=n,E=raw.shape[1],nnz=nnz,max_value=int(a.max()),
        A_sha256=hashlib.sha256(a.tobytes()).hexdigest(),noncontiguous=noncontig,
        scratch_bytes=4*n*((n+31)//32)*32,
        capacity=min(2*raw.shape[1],n*n),all_pass=True)


def probe_one(k, n=None, raw=None):
    if n is None:
        n,raw = generated(k)
    e,gg,result = structure(n,raw,k%2==1)
    rng = np.random.RandomState(2060905+k)
    q = torch.tensor(rng.randint(0,n,(2,3)).astype(np.int64),device='cuda')
    plans = [ref.fresh(n,e,q),ref.from_graph(n,q,gg)]
    for field in ['rp','col','iv']:
        assert torch.equal(getattr(plans[0],field),getattr(plans[1],field))
    x0 = torch.tensor(rng.randn(n,7)*.1,dtype=torch.float32,device='cuda')
    up = torch.tensor(rng.randn(3,49)*.1,dtype=torch.float32,device='cuda')
    values=[]
    for p in plans:
        x=x0.clone().requires_grad_();y=p.consume(x,library=True)
        (y*up).sum().backward();values.append(dict(y=y.detach(),dx=x.grad.detach()))
    result['consumer_check']=ref.compare(*values)
    return result


def graph_replays():
    n,m=65,260
    e=torch.zeros((2,m),dtype=torch.int64,device='cuda')
    work=builder.Workspace(n,m,e.device)
    side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):work.run(e)
    side.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=side):work.run(e)
    for k in range(40):
        rng=np.random.RandomState(920260905+k)
        raw=rng.randint(0,n,(2,m)).astype(np.int64)
        if k%3==0:raw[:,:100]=0
        e.copy_(torch.tensor(raw,device='cuda'))
        g.replay();torch.cuda.synchronize()
        target=ref.graph(n,e);nnz=int(target[0][-1].item())
        for a,b in zip([work.rp,work.col[:nnz],work.values[:nnz],work.bits],target):
            assert torch.equal(a,b)
        emit('graph_replay',index=k,nnz=nnz,all_pass=True)


def correct(ns):
    for k in range(128):
        _,_,result=structure(*generated(k),noncontig=k%2==1)
        emit('generated',index=k,**result)
    _,_,bound=structure(1,np.zeros((2,1000000),np.int64))
    assert bound['max_value']==2000000
    emit('count_bound',**bound)
    states=json.loads((ref.PREV/'output/census1/census.json').read_text())
    cchecks=pairchecks=0
    for i,case in enumerate(states):
        for state in case['states']:
            with np.load(ref.PREV/'output/census1'/state['fixture']) as z:
                n=len(z['ids']);raw=z['edges'].copy();q=torch.tensor(z['targets'],device='cuda')
            e,gg,inventory=structure(n,raw)
            plans=[ref.fresh(n,e,q),ref.from_graph(n,q,gg)]
            for p in plans:
                cc=ref.dense(p)
                assert hashlib.sha256(cc.tobytes()).hexdigest()==state['coefficient_sha256']
                cchecks+=1
            checks=[]
            if state['state']=='before':
                snaps=[]
                for factory in [lambda:ref.fresh(n,e,q),lambda:b1(n,e,q)]:
                    m,x,opt=ref.setup(ns,n,20260905+i)
                    snaps.append([ref.step(m,x,opt,q,factory,True) for _ in range(2)])
                checks=[ref.compare(snaps[0][j],snaps[1][j]) for j in range(2)]
                pairchecks+=len(checks)
            emit('real',case=case['case'],state=state['state'],fixture=state['fixture'],
                 coefficient_sha256=state['coefficient_sha256'],checks=checks,**inventory)
            ref.guard()
    small=[]
    for seed,n,e,q,x,up,ti in fixtures():
        if seed>=8:break
        ed=torch.tensor(e,device='cuda');qu=torch.tensor(q,device='cuda')
        for dtype in [torch.float32,torch.float64]:
            torch.set_default_dtype(dtype)
            for decay in [False,True]:
                snapshots=[]
                for factory in [lambda:ref.fresh(n,ed,qu),lambda:b1(n,ed,qu)]:
                    torch.manual_seed(seed);m=ns['NCNPredictor'](7,11,3,2).cuda()
                    xx=torch.tensor(x,dtype=dtype,device='cuda',requires_grad=True)
                    opt=torch.optim.Adam(m.parameters(),lr=1e-4)
                    info=tuple(torch.tensor(t,dtype=dtype,device='cuda') for t in ti)
                    pp=factory();y=pp.consume(xx,decay,info,library=True)
                    out=m.xsmlp(torch.cat([xx[qu[0]]*xx[qu[1]],y],-1))
                    (out*torch.tensor(up,dtype=dtype,device='cuda')).sum().backward();opt.step()
                    snapshots.append(ref.snap(m,xx,out,opt))
                check=ref.compare(*snapshots)
                if dtype==torch.float64:
                    for key in snapshots[0]:
                        if snapshots[0][key] is not None:
                            torch.testing.assert_close(snapshots[0][key],snapshots[1][key],atol=1e-9,rtol=1e-8)
                small.append(dict(seed=seed,dtype=str(dtype),decay=decay,**check))
    torch.set_default_dtype(torch.float32)
    emit('small_decoder',checks=small,all_pass=True)
    e=torch.zeros((2,1),dtype=torch.int64,device='cuda');rejected=0
    for n,ed in [(0,e),(2049,e),(1,e.float()),(1,e.cpu()),(1,e+1),(1,e-1)]:
        try:builder.graph(n,ed)
        except (ValueError,TypeError):rejected+=1
        else:raise AssertionError('invalid accepted')
    assert rejected==6
    emit('summary',mode='correct',generated_graphs=129,real_graphs=36,coefficient_checks=cchecks,
         optimizer_checks=pairchecks,small_decoder_checks=len(small),invalid_rejected=rejected,all_pass=True)


def stress(count, graph=False):
    streams=[torch.cuda.Stream(),torch.cuda.Stream()]
    for k in range(count):
        streams[k%2].wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(streams[k%2]):
            result=probe_one(k)
        streams[k%2].synchronize()
        emit('probe',index=k,**result)
        if k%24==0:ref.guard()
    if graph:graph_replays()
    emit('summary',mode='stress' if graph else 'sanitizer',probes=count,graph_replays=40 if graph else 0,all_pass=True)


def bench(ns,process,sustained=False):
    order=['B0','B1'] if process%2==0 else ['B1','B0']
    if sustained:
        data=list(ref.states())
        for name in order:
            ref.guard();torch.manual_seed(20260905)
            m=ns['NCNPredictor'](100,256,1,2).cuda();opt=torch.optim.Adam(m.parameters(),lr=1e-4)
            xs=[]
            for i,(_,_,n,_,_) in enumerate(data):
                rng=np.random.RandomState(20260905+i)
                xs.append(torch.tensor(rng.randn(n,100)*.1,dtype=torch.float32,device='cuda',requires_grad=True))
            for sweep in range(8):
                torch.cuda.synchronize();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start=time.perf_counter_ns();a.record()
                for x,(_,_,n,e,q) in zip(xs,data):
                    factory=(lambda:ref.fresh(n,e,q)) if name=='B0' else (lambda:b1(n,e,q))
                    ref.step(m,x,opt,q,factory)
                b.record();torch.cuda.synchronize();wall=(time.perf_counter_ns()-start)/1000
                ref.guard();emit('sweep',process=process,order=order,variant=name,observation=sweep-3,
                                warmup=sweep<3,wall_us=wall,event_us=a.elapsed_time(b)*1000,eligible=True)
        emit('summary',mode='sustained',process=process,retained=10,warmups=6,all_pass=True)
        return
    for i,(case,state,n,e,q) in enumerate(ref.states()):
        for name in order:
            ref.guard();m,x,opt=ref.setup(ns,n,20260905+i)
            factory=(lambda:ref.fresh(n,e,q)) if name=='B0' else (lambda:b1(n,e,q))
            torch.cuda.synchronize();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            rows=[]
            for k in range(10):
                torch.cuda.synchronize();start=time.perf_counter_ns();a.record()
                ref.step(m,x,opt,q,factory);b.record();torch.cuda.synchronize()
                rows.append(dict(case=case['case'],dataset=case['dataset'],process=process,order=order,
                    variant=name,observation=k-3,warmup=k<3,wall_us=(time.perf_counter_ns()-start)/1000,
                    event_us=a.elapsed_time(b)*1000,eligible=True))
            try:ref.guard()
            except Exception:
                for r in rows:r['eligible']=False;emit('sample',**r)
                raise
            for r in rows:emit('sample',**r)
    emit('summary',mode='bench',process=process,retained=252,warmups=108,all_pass=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['correct','sanitizer','stress','bench','sustained'],required=True)
    parser.add_argument('--process',type=int,default=0);args=parser.parse_args()
    ref.guard(True);torch.set_num_threads(1);torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.zeros(1,device='cuda');torch.cuda.synchronize()
    emit('opening',mode=args.mode,process=args.process,pid=os.getpid(),apps=ref.guard(),
         torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
         builder_sha256=hashlib.sha256(Path(builder.LIB).read_bytes()).hexdigest(),
         keeper_sha256=ref.sha(ref.LIB),source_manifest_sha256=ref.sha(EXP/'source_manifest.json'),
         hardware=subprocess.check_output(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],
          '--query-gpu=uuid,driver_version,clocks.current.sm,clocks.current.memory,power.limit,pstate,temperature.gpu,memory.used,utilization.gpu','--format=csv'],text=True))
    if args.mode=='correct':correct(ref.source('interpreter'))
    elif args.mode in ['sanitizer','stress']:stress(24 if args.mode=='sanitizer' else 200,args.mode=='stress')
    else:bench(ref.source('interpreter'),args.process,args.mode=='sustained')
    emit('closing',pid=os.getpid(),apps=ref.guard(),all_pass=True)


if __name__=='__main__':main()
