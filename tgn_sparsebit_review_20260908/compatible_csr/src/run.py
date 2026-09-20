import argparse
import gc
import json
from pathlib import Path
import random
import time
import types
import bridge as b
import audit as oldaudit
p,torch,d = b.p,b.torch,b.d
ROOT=Path(__file__).resolve().parents[1]


def cases():
    for name in ['wikipedia','college']:
        for bs in [32,200]:
            for start in ([128,256,512] if bs==32 else [20,40,80]):
                yield name,bs,start,f'{name}_b{bs}_s{start}'


def prepare(tr,variant,**kwargs):
    if variant=='compatible':
        d.install_compatible(tr)
    elif variant=='keeper':
        if 'get_cn_emb' in tr.model['pred'].__dict__:
            del tr.model['pred'].get_cn_emb
    else:
        b.install(tr,variant,**kwargs)


def qualify(out):
    totals={v:dict(steps=0,bitwise=0,tolerance_pass=0) for v in ['compatible','direct','fused_bounds','keeper']}
    for name,bs,start,label in cases():
        ck=torch.load(d.ART/(label+'_checkpoint.pt'),weights_only=False)
        reference=torch.load(d.ART/(label+'_source_replay_checks.pt'),weights_only=False)
        for variant in totals:
            tr=p.Trainer(name,bs); captured=[]
            prepare(tr,variant,capture=captured,verify=True)
            tr.restore(ck); checks=[]; rows=[]
            for j in range(8):
                result=b.step(tr,start+j,variant,check=True)['check']
                compare=oldaudit.comp(result,reference[j])
                checks.append(dict(index=start+j,**compare)); rows.append(result)
                totals[variant]['steps']+=1
                totals[variant]['bitwise']+=compare['bitwise_equal']
                totals[variant]['tolerance_pass']+=compare['pass_all']
            torch.save(rows,out/(label+'_'+variant+'_checks.pt'))
            if captured:
                assert len(captured)==16
                torch.save(captured,out/(label+'_'+variant+'_csr.pt'))
            p.emit('qualification',case=label,variant=variant,checks=checks,csr_calls=len(captured))
            del tr,rows,captured
        p.ref.guard()
    (out/'qualification.json').write_text(json.dumps(totals,indent=2)+'\n')
    p.emit('qualification_summary',totals=totals)


def install_legacy_profile(tr,clock):
    pred=tr.model['pred'];pred._paid_graph=None
    def get_cn(self,x,edges,q,mode,decay=False,time_info=None):
        assert mode==2 and not decay
        with clock.phase('graph_build'):
            if self._paid_graph is None:
                self._paid_graph=p.builder.graph(len(x),edges,checked=True)
        with clock.phase('integer_producer'):
            plans=[p.ref.from_graph(len(x),q[:,j:j+64],self._paid_graph) for j in range(0,q.shape[1],64)]
        with clock.phase('csr_conversion'):
            coeff=d.dense_plans(plans).float()
        ys=[]
        for c in coeff:
            with clock.phase('csr_conversion'):
                sparse=p.torch_sparse.SparseTensor.from_dense(c)
            with clock.phase('seven_spmm'):
                ys.append(p.predmod.spmm_add(sparse,x))
        return torch.cat(ys,dim=-1)
    pred.get_cn_emb=types.MethodType(get_cn,pred)


def timing(out,qualification):
    gate=json.loads(qualification.read_text())
    eligible=[v for v in ['direct','fused_bounds'] if gate[v]['bitwise']==96]
    assert gate['compatible']['bitwise']==96 and eligible
    variants=['compatible','keeper']+eligible
    rng=random.Random(20260908)
    for name,bs,start,label in cases():
        ck=torch.load(d.ART/(label+'_checkpoint.pt'),weights_only=False)
        tr=p.Trainer(name,bs)
        for rnd in range(9):
            order=variants.copy();rng.shuffle(order)
            for variant in order:
                prepare(tr,variant);tr.restore(ck);gc.collect();torch.cuda.synchronize()
                a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();begin=time.perf_counter_ns()
                for j in range(8): b.step(tr,start+j,variant)
                z.record();torch.cuda.synchronize()
                p.emit('timing',case=label,variant=variant,round=rnd,order=order,steps=8,
                       timeline_ms=a.elapsed_time(z),wall_ms=(time.perf_counter_ns()-begin)/1e6)
        # Separately instrument conversion and real training; never pool with timing.
        expected=torch.load(d.ART/(label+'_source_replay_checks.pt'),weights_only=False)
        for variant in ['compatible']+eligible:
            clock=p.Clock(enabled=True)
            if variant=='compatible':install_legacy_profile(tr,clock)
            else:b.install(tr,variant,clock=clock)
            tr.restore(ck);gc.collect();torch.cuda.synchronize()
            a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record()
            for j in range(8): b.step(tr,start+j,variant)
            z.record();torch.cuda.synchronize()
            p.emit('phases',case=label,variant=variant,steps=8,total_timeline_ms=a.elapsed_time(z),phases=clock.result())
        del tr,ck,expected
        gc.collect();p.ref.guard()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--stage',choices=['qualify','timing'],default='qualify')
    ap.add_argument('--qualification',type=Path)
    args=ap.parse_args();args.output.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p.ref.guard(initial=True);torch.zeros(1,device='cuda');p.ref.guard()
    p.emit('opening',stage=args.stage,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
           gpu_uuid=__import__('os').environ['CUDA_VISIBLE_DEVICES'],
           bridge_sha256=p.sha(ROOT/'src/bridge.py'),runner_sha256=p.sha(__file__),
           prior_runner_sha256=p.sha(d.OLD/'src/profile_training.py'))
    if args.stage=='qualify':qualify(args.output)
    else:timing(args.output,args.qualification)
    p.emit('complete',apps=p.ref.guard())


if __name__=='__main__':main()
