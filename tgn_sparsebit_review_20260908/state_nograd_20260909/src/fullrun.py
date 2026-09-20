import argparse
import gc
import json
from pathlib import Path
import random
import time
import candidate as c
import audit as audit
p,torch,d,i=c.p,c.torch,c.d,c.i
ROOT=Path(__file__).resolve().parents[1]
ART=c.PREV/'output/qualify_v1/artifacts'


def cases():
    for name in ['wikipedia','college']:
        for bs in [32,200]:
            for start in ([128,256,512] if bs==32 else [20,40,80]):
                yield name,bs,start,f'{name}_b{bs}_s{start}'


def qualify(out):
    totals={v:dict(steps=0,bitwise=0,elements=0,source_bitwise=0) for v in c.VARIANTS}
    for name,bs,start,label in cases():
        ck=torch.load(d.ART/(label+'_checkpoint.pt'),weights_only=False)
        source=torch.load(d.ART/(label+'_source_replay_checks.pt'),weights_only=False)
        expected=torch.load(ART/(label+'_direct_checks.pt'),weights_only=False)
        for variant in c.VARIANTS:
            tr=p.Trainer(name,bs);c.prepare(tr,variant);tr.restore(ck);rows=[];checks=[]
            for j in range(8):
                result=i.b.step(tr,start+j,'direct',check=True)['check']
                full,storage=i.expanded_check(tr,result)
                cmp=audit.comp(full,expected[j]);original=audit.comp(result,source[j])
                checks.append(dict(index=start+j,comparison=cmp,source=original,storage=storage));rows.append(full)
                totals[variant]['steps']+=1;totals[variant]['bitwise']+=cmp['bitwise_equal'];totals[variant]['elements']+=cmp['elements'];totals[variant]['source_bitwise']+=original['bitwise_equal']
            torch.save(rows,out/(label+'_'+variant+'_checks.pt'))
            p.emit('qualification',case=label,variant=variant,fields=len(rows[0]),checks=checks)
            del tr,rows;gc.collect()
        p.ref.guard()
    (out/'qualification.json').write_text(json.dumps(totals,indent=2)+'\n')
    p.emit('qualification_summary',totals=totals)


def eligible(qualification):
    gate=json.loads(qualification.read_text())
    cpu=json.loads((qualification.parent.parent/'audit.json').read_text())['summary']
    choices=[v for v in c.VARIANTS if gate[v]['bitwise']==gate[v]['source_bitwise']==96 and cpu[v]['bitwise']==cpu[v]['source_bitwise']==96]
    assert 'direct' in choices and 'batch_store' in choices
    assert any(v.endswith('nograd') for v in choices), 'No no-grad candidate qualified'
    return choices


def state_compare(tr,expected):
    full,storage=i.expanded_check(tr,tr.state_for_check())
    return dict(comparison=audit.comp(full,{k:expected[k] for k in full}),fields=len(full),storage=storage)


def timing(out,qualification):
    variants=eligible(qualification);rng=random.Random(202609091)
    for name,bs,start,label in cases():
        ck=torch.load(d.ART/(label+'_checkpoint.pt'),weights_only=False);trainers={}
        for v in variants:
            tr=p.Trainer(name,bs);c.prepare(tr,v);tr.restore(ck)
            for j in range(8):i.b.step(tr,start+j,'direct')
            trainers[v]=tr
        for rnd in range(9):
            order=variants.copy();rng.shuffle(order)
            for variant in order:
                tr=trainers[variant];tr.restore(ck);gc.collect();torch.cuda.synchronize()
                a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();begin=time.perf_counter_ns()
                for j in range(8):i.b.step(tr,start+j,'direct')
                z.record();torch.cuda.synchronize()
                p.emit('timing',case=label,variant=variant,round=rnd,order=order,steps=8,
                    timeline_ms=a.elapsed_time(z),wall_ms=(time.perf_counter_ns()-begin)/1e6)
        del trainers,tr,ck;gc.collect();p.ref.guard()


def profiles(out,qualification):
    variants=eligible(qualification)
    for name,bs,start,label in cases():
        ck=torch.load(d.ART/(label+'_checkpoint.pt'),weights_only=False)
        expected=torch.load(ART/(label+'_direct_checks.pt'),weights_only=False)
        for variant in variants:
            tr=p.Trainer(name,bs);clock=c.prepare(tr,variant,'events');tr.restore(ck)
            i.b.step(tr,start,'direct');tr.restore(ck);gc.collect();torch.cuda.synchronize();clock.rows.clear();clock.stats.clear()
            a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
            for j in range(8):i.b.step(tr,start+j,'direct')
            z.record();torch.cuda.synchronize();check=state_compare(tr,expected[-1])
            p.emit('phases',case=label,variant=variant,steps=8,total_timeline_ms=a.elapsed_time(z),**clock.results(),**check)
            assert check['comparison']['bitwise_equal']
            del tr,clock;gc.collect()
        if start in [256,40]:
            for variant in variants:
                tr=p.Trainer(name,bs);c.prepare(tr,variant);tr.restore(ck);observed=c.inspect_state_graph(tr)
                result=i.b.step(tr,start,'direct',check=True)['check'];full,storage=i.expanded_check(tr,result)
                cmp=audit.comp(full,expected[0])
                p.emit('mechanism',case=label,variant=variant,index=start,observations=observed,comparison=cmp,fields=len(full))
                assert cmp['bitwise_equal'] and len(observed)==1 and observed[0]['outer_grad_enabled']
                assert observed[0]['requires_grad']==(not variant.endswith('nograd'))
                assert (observed[0]['reachable_nodes']==0)==variant.endswith('nograd')
                del tr;gc.collect()
        p.ref.guard()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--stage',choices=['qualify','timing','profiles'],required=True)
    ap.add_argument('--qualification',type=Path);args=ap.parse_args();args.output.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p.ref.guard(initial=True);torch.zeros(1,device='cuda');p.ref.guard()
    p.emit('opening',stage=args.stage,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        gpu_uuid=__import__('os').environ['CUDA_VISIBLE_DEVICES'],
        hashes={str(f):p.sha(f) for f in [Path(__file__),ROOT/'src/candidate.py',ROOT/'CONTRACT.md',c.PREV/'src/instrument.py',i.PREV/'src/bridge.py',d.OLD/'src/profile_training.py']})
    if args.stage=='qualify':qualify(args.output)
    elif args.stage=='timing':timing(args.output,args.qualification)
    else:profiles(args.output,args.qualification)
    p.emit('complete',apps=p.ref.guard())


if __name__=='__main__':main()
