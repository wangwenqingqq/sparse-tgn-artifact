import argparse
import gc
import json
from pathlib import Path
import random
import time
import event_store as o
import audit as audit
p,torch,d,i=o.p,o.torch,o.d,o.i
ROOT=Path(__file__).resolve().parents[1]
ART=o.PREV/'output/qualify_v1/artifacts'
VARIANTS=['batch_store','event_index']


def cases():
    for name in ['wikipedia','college']:
        for bs in [32,200]:
            for start in ([128,256,512] if bs==32 else [20,40,80]):
                yield name,bs,start,f'{name}_b{bs}_s{start}'


def qualify(out):
    totals={v:dict(steps=0,bitwise=0,elements=0,source_bitwise=0) for v in VARIANTS}
    for name,bs,start,label in cases():
        ck=torch.load(d.ART/(label+'_checkpoint.pt'),weights_only=False)
        source=torch.load(d.ART/(label+'_source_replay_checks.pt'),weights_only=False)
        expected=torch.load(ART/(label+'_direct_checks.pt'),weights_only=False)
        for variant in VARIANTS:
            tr=p.Trainer(name,bs);o.prepare(tr,variant,start,capture=True);tr.restore(ck);checks=[];rows=[]
            for j in range(8):
                result=o.step(tr,start+j,check=True)['check']
                full,storage=o.expanded_check(tr,result)
                cmp=audit.comp(full,expected[j]);original=audit.comp(result,source[j])
                checks.append(dict(index=start+j,comparison=cmp,source=original,storage=storage));rows.append(full)
                totals[variant]['steps']+=1;totals[variant]['bitwise']+=cmp['bitwise_equal'];totals[variant]['elements']+=cmp['elements'];totals[variant]['source_bitwise']+=original['bitwise_equal']
            torch.save(rows,out/(label+'_'+variant+'_checks.pt'))
            p.emit('qualification',case=label,variant=variant,fields=len(rows[0]),checks=checks)
            if variant=='event_index':
                assert len(tr.event_state.records)==32
                torch.save(tr.event_state.records,out/(label+'_tape.pt'))
                p.emit('setup',case=label,variant=variant,**tr.event_state.setup)
            del tr,rows;gc.collect()
        gc.collect();p.ref.guard()
    (out/'qualification.json').write_text(json.dumps(totals,indent=2)+'\n')
    p.emit('qualification_summary',totals=totals)


def gate(qualification):
    q=json.loads(qualification.read_text());cpu=json.loads((qualification.parent.parent/'audit.json').read_text())['summary']
    assert all(q[v]['bitwise']==q[v]['source_bitwise']==cpu[v]['bitwise']==cpu[v]['source_bitwise']==96 for v in VARIANTS)
    assert cpu['prefix']['calls']==384 and cpu['prefix']['exact']
    assert cpu['event_gathers']['calls']==384 and cpu['event_gathers']['exact']


def state_compare(tr,expected):
    full,storage=o.expanded_check(tr,tr.state_for_check())
    return dict(comparison=audit.comp(full,{k:expected[k] for k in full}),fields=len(full),storage=storage)


def timing(out,qualification):
    gate(qualification);variants=VARIANTS;rng=random.Random(202609093)
    for name,bs,start,label in cases():
        ck=torch.load(d.ART/(label+'_checkpoint.pt'),weights_only=False)
        trainers={}
        for variant in variants:
            tr=p.Trainer(name,bs);o.prepare(tr,variant,start);tr.restore(ck)
            for j in range(8):o.step(tr,start+j)
            trainers[variant]=tr
        p.emit('setup',case=label,variant='event_index',**trainers['event_index'].event_state.setup)
        for rnd in range(9):
            order=variants.copy();rng.shuffle(order)
            for variant in order:
                tr=trainers[variant];tr.restore(ck);gc.collect();torch.cuda.synchronize()
                a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();begin=time.perf_counter_ns()
                for j in range(8):o.step(tr,start+j)
                z.record();torch.cuda.synchronize()
                p.emit('timing',case=label,variant=variant,round=rnd,order=order,steps=8,
                    timeline_ms=a.elapsed_time(z),wall_ms=(time.perf_counter_ns()-begin)/1e6)
        del trainers,tr,ck;gc.collect();p.ref.guard()


def profiles(out,qualification):
    gate(qualification)
    for name,bs,start,label in cases():
        ck=torch.load(d.ART/(label+'_checkpoint.pt'),weights_only=False)
        expected=torch.load(ART/(label+'_direct_checks.pt'),weights_only=False)
        for variant in VARIANTS:
            tr=p.Trainer(name,bs);clock=o.prepare(tr,variant,start,'events');tr.restore(ck)
            o.step(tr,start);tr.restore(ck);gc.collect();torch.cuda.synchronize();clock.rows.clear();clock.stats.clear()
            a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
            for j in range(8):o.step(tr,start+j)
            z.record();torch.cuda.synchronize();check=state_compare(tr,expected[-1])
            p.emit('phases',case=label,variant=variant,steps=8,total_timeline_ms=a.elapsed_time(z),**clock.results(),**check)
            assert check['comparison']['bitwise_equal']
            del tr,clock;gc.collect()
        gc.collect();p.ref.guard()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--stage',choices=['qualify','timing','profiles'],required=True)
    ap.add_argument('--qualification',type=Path);args=ap.parse_args();args.output.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p.ref.guard(initial=True);torch.zeros(1,device='cuda');p.ref.guard()
    module,_,_=o.source.source_method()
    p.emit('opening',stage=args.stage,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        gpu_uuid=__import__('os').environ['CUDA_VISIBLE_DEVICES'],
        hashes={str(f):p.sha(f) for f in [Path(__file__),ROOT/'src/event_store.py',ROOT/'CONTRACT.md',o.PREV/'src/instrument.py',i.PREV/'src/bridge.py',d.OLD/'src/profile_training.py',Path(module.__file__)]})
    if args.stage=='qualify':qualify(args.output)
    elif args.stage=='timing':timing(args.output,args.qualification)
    else:profiles(args.output,args.qualification)
    p.emit('complete',apps=p.ref.guard())


if __name__=='__main__':main()
