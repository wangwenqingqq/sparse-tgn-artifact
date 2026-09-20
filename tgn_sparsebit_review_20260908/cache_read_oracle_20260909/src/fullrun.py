import argparse
import gc
import json
from pathlib import Path
import random
import time
import read_oracle as o
import audit as audit
p,torch,d,i=o.p,o.torch,o.d,o.i
ROOT=Path(__file__).resolve().parents[1]
ART=o.PREV/'output/qualify_v1/artifacts'
VARIANTS=['capture','batch_store','free_cache_reads']


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
        tape=o.Tape(verify=True)
        for variant in VARIANTS:
            tr=p.Trainer(name,bs);o.prepare(tr,variant,tape);tr.restore(ck);checks=[];rows=[]
            for j in range(8):
                result=o.step(tr,start+j,start,tape if variant=='free_cache_reads' else None,check=True)['check']
                full,storage=i.expanded_check(tr,result)
                cmp=audit.comp(full,expected[j]);original=audit.comp(result,source[j])
                checks.append(dict(index=start+j,comparison=cmp,source=original,storage=storage));rows.append(full)
                totals[variant]['steps']+=1;totals[variant]['bitwise']+=cmp['bitwise_equal'];totals[variant]['elements']+=cmp['elements'];totals[variant]['source_bitwise']+=original['bitwise_equal']
            torch.save(rows,out/(label+'_'+variant+'_checks.pt'))
            p.emit('qualification',case=label,variant=variant,fields=len(rows[0]),checks=checks)
            if variant=='capture':
                assert len(tape.records)==32
                assert [r['direction'] for r in tape.records]==['s','d','s','d']*8
                torch.save(tape.cpu_records(),out/(label+'_tape.pt'))
                p.emit('tape',case=label,**tape.stats())
            del tr,rows;gc.collect()
        del tape;gc.collect();p.ref.guard()
    (out/'qualification.json').write_text(json.dumps(totals,indent=2)+'\n')
    p.emit('qualification_summary',totals=totals)


def gate(qualification):
    q=json.loads(qualification.read_text());cpu=json.loads((qualification.parent.parent/'audit.json').read_text())['summary']
    assert all(q[v]['bitwise']==q[v]['source_bitwise']==cpu[v]['bitwise']==cpu[v]['source_bitwise']==96 for v in VARIANTS)
    assert cpu['prefix']['calls']==384 and cpu['prefix']['exact']


def load_tape(qualification,label,verify=False):
    records=torch.load(qualification.parent/(label+'_tape.pt'),map_location='cuda',weights_only=False)
    assert len(records)==32
    return o.Tape(records,verify=verify)


def state_compare(tr,expected):
    full,storage=i.expanded_check(tr,tr.state_for_check())
    return dict(comparison=audit.comp(full,{k:expected[k] for k in full}),fields=len(full),storage=storage)


def timing(out,qualification):
    gate(qualification);variants=['batch_store','free_cache_reads'];rng=random.Random(202609092)
    for name,bs,start,label in cases():
        ck=torch.load(d.ART/(label+'_checkpoint.pt'),weights_only=False)
        tape=load_tape(qualification,label);trainers={}
        # Both arms retain the identical oracle tensors throughout timing.
        for variant in variants:
            tr=p.Trainer(name,bs);o.prepare(tr,variant,tape);tr.restore(ck)
            for j in range(8):o.step(tr,start+j,start,tape if variant=='free_cache_reads' else None)
            trainers[variant]=tr
        p.emit('resident_tape',case=label,**tape.stats())
        for rnd in range(9):
            order=variants.copy();rng.shuffle(order)
            for variant in order:
                tr=trainers[variant];tr.restore(ck);gc.collect();torch.cuda.synchronize()
                a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();begin=time.perf_counter_ns()
                for j in range(8):o.step(tr,start+j,start,tape if variant=='free_cache_reads' else None)
                z.record();torch.cuda.synchronize()
                p.emit('timing',case=label,variant=variant,round=rnd,order=order,steps=8,
                    timeline_ms=a.elapsed_time(z),wall_ms=(time.perf_counter_ns()-begin)/1e6)
        del trainers,tr,tape,ck;gc.collect();p.ref.guard()


def profiles(out,qualification):
    gate(qualification)
    for name,bs,start,label in cases():
        ck=torch.load(d.ART/(label+'_checkpoint.pt'),weights_only=False)
        expected=torch.load(ART/(label+'_direct_checks.pt'),weights_only=False)
        tape=load_tape(qualification,label)
        for variant in ['batch_store','free_cache_reads']:
            tr=p.Trainer(name,bs);clock=o.prepare(tr,variant,tape,'events');tr.restore(ck)
            use=tape if variant=='free_cache_reads' else None
            o.step(tr,start,start,use);tr.restore(ck);gc.collect();torch.cuda.synchronize();clock.rows.clear();clock.stats.clear()
            a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
            for j in range(8):o.step(tr,start+j,start,use)
            z.record();torch.cuda.synchronize();check=state_compare(tr,expected[-1])
            p.emit('phases',case=label,variant=variant,steps=8,total_timeline_ms=a.elapsed_time(z),**clock.results(),**check)
            assert check['comparison']['bitwise_equal']
            del tr,clock;gc.collect()
        del tape;gc.collect();p.ref.guard()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--stage',choices=['qualify','timing','profiles'],required=True)
    ap.add_argument('--qualification',type=Path);args=ap.parse_args();args.output.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p.ref.guard(initial=True);torch.zeros(1,device='cuda');p.ref.guard()
    module,_,_=o.source_method()
    p.emit('opening',stage=args.stage,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        gpu_uuid=__import__('os').environ['CUDA_VISIBLE_DEVICES'],
        hashes={str(f):p.sha(f) for f in [Path(__file__),ROOT/'src/read_oracle.py',ROOT/'CONTRACT.md',o.PREV/'src/instrument.py',i.PREV/'src/bridge.py',d.OLD/'src/profile_training.py',Path(module.__file__)]})
    if args.stage=='qualify':qualify(args.output)
    elif args.stage=='timing':timing(args.output,args.qualification)
    else:profiles(args.output,args.qualification)
    p.emit('complete',apps=p.ref.guard())


if __name__=='__main__':main()
