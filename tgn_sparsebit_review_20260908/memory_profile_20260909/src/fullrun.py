"""Qualification, uninstrumented timing and separately attributed replays."""
import argparse
import gc
import json
from pathlib import Path
import random
import time
import instrument as i
import audit as audit
from pilot import serialize
p,torch,d=i.p,i.torch,i.d
ROOT=Path(__file__).resolve().parents[1]
VARIANTS=['direct','batch_store']


def cases():
    for name in ['wikipedia','college']:
        for bs in [32,200]:
            for start in ([128,256,512] if bs==32 else [20,40,80]):
                yield name,bs,start,f'{name}_b{bs}_s{start}'


def qualify(out):
    totals={v:dict(steps=0,bitwise=0,elements=0) for v in VARIANTS}
    for name,bs,start,label in cases():
        ck=torch.load(d.ART/(label+'_checkpoint.pt'),weights_only=False)
        source=torch.load(d.ART/(label+'_source_replay_checks.pt'),weights_only=False)
        baseline=None
        for variant in VARIANTS:
            tr=p.Trainer(name,bs);i.prepare(tr,variant);tr.restore(ck)
            rows=[];checks=[]
            for j in range(8):
                result=i.b.step(tr,start+j,'direct',check=True)['check']
                full,storage=i.expanded_check(tr,result)
                archived=audit.comp(result,source[j])
                cmp=archived if baseline is None else audit.comp(full,baseline[j])
                checks.append(dict(index=start+j,comparison=cmp,archived_source=archived,storage=storage))
                rows.append(full)
                totals[variant]['steps']+=1;totals[variant]['bitwise']+=cmp['bitwise_equal'];totals[variant]['elements']+=cmp['elements']
            torch.save(rows,out/(label+'_'+variant+'_checks.pt'))
            p.emit('qualification',case=label,variant=variant,fields=len(rows[0]),checks=checks)
            if variant=='direct':baseline=rows
            del tr;gc.collect()
        p.ref.guard()
    (out/'qualification.json').write_text(json.dumps(totals,indent=2)+'\n')
    p.emit('qualification_summary',totals=totals)


def state_compare(tr,expected):
    actual,storage=i.expanded_check(tr,tr.state_for_check())
    cmp=audit.comp(actual,{k:expected[k] for k in actual})
    return dict(comparison=cmp,fields=len(actual),storage=storage)


def timing(out,qualification):
    gate=json.loads(qualification.read_text())
    assert all(gate[v]['bitwise']==96 for v in VARIANTS)
    audit_gate=json.loads((qualification.parent.parent/'audit.json').read_text())['summary']
    assert audit_gate['candidate']['bitwise']==96 and audit_gate['baseline_source']['bitwise']==96
    rng=random.Random(20260909)
    for name,bs,start,label in cases():
        ck=torch.load(d.ART/(label+'_checkpoint.pt'),weights_only=False)
        trainers={}
        for v in VARIANTS:
            tr=p.Trainer(name,bs);i.prepare(tr,v);tr.restore(ck)
            for j in range(8):i.b.step(tr,start+j,'direct')
            trainers[v]=tr
        for rnd in range(9):
            order=VARIANTS.copy();rng.shuffle(order)
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
    assert all(v['bitwise']==96 for v in json.loads(qualification.read_text()).values())
    for name,bs,start,label in cases():
        ck=torch.load(d.ART/(label+'_checkpoint.pt'),weights_only=False)
        expected=torch.load(qualification.parent/(label+'_direct_checks.pt'),weights_only=False)
        for variant in VARIANTS:
            tr=p.Trainer(name,bs);clock=i.prepare(tr,variant,'events');tr.restore(ck)
            i.b.step(tr,start,'direct');tr.restore(ck);gc.collect();torch.cuda.synchronize();clock.rows.clear();clock.stats.clear()
            a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();begin=time.perf_counter_ns()
            for j in range(8):i.b.step(tr,start+j,'direct')
            z.record();torch.cuda.synchronize()
            elapsed=(time.perf_counter_ns()-begin)/1e6
            check=state_compare(tr,expected[-1])
            p.emit('phases',case=label,variant=variant,steps=8,total_timeline_ms=a.elapsed_time(z),wall_ms=elapsed,
                   **clock.results(),**check)
            assert check['comparison']['bitwise_equal']
            del tr,clock;gc.collect()
        if start not in [256,40]:continue
        for variant in VARIANTS:
            tr=p.Trainer(name,bs);clock=i.prepare(tr,variant,'trace');tr.restore(ck)
            i.b.step(tr,start,'direct');tr.restore(ck);gc.collect();torch.cuda.synchronize();clock.stats.clear()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                       record_shapes=False,with_stack=False,profile_memory=False) as prof:
                result=i.b.step(tr,start,'direct')
                torch.cuda.synchronize()
            check=state_compare(tr,expected[0])
            prefix=label+'_'+variant
            prof.export_chrome_trace(str(out/(prefix+'_trace.json')))
            events=serialize(prof)
            (out/(prefix+'_events.json')).write_text(json.dumps(events))
            p.emit('trace',case=label,variant=variant,index=start,stats=clock.stats,events=len(events),
                   loss=result['loss'],**check)
            assert check['comparison']['bitwise_equal']
            del tr,clock,prof,events;gc.collect()
        p.ref.guard()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--stage',choices=['qualify','timing','profiles'],required=True)
    ap.add_argument('--qualification',type=Path);args=ap.parse_args();args.output.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p.ref.guard(initial=True);torch.zeros(1,device='cuda');p.ref.guard()
    p.emit('opening',stage=args.stage,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
           gpu_uuid=__import__('os').environ['CUDA_VISIBLE_DEVICES'],
           hashes={str(f):p.sha(f) for f in [Path(__file__),ROOT/'src/instrument.py',ROOT/'src/pilot.py',i.PREV/'src/bridge.py',d.OLD/'src/profile_training.py']})
    if args.stage=='qualify':qualify(args.output)
    elif args.stage=='timing':timing(args.output,args.qualification)
    else:profiles(args.output,args.qualification)
    p.emit('complete',apps=p.ref.guard())


if __name__=='__main__':main()
