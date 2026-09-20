import argparse
import dataclasses
import gc
import json
import math
from pathlib import Path
import random
import time
import core as c
p=c.torch
ROOT=c.ROOT


def qualify(out,monitor):
    summary=[]
    for case in c.cases():
        c.ACTIVE_CLOCK=None;monitor.reset();monitor.capture_samples=True
        tr=c.make(case);clock=c.Clock('none');c.attach(tr,clock,monitor);c.capture_steps(tr)
        schedules=tr._build_schedules(0,12*case['bs'])
        assert len(schedules)==12
        monitor.capture_samples=False
        p.save(monitor.samples,out/(case['label']+'_sampler.pt'))
        tr.run_epoch(0,8*case['bs'],train=True,schedules=schedules[:8],precomp_mode='gpu')
        ck=c.checkpoint(tr);records={};comparisons={}
        for arm,mode in [('source_a','none'),('source_b','none'),('instrumented','events')]:
            c.restore(tr,ck);tr.capture=True;tr.captured=[];clock.mode=mode;clock.rows.clear()
            tr.run_epoch(8*case['bs'],12*case['bs'],train=True,schedules=schedules[8:],precomp_mode='gpu')
            p.cuda.synchronize();assert len(tr.captured)==4
            records[arm]=tr.captured
            p.save(tr.captured,out/(case['label']+'_'+arm+'.pt'))
            if arm!='source_a':comparisons[arm]=[c.compare(a,b) for a,b in zip(tr.captured,records['source_a'])]
            tr.capture=False
        row=dict(case=case,comparisons=comparisons,cuda_calls=monitor.counts.copy(),cuda_errors=monitor.errors.copy(),
            operator_counts=getattr(c.flash_tgn,'_flash_tgn_operator_counts',{}).copy(),
            fields=len(records['source_a'][0]),steps=4)
        summary.append(row);c.emit('qualification',**row)
        c.ACTIVE_CLOCK=None;del tr,clock,ck,records,schedules;gc.collect();p.cuda.empty_cache();c.guard()
        assert not monitor.errors,monitor.errors
        c.reset_operator_counters()
    result=dict(cases=summary,all_pass=all(x['pass_all'] for r in summary for rows in r['comparisons'].values() for x in rows))
    (out/'qualification.json').write_text(json.dumps(result,indent=2)+'\n')
    c.emit('qualification_summary',all_pass=result['all_pass'],cases=len(summary))


def gate(path):
    q=json.loads(path.read_text());a=json.loads((path.parent.parent/'audit.json').read_text())
    assert q['all_pass'] and a['all_pass'] and a['sampler']['all_pass'],'baseline qualification failed'


def run_training(out,monitor,mode,rep,qualification):
    gate(qualification)
    cases=c.cases();random.Random(202609090+rep).shuffle(cases)
    c.emit('case_order',rep=rep,cases=cases)
    for case in cases:
        c.ACTIVE_CLOCK=None;monitor.reset();monitor.capture_samples=False;c.reset_operator_counters()
        setup=time.perf_counter_ns();tr=c.make(case,epochs=4 if mode=='coarse' else 2)
        p.cuda.synchronize();setup_ms=(time.perf_counter_ns()-setup)/1e6
        clock=c.Clock(mode);c.attach(tr,clock,monitor)
        p.cuda.reset_peak_memory_stats();p.cuda.synchronize()
        c.emit('configuration',case=case,config=dataclasses.asdict(tr.cfg),setup_ms=setup_ms,
            train_end=c.tm.split_edges(tr.graph.num_edges,tr.cfg.bsize)[0],
            fp32_precision=p.get_float32_matmul_precision(),tf32=p.backends.cuda.matmul.allow_tf32,
            deterministic=p.are_deterministic_algorithms_enabled())
        start=time.perf_counter_ns();tr.train();p.cuda.synchronize()
        elapsed=(time.perf_counter_ns()-start)/1e6
        final=c.snapshot(tr)
        finite=all(bool(p.isfinite(v).all()) for v in final.values() if p.is_tensor(v) and v.is_floating_point())
        p.save(final,out/(case['label']+'_final.pt'))
        c.emit('training_complete',case=case,mode=mode,rep=rep,train_invocation_wall_ms=elapsed,
            finite=finite,fields=len(final),initial_phases=clock.results(0),
            cuda_calls=monitor.counts.copy(),cuda_errors=monitor.errors.copy(),
            operator_counts=getattr(c.flash_tgn,'_flash_tgn_operator_counts',{}).copy(),
            peak_allocated=p.cuda.max_memory_allocated(),peak_reserved=p.cuda.max_memory_reserved())
        assert finite and not monitor.errors
        c.ACTIVE_CLOCK=None;del tr,clock,final;gc.collect();p.cuda.empty_cache();c.guard()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--stage',choices=['qualify','timing','profiles'],required=True)
    ap.add_argument('--rep',type=int,default=0);ap.add_argument('--qualification',type=Path)
    args=ap.parse_args();args.output.mkdir(exist_ok=False)
    p.set_num_threads(1)
    # Source defaults remain in force; no deterministic/precision substitution.
    assert c.sklearn.__version__=='1.7.2',c.sklearn.__version__
    frozen=json.loads((ROOT/'input_manifest.json').read_text())
    for path,expected in frozen['files'].items():assert c.sha(path)==expected['sha256'],path
    c.guard(initial=True);p.zeros(1,device='cuda');c.guard()
    c.emit('opening',stage=args.stage,rep=args.rep,gpu=p.cuda.get_device_name(),gpu_uuid=c.os.environ['CUDA_VISIBLE_DEVICES'],
        torch=p.__version__,cuda=p.version.cuda,sklearn=c.sklearn.__version__,sklearn_path=c.sklearn.__file__,
        hashes={str(f):c.sha(f) for f in [Path(__file__),ROOT/'src/core.py',ROOT/'CONTRACT.md',ROOT/'input_manifest.json']})
    monitor=c.Monitor();c.install_global_scopes()
    if args.stage=='qualify':qualify(args.output,monitor)
    else:run_training(args.output,monitor,'coarse' if args.stage=='timing' else 'events',args.rep,args.qualification)
    c.emit('complete',apps=c.guard())


if __name__=='__main__':main()
