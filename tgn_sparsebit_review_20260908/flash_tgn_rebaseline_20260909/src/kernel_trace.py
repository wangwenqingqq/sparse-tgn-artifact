import argparse
import contextlib
import gc
import json
from pathlib import Path
import core as c
p=c.torch


class TraceClock(c.Clock):
    @contextlib.contextmanager
    def phase(self,name,essential=False):
        if name in ['epoch','online_loop']:
            yield;return
        self.stack.append(name);path='/'.join(self.stack)
        try:
            with p.profiler.record_function('scope:'+path):yield
        finally:self.stack.pop()


def serialize(prof):
    events=list(prof.events());uids={id(e):i for i,e in enumerate(events)}
    return [dict(uid=uids[id(e)],id=e.id,name=e.name,device=str(e.device_type),thread=e.thread,
        fwd_thread=e.fwd_thread,seq=e.sequence_nr,scope=e.scope,is_user_annotation=e.is_user_annotation,
        parent_uid=uids.get(id(e.cpu_parent)) if e.cpu_parent is not None else None,
        start=e.time_range.start,end=e.time_range.end,cpu_us=e.cpu_time_total,self_cpu_us=e.self_cpu_time_total,
        self_device_us=e.self_device_time_total,device_us=e.device_time_total,
        shapes=e.input_shapes,kernels=[dict(name=k.name,duration_us=k.duration) for k in e.kernels]) for e in events]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--stage');ap.add_argument('--rep',type=int);ap.add_argument('--qualification',type=Path)
    args=ap.parse_args();args.output.mkdir(exist_ok=False);p.set_num_threads(1)
    q=json.loads(args.qualification.read_text());assert q['all_pass']==False
    frozen=json.loads((c.ROOT/'input_manifest.json').read_text())
    for path,info in frozen['files'].items():assert c.sha(path)==info['sha256'],path
    c.guard();p.zeros(1,device='cuda');c.guard()
    c.emit('opening',stage='kernel_trace',gpu=p.cuda.get_device_name(),gpu_uuid=c.os.environ['CUDA_VISIBLE_DEVICES'],qualified=False,
        hashes={str(x):c.sha(x) for x in [Path(__file__),c.ROOT/'src/core.py',c.ROOT/'CONTRACT.md',c.ROOT/'AMENDMENT_1.md',c.ROOT/'AMENDMENT_2.md',c.ROOT/'AMENDMENT_3.md',c.ROOT/'input_manifest.json']})
    monitor=c.Monitor();c.install_global_scopes()
    for case in c.cases():
        c.ACTIVE_CLOCK=None;monitor.reset();c.reset_operator_counters();tr=c.make(case,epochs=1)
        clock=TraceClock('none');c.attach(tr,clock,monitor)
        train_end=c.tm.split_edges(tr.graph.num_edges,case['bs'])[0];batches=train_end//case['bs'];mid=batches//2
        observed=[];status=dict(step=0,ctx=None)
        def ready(prof):
            prof.export_chrome_trace(str(args.output/(case['label']+'_trace.json')))
            events=serialize(prof);(args.output/(case['label']+'_events.json')).write_text(json.dumps(events)+'\n')
            observed.append(len(events))
        original_step=tr._run_model_step;original_write=tr.state.store_raw_messages
        def step(*a,**kw):
            assert status['ctx'] is None
            status['ctx']=p.profiler.record_function('scope:training_step');status['ctx'].__enter__()
            return original_step(*a,**kw)
        def write(*a,**kw):
            r=original_write(*a,**kw)
            if status['step']==mid+1:p.cuda.synchronize()
            status['ctx'].__exit__(None,None,None);status['ctx']=None;status['step']+=1;prof.step();return r
        tr._run_model_step=step;tr.state.store_raw_messages=write
        with p.profiler.profile(activities=[p.profiler.ProfilerActivity.CPU,p.profiler.ProfilerActivity.CUDA],
            schedule=p.profiler.schedule(wait=mid,warmup=1,active=1,repeat=1),on_trace_ready=ready,
            record_shapes=True,with_stack=False,profile_memory=False) as prof:
            tr.train()
        assert len(observed)==1 and status['step']==batches and status['ctx'] is None and not monitor.errors
        finite=all(bool(p.isfinite(v).all()) for v in tr.model.parameters())
        assert finite
        c.emit('trace_complete',case=case,batches=batches,recorded_step_zero_based=mid+1,events=observed[0],finite=finite,
            cuda_calls=monitor.counts.copy(),cuda_errors=monitor.errors.copy(),operator_counts=getattr(c.flash_tgn,'_flash_tgn_operator_counts',{}).copy())
        c.ACTIVE_CLOCK=None;del tr,clock,prof,original_step,original_write;gc.collect();p.cuda.empty_cache();c.guard()
    c.emit('complete',apps=c.guard())


if __name__=='__main__':main()
