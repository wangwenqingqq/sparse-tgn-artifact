import argparse
import json
from pathlib import Path
import instrument as i
import audit as previous_audit
p,torch,d=i.p,i.torch,i.d


def serialize(prof):
    rows=[];events=list(prof.events());uids={id(e):j for j,e in enumerate(events)}
    for e in events:
        rows.append(dict(uid=uids[id(e)],id=e.id,name=e.name,thread=e.thread,fwd_thread=e.fwd_thread,
            seq=e.sequence_nr,scope=e.scope,device=str(e.device_type),start=e.time_range.start,end=e.time_range.end,
            parent=e.cpu_parent.id if e.cpu_parent is not None else None,
            parent_uid=uids.get(id(e.cpu_parent)),is_async=e.is_async,is_user_annotation=e.is_user_annotation,
            self_cpu_us=e.self_cpu_time_total,cpu_us=e.cpu_time_total,
            self_device_us=e.self_device_time_total,device_us=e.device_time_total,
            kernels=[dict(name=k.name,device=k.device,duration=k.duration) for k in e.kernels]))
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--stage');ap.add_argument('--qualification');args=ap.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p.ref.guard(initial=True);torch.zeros(1,device='cuda');p.ref.guard()
    label='wikipedia_b200_s40';ck=torch.load(d.ART/(label+'_checkpoint.pt'),weights_only=False)
    expected=torch.load(d.ART/(label+'_source_replay_checks.pt'),weights_only=False)
    for variant in ['direct','batch_store']:
        tr=p.Trainer('wikipedia',200);clock=i.prepare(tr,variant,'trace');tr.restore(ck)
        # Warm up before restoring the identical starting state for the trace.
        i.b.step(tr,40,'direct');tr.restore(ck);torch.cuda.synchronize();clock.stats.clear()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                   record_shapes=False,with_stack=False,profile_memory=False) as prof:
            result=i.b.step(tr,40,'direct',check=True)
        cmp=previous_audit.comp(result['check'],expected[0])
        prof.export_chrome_trace(str(args.output/(variant+'_trace.json')))
        rows=serialize(prof)
        (args.output/(variant+'_events.json')).write_text(json.dumps(rows))
        torch.save(result['check'],args.output/(variant+'_check.pt'))
        p.emit('pilot',variant=variant,comparison=cmp,stats=clock.stats,events=len(rows),
               device_events=sum(e['device']!='DeviceType.CPU' for e in rows),
               kernel_links=sum(len(e['kernels']) for e in rows))
        del tr,prof,rows
    p.emit('complete',apps=p.ref.guard())


if __name__=='__main__':main()
