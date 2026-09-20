import argparse
import contextlib
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import core as c
import fullrun as f
import mailbox_cpu as cpu
import mailbox_repair as repair
from flash_tgn.state import TemporalState
p=c.torch


def probe(out):
    qualification=json.loads((c.ROOT/'output/qualify_v1/artifacts/qualification.json').read_text())
    summary=[]
    for row in qualification['cases']:
        case=row['case'];data=cpu.inputs(c.ROOT/'output/qualify_v1/artifacts',case)
        raw={key:p.from_numpy(data[key].copy()) for key in ['src','dst','ts','edge','nodes','mails','times','unique','last']}
        state=TemporalState.__new__(TemporalState)
        state.mem_data=data['a']['state/mem_data'].cuda();state.device=p.device('cuda')
        state.mail_data=p.zeros_like(data['a']['state/mail_data'],device='cuda');state.mail_ts=p.zeros_like(data['a']['state/mail_ts'],device='cuda')
        edge=raw['edge'].cuda();features=SimpleNamespace(dim_edge=edge.shape[1],edge_features_for_messages=lambda a,b:edge)
        args=(raw['src'].cuda(),raw['dst'].cuda(),raw['ts'].cuda(),8*case['bs'],9*case['bs'],features)
        original=state.store_raw_messages;repair.install(SimpleNamespace(state=state));fixed=state.store_raw_messages
        unique=raw['unique'].cuda();records={};checks={}
        for arm,method,det in [('original',original,False),('writer_deterministic',original,True),('explicit_last',fixed,False)]:
            records[arm]=[];checks[arm]=[]
            for trial in range(16):
                state.mail_data.zero_();state.mail_ts.zero_();old=p.are_deterministic_algorithms_enabled()
                try:
                    p.use_deterministic_algorithms(det);method(*args)
                finally:p.use_deterministic_algorithms(old)
                actual=dict(mail=state.mail_data[unique].cpu(),time=state.mail_ts[unique].cpu())
                records[arm].append(actual)
                checks[arm].append(cpu.check(actual['mail'].numpy(),actual['time'].numpy(),data))
            for check in checks[arm]:assert check['touched_nodes']==len(unique)
        p.save(dict(inputs=raw,outputs=records),out/(case['label']+'_probe.pt'))
        result=dict(case=case,checks=checks,distinct_outputs={arm:len(set((v['mail'].numpy().tobytes(),v['time'].numpy().tobytes()) for v in rows)) for arm,rows in records.items()})
        summary.append(result)
        c.emit('probe',case=case,passed={arm:sum(v['all_pass'] for v in rows) for arm,rows in checks.items()},distinct_outputs=result['distinct_outputs'])
        del data,raw,state,records,args,edge,features,unique,original,fixed;gc.collect();p.cuda.empty_cache();c.guard()
    (out/'probe.json').write_text(json.dumps(dict(cases=summary),indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--stage',required=True,choices=['probe','qualify_repaired','diagnostic_timing','diagnostic_profiles']);ap.add_argument('--rep',type=int,default=0);ap.add_argument('--qualification',type=Path)
    args=ap.parse_args();args.output.mkdir(exist_ok=False);p.set_num_threads(1)
    frozen=json.loads((c.ROOT/'input_manifest.json').read_text())
    for path,expected in frozen['files'].items():assert c.sha(path)==expected['sha256'],path
    c.guard();p.zeros(1,device='cuda');c.guard()
    c.emit('opening',stage=args.stage,rep=args.rep,gpu=p.cuda.get_device_name(),gpu_uuid=c.os.environ['CUDA_VISIBLE_DEVICES'],torch=p.__version__,cuda=p.version.cuda,
        interpretation='diagnostic only; original replay gate failed; amendment 1',
        hashes={str(x):c.sha(x) for x in [Path(__file__),c.ROOT/'src/core.py',c.ROOT/'src/fullrun.py',c.ROOT/'src/mailbox_cpu.py',c.ROOT/'src/mailbox_repair.py',c.ROOT/'CONTRACT.md',c.ROOT/'AMENDMENT_1.md',c.ROOT/'AMENDMENT_2.md',c.ROOT/'input_manifest.json']})
    if args.stage=='probe':probe(args.output)
    else:
        monitor=c.Monitor();c.install_global_scopes()
        if args.stage=='qualify_repaired':
            original_make=c.make
            def make(*a,**kw):
                tr=original_make(*a,**kw);repair.install(tr);return tr
            c.make=make;f.qualify(args.output,monitor)
        else:
            # Explicit diagnostic exception under AMENDMENT_1; no change to v1 gate.
            def diagnostic_gate(path):
                q=json.loads(path.read_text());a=json.loads((path.parent.parent/'audit.json').read_text())
                assert not q['all_pass'] and not a['all_pass'] and a['sampler']['all_pass']
                c.emit('diagnostic_gate',qualified=False,original_qualification=str(path),reason='known source self-replay failure; timing has no accuracy/speedup qualification')
            f.gate=diagnostic_gate
            f.run_training(args.output,monitor,'coarse' if args.stage=='diagnostic_timing' else 'events',args.rep,args.qualification)
    c.emit('complete',apps=c.guard())


if __name__=='__main__':main()
