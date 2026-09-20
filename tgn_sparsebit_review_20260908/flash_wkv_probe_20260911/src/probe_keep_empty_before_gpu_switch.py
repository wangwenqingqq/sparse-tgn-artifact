import argparse
import gc
import json
from pathlib import Path
import random
import time
import common as a
import candidate_keep_empty as ca
from compare import compare
c=a.c;p=a.p
CAPTURE=a.ROOT/'output/capture_v1/artifacts'


def qualify(out,meta):
    rows=[]
    for port in meta['ports']:
        f=CAPTURE/port['input_file'];assert c.sha(f)==port['sha256'];raw=p.load(f,map_location='cpu',weights_only=False)
        bundle=ca.prepare(raw);records={};checks={}
        for arm in ['source_a','source_b','candidate']:
            result=ca.run(bundle,'source' if arm.startswith('source') else 'candidate');records[arm]=a.cpu(result);del result;p.cuda.synchronize()
            p.save(records[arm],out/(port['label']+'_'+arm+'.pt'))
        checks['capture_forward']=compare({'output':records['source_a']['output']},{'output':raw['source_output']})
        checks['source_replay']=compare(records['source_b'],records['source_a']);checks['candidate']=compare(records['candidate'],records['source_a'])
        row=dict(port=port,checks=checks);rows.append(row);c.emit('qualification',label=port['label'],source_packed=port['source_packed'],checks={k:dict(pass_all=v['pass_all'],bitwise=v['bitwise']) for k,v in checks.items()})
        del raw,bundle,records;gc.collect();p.cuda.empty_cache();c.guard()
    result=dict(cases=rows,all_pass=all(x['pass_all'] for row in rows for x in row['checks'].values()))
    (out/'qualification.json').write_text(json.dumps(result,indent=2)+'\n');c.emit('qualification_summary',all_pass=result['all_pass'])


def noise(out,meta):
    module=c.cuda_extension();rows=[]
    for info in meta['noise']:
        f=CAPTURE/info['input_file'];assert c.sha(f)==info['sha256'];raw=p.load(f,map_location='cpu',weights_only=False);args=a.gpu(raw['args']);fn=getattr(module,info['name']);records=[];checks=[]
        for trial in range(16):
            result=fn(*args);record=a.cpu(result);del result;records.append(record)
            checks.append(compare({f'output_{i}':x for i,x in enumerate(record)},{f'output_{i}':x for i,x in enumerate(raw['source_outputs'])}))
        after=a.cpu(args)
        unchanged=compare({f'arg_{i}':x for i,x in enumerate(after) if p.is_tensor(x)},{f'arg_{i}':x for i,x in enumerate(raw['args']) if p.is_tensor(x)})
        assert unchanged['bitwise'];p.save(records,out/(info['label']+'_noise_outputs.pt'))
        row=dict(info=info,checks=checks,input_unchanged=True);rows.append(row)
        c.emit('noise',label=info['label'],pass_count=sum(x['pass_all'] for x in checks),bitwise_count=sum(x['bitwise'] for x in checks),max_scaled=max(f['max_scaled_error'] for x in checks for f in x['fields'].values()))
        del raw,args,records,after;gc.collect();p.cuda.empty_cache();c.guard()
    (out/'noise.json').write_text(json.dumps(dict(cases=rows),indent=2)+'\n')


def timing(out,meta):
    q=json.loads((a.ROOT/'output/qualify_keep_empty_v1/artifacts/qualification.json').read_text());audit=json.loads((a.ROOT/'output/qualify_keep_empty_v1/audit.json').read_text());assert q['all_pass'] and audit['all_pass']
    records=[]
    for port in meta['ports']:
        raw=p.load(CAPTURE/port['input_file'],map_location='cpu',weights_only=False);bundle=ca.prepare(raw)
        for arm in ['source','candidate']:
            for _ in range(4):result=ca.run(bundle,arm);del result
        p.cuda.synchronize();rng=random.Random(20260911+port['step']+port['case']['bs']+len(port['module']))
        for rep in range(9):
            order=['source','candidate'];rng.shuffle(order)
            for arm in order:
                gc.collect();p.cuda.synchronize();start,end=p.cuda.Event(enable_timing=True),p.cuda.Event(enable_timing=True)
                wall=time.perf_counter_ns();start.record()
                for _ in range(5):result=ca.run(bundle,arm);del result
                end.record();end.synchronize();ms=(time.perf_counter_ns()-wall)/1e6
                row=dict(label=port['label'],rep=rep,arm=arm,calls=5,cuda_ms=start.elapsed_time(end),wall_ms=ms);records.append(row);c.emit('interval',**row)
        del raw,bundle;gc.collect();p.cuda.empty_cache();c.guard()
    (out/'timing.json').write_text(json.dumps(dict(intervals=records),indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--stage',choices=['qualify','noise','timing'],required=True);args=ap.parse_args();args.output.mkdir(exist_ok=False)
    a.opening(args.stage,[Path(__file__),a.ROOT/'src/candidate_keep_empty.py',a.ROOT/'src/compare.py',a.ROOT/'AMENDMENT_1.md']);meta=json.loads((CAPTURE/'capture.json').read_text())
    monitor=c.Monitor()
    if args.stage=='qualify':qualify(args.output,meta)
    elif args.stage=='noise':noise(args.output,meta)
    else:timing(args.output,meta)
    assert not monitor.errors;c.emit('complete',cuda_calls=monitor.counts,cuda_errors=monitor.errors,apps=c.guard())

if __name__=='__main__':main()
