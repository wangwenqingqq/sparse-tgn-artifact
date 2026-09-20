import argparse
import dataclasses
import gc
import json
from pathlib import Path
import common as a
c=a.c;p=a.p
TARGET={'fused_gather_encode_backward_v2','fused_l0_gather_encode_backward_v2'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--stage');args=ap.parse_args();args.output.mkdir(exist_ok=False)
    a.opening('capture',[Path(__file__)]);monitor=c.Monitor();module=c.cuda_extension()
    original_attention=a.lm.temporal_attention;status=dict(step=0,active=False,ports=[],noise=[],names={},case=None,position=None)
    def attention(layer,q,z,nbr,policy):
        entry=None
        if status['active']:
            name=status['names'][id(layer)];label=f"{status['case']['label']}_s{status['step']}_{name.replace('.','_')}"
            entry=dict(label=label,case=status['case'],step=status['step'],position=status['position'],module=name,
                q=a.cpu(q),z=a.cpu(z),nbr=a.cpu(nbr),weight=a.cpu(layer.W_kv.weight),bias=a.cpu(layer.W_kv.bias),policy=dataclasses.asdict(policy),heads=layer.num_heads)
        out=original_attention(layer,q,z,nbr,policy)
        if entry is not None:
            entry['source_output']=a.cpu(out)
            out.register_hook(lambda g,e=entry:e.update(upstream=a.cpu(g)))
            status['ports'].append(entry)
        return out
    a.lm.temporal_attention=attention;a.l2.temporal_attention=attention
    for name in TARGET:
        original=getattr(module,name)
        def call(*args,_name=name,_original=original,**kwargs):
            result=_original(*args,**kwargs)
            if status['active']:
                assert not kwargs
                status['noise'].append(dict(name=_name,args=a.cpu(args),source_outputs=a.cpu(result)))
            return result
        setattr(module,name,call)
    metadata=[];noise_meta=[]
    for case in a.cases():
        monitor.reset();c.reset_operator_counters();c.ACTIVE_CLOCK=None;tr=c.make(case,epochs=1)
        n=c.tm.split_edges(tr.graph.num_edges,case['bs'])[0]//case['bs'];targets={8:'early',n//2+1:'middle'}
        status.update(step=0,active=False,ports=[],noise=[],names={id(v):k for k,v in tr.model.named_modules() if isinstance(v,a.lm.TemporalAttentionLayer)},case=case)
        original_step=tr._run_model_step;original_write=tr.state.store_raw_messages
        def step(*args,**kwargs):
            status['active']=status['step'] in targets;status['position']=targets.get(status['step']);return original_step(*args,**kwargs)
        def write(*args,**kwargs):
            result=original_write(*args,**kwargs)
            if status['active']:
                assert len(status['ports'])==len(status['noise'])==case['layers']
                for port in status['ports']:
                    assert 'upstream' in port
                    path=args_out/(port['label']+'_input.pt');p.save(port,path)
                    mask=port['nbr']>=0;valid=int(mask.sum());rows=mask.numel();ratio=valid/rows
                    row={k:port[k] for k in ['label','case','step','position','module']};row.update(rows=rows,valid=valid,valid_ratio=ratio,source_packed=ratio<port['policy']['packed_threshold'],input_file=path.name,bytes=path.stat().st_size,sha256=c.sha(path),q_shape=list(port['q'].shape),z_shape=list(port['z'].shape))
                    metadata.append(row);c.emit('capture',**row)
                for item in status['noise']:
                    label=f"{case['label']}_s{status['step']}_{item['name']}";path=args_out/(label+'_noise_input.pt');p.save(item,path)
                    row=dict(label=label,case=case,step=status['step'],position=status['position'],name=item['name'],input_file=path.name,sha256=c.sha(path),bytes=path.stat().st_size)
                    noise_meta.append(row)
                status['ports']=[];status['noise']=[];gc.collect()
            status['active']=False;status['step']+=1;return result
        args_out=args.output;tr._run_model_step=step;tr.state.store_raw_messages=write;tr.train();p.cuda.synchronize()
        assert status['step']==n and not monitor.errors
        c.emit('case_complete',case=case,batches=n,cuda_calls=monitor.counts.copy(),cuda_errors=monitor.errors.copy())
        del tr,original_step,original_write;gc.collect();p.cuda.empty_cache();c.guard()
    assert len(metadata)==len(noise_meta)==12
    (args.output/'capture.json').write_text(json.dumps(dict(ports=metadata,noise=noise_meta),indent=2)+'\n')
    c.emit('complete',ports=len(metadata),noise=len(noise_meta),apps=c.guard())


if __name__=='__main__':main()
