"""CPU serial-write oracle from archived first-step memory and immutable events."""
from pathlib import Path
import json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/home/data/wangxuran/factor_tgn_sptc_20260902/project/experiments/factor_tgn_20260904_anchored_time_quality_wikipedia_full/data_root/data/wiki')


def inputs(art,case):
    a=torch.load(art/(case['label']+'_source_a.pt'),map_location='cpu',weights_only=False)[0]
    b=torch.load(art/(case['label']+'_source_b.pt'),map_location='cpu',weights_only=False)[0]
    bs=case['bs'];start=8*bs;end=9*bs
    src,dst,ts=[np.load(DATA/(n+'.npy'))[start:end] for n in ['src','dst','ts']]
    edge=torch.load(DATA/'edge_features.pt',map_location='cpu',weights_only=False)[start:end].numpy()
    mem=a['state/mem_data'].numpy()
    assert mem.tobytes()==b['state/mem_data'].numpy().tobytes()
    nodes=np.concatenate([src,dst]).astype(np.int64)
    mails=np.concatenate([np.concatenate([mem[src],mem[dst],edge],1),np.concatenate([mem[dst],mem[src],edge],1)],0)
    times=np.concatenate([ts,ts])
    # Ordinary sequential Python assignments explicitly implement the source docstring.
    winners={}
    for position,node in enumerate(nodes):winners[int(node)]=position
    unique=np.array(sorted(winners),dtype=np.int64);last=np.array([winners[int(n)] for n in unique],dtype=np.int64)
    return dict(a=a,b=b,src=src,dst=dst,ts=ts,edge=edge,nodes=nodes,mails=mails,times=times,unique=unique,last=last)


def check(mail,time,data):
    nodes,mails,times,unique,last=[data[k] for k in ['nodes','mails','times','unique','last']]
    want=mails[last];want_ts=times[last]
    payload_bad=np.any(mail!=want,axis=1);time_bad=time!=want_ts
    mixed=0;pair_mismatch=0
    for j,node in enumerate(unique):
        positions=np.flatnonzero(nodes==node)
        valid=np.all(mails[positions]==mail[j],axis=1)
        mixed+=not bool(valid.any())
        pair_mismatch+=not bool(np.any(valid&(times[positions]==time[j])))
    return dict(touched_nodes=len(unique),duplicate_nodes=int(sum(np.count_nonzero(nodes==n)>1 for n in unique)),
        wrong_payload_rows=int(payload_bad.sum()),wrong_timestamp_rows=int(time_bad.sum()),
        payload_not_any_input_row=int(mixed),payload_timestamp_not_any_input_pair=int(pair_mismatch),
        max_payload_abs=float(np.max(np.abs(mail-want))),max_timestamp_abs=float(np.max(np.abs(time-want_ts))),
        all_pass=not bool(payload_bad.any() or time_bad.any()))


def main():
    assert not torch.cuda.is_initialized();art=ROOT/'output/qualify_v1/artifacts'
    q=json.loads((art/'qualification.json').read_text());rows=[]
    for r in q['cases']:
        case=r['case'];data=inputs(art,case);u=data['unique']
        row=dict(case=case,arms={arm:check(data[key]['state/mail_data'].numpy()[u],data[key]['state/mail_ts'].numpy()[u],data) for arm,key in [('source_a','a'),('source_b','b')]})
        rows.append(row);print(json.dumps(row),flush=True)
    (ROOT/'analysis/mailbox_cpu.json').write_text(json.dumps(dict(cases=rows,cuda_initialized=False),indent=2)+'\n')


if __name__=='__main__':main()
