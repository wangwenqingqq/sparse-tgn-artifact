"""Independent serialized-record CPU audit of mailbox probe outcomes."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]

def main():
    assert not torch.cuda.is_initialized();root=ROOT/'output/mailbox_probe_v1';art=root/'artifacts'
    logged=json.loads((art/'probe.json').read_text());rows=[];hashes={}
    for case in logged['cases']:
        f=art/(case['case']['label']+'_probe.pt');raw=torch.load(f,map_location='cpu',weights_only=False)
        nodes=raw['inputs']['nodes'].numpy();mails=raw['inputs']['mails'].numpy();times=raw['inputs']['times'].numpy();unique=raw['inputs']['unique'].numpy()
        expected={}
        for node,mail,time in zip(nodes,mails,times):expected[int(node)]=(mail,time)
        checks={}
        for arm,observations in raw['outputs'].items():
            checks[arm]=[]
            for trial,v in enumerate(observations):
                wrong_mail=wrong_time=mixed=wrong_pair=0
                for index,node in enumerate(unique):
                    want,ts=expected[int(node)];actual=v['mail'][index].numpy();actual_ts=v['time'][index].numpy()
                    wrong_mail+=want.tobytes()!=actual.tobytes();wrong_time+=ts.tobytes()!=actual_ts.tobytes()
                    possible=np.flatnonzero(nodes==node);valid=[i for i in possible if mails[i].tobytes()==actual.tobytes()]
                    mixed+=not bool(valid);wrong_pair+=not any(times[i].tobytes()==actual_ts.tobytes() for i in valid)
                result=dict(all_pass=wrong_mail==0 and wrong_time==0,wrong_payload_rows=int(wrong_mail),wrong_timestamp_rows=int(wrong_time),payload_not_any_input_row=int(mixed),payload_timestamp_not_any_input_pair=int(wrong_pair))
                for k,v in result.items():assert v==case['checks'][arm][trial][k]
                checks[arm].append(result)
        rows.append(dict(case=case['case'],checks=checks))
        h=hashlib.sha256()
        with f.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
        hashes[f.name]=dict(bytes=f.stat().st_size,sha256=h.hexdigest())
    totals={arm:dict(total=sum(len(r['checks'][arm]) for r in rows),passed=sum(x['all_pass'] for r in rows for x in r['checks'][arm])) for arm in ['original','writer_deterministic','explicit_last']}
    result=dict(cases=rows,totals=totals,hashes=hashes,all_logged_results_match=True,cuda_initialized=False)
    assert not torch.cuda.is_initialized();(root/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(totals))

if __name__=='__main__':main()
