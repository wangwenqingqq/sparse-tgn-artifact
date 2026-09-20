"""Summarize all declared configurations; failed qualification stays failed."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def read(run):
    records=[]
    for line in (ROOT/'evidence'/run/'stdout.jsonl').read_text().splitlines():
        if line.startswith('{'):
            try:records.append(json.loads(line))
            except json.JSONDecodeError:raise
    return records


def stats(values):
    a=np.array(values,dtype=float)
    return dict(n=len(a),median=float(np.median(a)),p10=float(np.quantile(a,.1)),p90=float(np.quantile(a,.9)),min=float(a.min()),max=float(a.max()),observations=a.tolist())


def qualifications():
    result={}
    for run in ['qualify_v1','qualify_repaired_v1']:
        root=ROOT/'evidence'/run;q=json.loads((root/'artifacts/qualification.json').read_text());audit=json.loads((root/'audit.json').read_text())
        assert q['all_pass']==audit['all_pass']==False
        rows=[]
        for row in q['cases']:
            arms={}
            for arm,steps in row['comparisons'].items():
                arms[arm]=dict(passed=sum(x['pass_all'] for x in steps),bitwise=sum(x['bitwise_equal'] for x in steps),steps=len(steps),first_step_failures=steps[0]['failures'],first_step_different_fields=steps[0]['different_fields'])
            rows.append(dict(case=row['case'],fields=row['fields'],arms=arms,cuda_calls=row['cuda_calls'],operator_counts=row['operator_counts']))
        result[run]=dict(all_pass=False,comparisons=audit['comparisons'],sampler=audit['sampler'],cases=rows)
    return result


def main():
    rows={};all_epochs=[];all_completions=[]
    for rep in [1,2,3]:
        run=f'diagnostic_timing_v{rep}'
        assert json.loads((ROOT/'evidence'/run/'run.json').read_text())['exit_code']==0
        data=read(run);configs={x['case']['label']:x for x in data if x['kind']=='configuration'}
        epochs=[{**x,'rep':rep} for x in data if x['kind']=='epoch'];assert len(epochs)==32
        completed=[{**x,'rep':rep} for x in data if x['kind']=='training_complete'];assert len(completed)==8
        assert all(x['finite'] and not x['cuda_errors'] for x in completed)
        for label,conf in configs.items():
            item=rows.setdefault(label,dict(case=conf['case'],config=conf['config'],train_end=conf['train_end'],epochs=[],completions=[],setups_ms=[]))
            item['setups_ms'].append(conf['setup_ms']);item['epochs'] += [e for e in epochs if e['case']==label]
            item['completions'] += [e for e in completed if e['case']['label']==label]
        all_epochs+=epochs;all_completions+=completed
    result=dict(qualified=False,interpretation='unmodified FlashTGN diagnostic timing; failed self replay, no speedup or quality qualification',qualification=qualifications(),timing={},profiles={})
    for label,item in sorted(rows.items()):
        warm=[x for x in item['epochs'] if x['epoch']>1];assert len(warm)==9
        r=dict(case=item['case'],train_end=item['train_end'],batches=item['train_end']//item['case']['bs'],config=item['config'])
        for metric,path,field in [('epoch_ms','epoch','timeline_ms'),('epoch_host_ms','epoch','host_ms'),('online_ms','epoch/online_loop','timeline_ms'),('schedule_ms','epoch/schedule_full','timeline_ms')]:
            r[metric]=stats([x['phases'].get(path,{}).get(field,0.) for x in warm])
        r['source_loop_ms']=stats([1000*x['source_result']['loop_time'] for x in warm]);r['epoch_loss']=stats([x['source_result']['loss'] for x in warm])
        r['setup_ms']=stats(item['setups_ms']);r['train_invocation_wall_ms']=stats([x['train_invocation_wall_ms'] for x in item['completions']])
        r['peak_allocated_bytes']=stats([x['peak_allocated'] for x in item['completions']]);r['peak_reserved_bytes']=stats([x['peak_reserved'] for x in item['completions']])
        r['cold_epochs']=[x for x in item['epochs'] if x['epoch']==1]
        r['schedule_modes']=['gpu' if item['case']['layers']==1 else 'online']
        r['timing_cuda_calls']=[x['cuda_calls'] for x in item['completions']]
        r['initial_schedule_ms']=[x['initial_phases'].get('schedule_full',{}).get('timeline_ms',0.) for x in item['completions']]
        r['throughput_edges_s']=item['train_end']/(r['epoch_ms']['median']/1000)
        result['timing'][label]=r
    profile=read('diagnostic_profiles_v1');ep=[x for x in profile if x['kind']=='epoch'];assert len(ep)==16
    for label in sorted(rows):
        obs=[x for x in ep if x['case']==label];assert len(obs)==2
        x=next(x for x in obs if x['epoch']==2);ph=x['phases'];total=ph['epoch']['timeline_ms'];online=ph['epoch/online_loop']['timeline_ms']
        categories={};details={}
        for path,v in ph.items():
            # Mutually disjoint top-level online scopes; nested work is in details.
            if path.startswith('epoch/online_loop/') and path.count('/')==2:categories[path.rsplit('/',1)[1]]=v['timeline_ms']
            details[path]=dict(**v,epoch_percent=100*v['timeline_ms']/total)
        categories['online_unscoped']=online-sum(categories.values())
        assert categories['online_unscoped']>=-.1
        for path,v in ph.items():
            if path.startswith('epoch/') and path.count('/')==1 and path!='epoch/online_loop':categories['outer_'+path.split('/')[1]]=v['timeline_ms']
        categories['outer_unscoped']=total-online-sum(v['timeline_ms'] for path,v in ph.items() if path.startswith('epoch/') and path.count('/')==1 and path!='epoch/online_loop')
        assert categories['outer_unscoped']>=-.1
        assert abs(sum(categories.values())-total)<1e-4
        shares={k:100*v/total for k,v in categories.items()}
        mailbox=categories['mailbox_write'];memory=categories['memory_update'];backward=categories['backward']
        backwards={}
        for path,v in ph.items():
            if path.startswith('epoch/online_loop/backward/') and path.count('/')==3:backwards[path.rsplit('/',1)[1]]=v['timeline_ms']
        backwards['unscoped_backward']=backward-sum(backwards.values())
        assert backwards['unscoped_backward']>=-.1
        r=dict(epoch_ms=total,online_ms=online,categories_ms=categories,categories_percent=shares,details=details,
            hypothetical_free_mailbox=total/(total-mailbox),hypothetical_free_memory_update=total/(total-memory),hypothetical_half_backward=total/(total-backward/2),
            backward_components_ms=backwards,backward_components_epoch_percent={k:100*v/total for k,v in backwards.items()},
            cuda_calls=x['cuda_calls'],operator_counts=x['operator_counts'],observations=obs)
        r['operator_counts_epoch']={k:v-obs[0]['operator_counts'].get(k,0) for k,v in x['operator_counts'].items()}
        result['profiles'][label]=r
    result['counts']=dict(timing_epochs=len(all_epochs),timing_measured_epochs=sum(1 for x in all_epochs if x['epoch']>1),profile_epochs=len(ep),timed_batches=sum(rows[x['case']]['train_end']//rows[x['case']]['case']['bs'] for x in all_epochs),profile_batches=sum(rows[x['case']]['train_end']//rows[x['case']]['case']['bs'] for x in ep))
    (ROOT/'analysis/summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'analysis/timing_epochs.json').write_text(json.dumps(all_epochs,indent=2)+'\n')
    (ROOT/'analysis/timing_completions.json').write_text(json.dumps(all_completions,indent=2)+'\n')
    lines=['|batch_size|邻居数|层数|整 epoch 中位秒 [p10,p90]|online 秒|schedule 秒|千边/秒|','|---:|---:|---:|---|---:|---:|---:|']
    for r in result['timing'].values():
        c=r['case'];e=r['epoch_ms'];lines.append(f"|{c['bs']}|{c['k']}|{c['layers']}|{e['median']/1000:.3f} [{e['p10']/1000:.3f}, {e['p90']/1000:.3f}]|{r['online_ms']['median']/1000:.3f}|{r['schedule_ms']['median']/1000:.3f}|{r['throughput_edges_s']/1000:.1f}|")
    (ROOT/'analysis/timing_table.md').write_text('\n'.join(lines)+'\n')
    lines=['|batch_size|邻居数|层数|反向 %|embedding %|memory update %|mailbox 写入 %|假设写入免费倍数|','|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label,r in result['profiles'].items():
        c=result['timing'][label]['case'];s=r['categories_percent'];lines.append(f"|{c['bs']}|{c['k']}|{c['layers']}|{s['backward']:.1f}|{s['embedding']:.1f}|{s['memory_update']:.1f}|{s['mailbox_write']:.1f}|{r['hypothetical_free_mailbox']:.3f}|")
    (ROOT/'analysis/profile_table.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result['counts']))

if __name__=='__main__':main()
