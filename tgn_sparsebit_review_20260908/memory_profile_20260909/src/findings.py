from collections import defaultdict
import json
from pathlib import Path
import statistics as st
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'analysis'
s=json.loads((OUT/'summary.json').read_text());traces=json.loads((OUT/'trace_summary.json').read_text())
out=dict(phases={},backward=[],trace_coverage=[])
for variant in ['direct','batch_store']:
    macro=defaultdict(list);detail=defaultdict(list)
    for case in s['cases']:
        phases=case['phases'][variant];percase=defaultdict(float)
        for k,v in phases.items():
            if '/' not in k:macro[k].append(v['timeline_ms_per_step'])
            if k.startswith(('memory_read/','memory_update/')):
                leaf=k.split('/')[-1]
                if leaf in ['message.lookup','message.unpack','message.concat','message.time','message.compose','memory.gru','memory.last_aggregate','memory.last_time','state.write','state.recompute'] or leaf.startswith('store.'):
                    percase[leaf]+=v['timeline_ms_per_step']
        for k,v in percase.items():detail[k].append(v)
    means={k:st.mean(v) for k,v in macro.items()};total=sum(means.values())
    out['phases'][variant]=dict(macro_ms=means,macro_percent={k:100*v/total for k,v in means.items()},
        macro_sum_ms=total,memory_detail_ms={k:st.mean(v) for k,v in detail.items()})
for t in traces:
    gpu=defaultdict(lambda:dict(activities=0,gpu_us=0.,kernels=0));cpu=defaultdict(lambda:dict(nodes=0,cpu_us=0.))
    for k,v in t['activities_by_owner'].items():
        if k.startswith('backward/'):
            g=gpu[k.split('/')[1]];g['activities']+=v['kernel_count']+v['memcpy_count']+v['memset_count'];g['gpu_us']+=v['kernel_us']+v['memcpy_us']+v['memset_us'];g['kernels']+=v['kernel_count']
    for k,v in t['backward_engines'].items():
        g=cpu[k.split('/')[0]];g['nodes']+=v['engine_count'];g['cpu_us']+=v['engine_cpu_us']
    out['backward'].append(dict(case_variant=t['case_variant'],cpu=dict(cpu),gpu=dict(gpu),
        backward_wall_us=t['macros']['backward']['cpu_us']))
    out['trace_coverage'].append(dict(case_variant=t['case_variant'],roi_us=t['roi_us'],activity_union_us=t['gpu_activity_union_us'],
        active_fraction=t['gpu_activity_union_us']/t['roi_us'],linked=t['link_coverage']['linked'],total=t['link_coverage']['total'],
        flow_resolved=len(t['flow_resolved_ambiguities']),ambiguous=len(t['backward_ambiguities']),
        owner_disagreements=len(t['link_coverage']['owner_crosscheck_disagreements'])))
(OUT/'findings.json').write_text(json.dumps(out,indent=2)+'\n')
for t in out['backward']:
    if t['case_variant'].endswith('_direct'):
        print(t['case_variant'],'BW wall ms',round(t['backward_wall_us']/1000,3),'GPU',t['gpu'])
