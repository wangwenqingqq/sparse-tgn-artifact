"""Attribute Chrome CUDA activities using CPU UID trees and forward sequence IDs."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import statistics as st
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'analysis';OUT.mkdir(exist_ok=True)
DATA=ROOT/'evidence/profiles_v1/artifacts'
FORWARD={'memory_read','memory_update','gnn_embedding','source_decoder','loss','query_mapping'}
ENGINE='autograd::engine::evaluate_function: '


def union(intervals):
    total=0;end=None
    for a,b in sorted(intervals):
        if end is None or a>=end:total+=b-a;end=b
        elif b>end:total+=b-end;end=b
    return total


def analyze(path):
    ev=json.loads(path.read_text());byuid={e['uid']:e for e in ev}
    chrome=json.loads(path.with_name(path.name.replace('_events.json','_trace.json')).read_text())['traceEvents']
    cpu=[e for e in ev if e['device']=='DeviceType.CPU']
    def ancestors(e):
        seen=set()
        while e is not None:
            assert e['uid'] not in seen;seen.add(e['uid']);yield e
            e=byuid.get(e['parent_uid'])
    def scope(e):
        return next((a['name'][6:] for a in ancestors(e) if a['name'].startswith('scope:')),None)
    owners={e['uid']:scope(e) for e in cpu}
    seq=defaultdict(set);seq_events=defaultdict(list)
    for e in cpu:
        s=owners[e['uid']]
        if s and s.split('/')[0] in FORWARD and e['scope']==0 and e['seq']>=0 and not e['is_user_annotation']:
            seq[(e['thread'],e['seq'])].add(s);seq_events[(e['thread'],e['seq'])].append(e['name'])
    ambiguous=[]
    def backward_owner(e):
        key=(e['fwd_thread'],e['seq']);possible=sorted(seq.get(key,[]))
        if e['seq']<0:return 'leaf_or_unsequenced'
        if not possible:return 'unassigned'
        if len(possible)==1:return possible[0]
        # Nested scopes can have the same sequence on an outer module and its operation.
        deepest=max(possible,key=len)
        if all(deepest==x or deepest.startswith(x+'/') for x in possible):return deepest
        ambiguous.append(dict(backward=e['name'],key=key,scopes=possible,forward_ops=seq_events[key]))
        return 'ambiguous'
    engines=[e for e in cpu if e['name'].startswith(ENGINE)]
    engine_owners={e['uid']:backward_owner(e) for e in engines}
    def owner(e):
        engine=next((a for a in ancestors(e) if a['name'].startswith(ENGINE)),None)
        if engine:return 'backward/'+engine_owners[engine['uid']]
        return owners.get(e['uid']) or 'unassigned'
    chrome_cpu=[e for e in chrome if e.get('cat') in ['cpu_op','user_annotation'] and e.get('ph')=='X']
    chrome_keys={(e['args'].get('External id'),e['name']) for e in chrome_cpu}
    external=defaultdict(list)
    for e in cpu:
        if (e['id'],e['name']) in chrome_keys:external[e['id']].append(e)
    # CPU runtime records sometimes reuse an op's external ID. Only cpu_op and
    # user_annotation records enter this map, never the runtime/driver aliases.
    mapping_duplicates={str(k):[e['name'] for e in v] for k,v in external.items() if len(v)>1}
    correlation={e['args']['correlation']:e['args'].get('External id') for e in chrome
        if e.get('cat') in ['cuda_runtime','cuda_driver'] and 'correlation' in e.get('args',{})}
    scopes=[e for e in cpu if e['name'].startswith('scope:')]
    macro=[e for e in scopes if '/' not in e['name']]
    first=min(macro,key=lambda e:e['start']);last=max(macro,key=lambda e:e['end'])
    match=next(e for e in chrome_cpu if e['name']==first['name'] and e['args'].get('External id')==first['id'])
    offset=match['ts']-first['start'];lo=first['start']+offset;hi=last['end']+offset
    activities=[e for e in chrome if e.get('cat') in ['kernel','gpu_memcpy','gpu_memset'] and e.get('ph')=='X']
    groups=defaultdict(lambda:dict(kernel_count=0,kernel_us=0.,memcpy_count=0,memcpy_us=0.,memset_count=0,memset_us=0.))
    rows=[];unmatched=[];outside=[]
    for e in activities:
        ext=e['args'].get('External id');found=external.get(ext,[])
        method='external_id'
        if not found:
            found=external.get(correlation.get(e['args'].get('correlation')),[]);method='runtime_correlation'
        own=owner(found[0]) if len(found)==1 else ('ambiguous_link' if found else 'unlinked')
        if not(lo<=e['ts']<=hi):outside.append(dict(name=e['name'],owner=own));continue
        cat=e['cat'].removeprefix('gpu_');groups[own][cat+'_count']+=1;groups[own][cat+'_us']+=e['dur']
        rows.append(dict(name=e['name'],owner=own,cat=cat,duration_us=e['dur'],start=e['ts']-lo,
            end=e['ts']-lo+e['dur'],link=method if len(found)==1 else own,
            cpu_op=found[0]['name'] if len(found)==1 else None))
        if len(found)!=1:unmatched.append(dict(name=e['name'],external_id=ext,correlation=e['args'].get('correlation'),owner=own))
    backward=defaultdict(lambda:dict(engine_count=0,engine_cpu_us=0.))
    for e in engines:
        g=backward[engine_owners[e['uid']]];g['engine_count']+=1;g['engine_cpu_us']+=e['cpu_us']
    macros={}
    for e in macro:
        a,b=e['start']-first['start'],e['end']-first['start']
        active=union([(max(a,r['start']),min(b,r['end'])) for r in rows if r['end']>a and r['start']<b])
        macros[e['name'][6:]]=dict(cpu_us=e['cpu_us'],gpu_activity_union_us=active,gpu_activity_gap_us=b-a-active)
    operations=Counter((owner(e),e['name']) for e in cpu if (e['id'],e['name']) in chrome_keys and not e['is_user_annotation'])
    result=dict(case_variant=path.name.removesuffix('_events.json'),macros=macros,
        roi_us=hi-lo,gpu_activity_union_us=union([(max(0,r['start']),min(hi-lo,r['end'])) for r in rows]),
        gpu_activity_sum_us=sum(r['duration_us'] for r in rows),gpu_activities=len(rows),
        cuda_kernels=sum(r['cat']=='kernel' for r in rows),activities_by_owner=dict(groups),
        backward_engines=dict(backward),backward_ambiguities=ambiguous,
        link_coverage=dict(linked=len(rows)-len(unmatched),total=len(rows),unmatched=unmatched,
            duplicate_cpu_external_ids=mapping_duplicates,outside_roi=outside),
        cpu_ops=[dict(owner=a,name=b,count=n) for (a,b),n in operations.items()],
        top_gpu_operations=[dict(owner=a,name=b,**v) for (a,b),v in aggregate_ops(rows).items()])
    (OUT/(result['case_variant']+'_attribution.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result


def aggregate_ops(rows):
    out=defaultdict(lambda:dict(count=0,duration_us=0.))
    for r in rows:
        g=out[(r['owner'],r['cpu_op'] or r['name'])];g['count']+=1;g['duration_us']+=r['duration_us']
    return dict(sorted(out.items(),key=lambda kv:kv[1]['duration_us'],reverse=True))


if __name__=='__main__':
    result=[analyze(p) for p in sorted(DATA.glob('*_events.json'))]
    assert len(result)==8
    compact=[{k:r[k] for k in ['case_variant','macros','roi_us','gpu_activity_union_us','gpu_activity_sum_us','gpu_activities','cuda_kernels','activities_by_owner','backward_engines','backward_ambiguities','link_coverage']} for r in result]
    (OUT/'trace_summary.json').write_text(json.dumps(compact,indent=2)+'\n')
    for r in compact:
        print(r['case_variant'],r['roi_us'],r['gpu_activity_union_us'],r['cuda_kernels'],r['link_coverage']['linked'],r['link_coverage']['total'],'ambiguous',len(r['backward_ambiguities']))
        print(r['backward_engines'])
