import json
from pathlib import Path
r=Path(__file__).resolve().parents[1]/'analysis'
rows=[]
for f in sorted(r.glob('*_attribution.json')):
    x=json.loads(f.read_text());backward=[v for v in x['top_gpu_operations'] if v['owner'].startswith('backward/')];total=sum(v['duration_us'] for v in backward)
    mm=sum(v['duration_us'] for v in backward if '.W_kv' in v['owner'] and v['name']=='aten::mm')
    bias=sum(v['duration_us'] for v in backward if '.W_kv' in v['owner'] and v['name']=='aten::sum')
    row=dict(label=f.name.replace('_attribution.json',''),backward_gpu_us=total,wkv_mm_us=mm,wkv_bias_reduce_us=bias,wkv_mm_percent=100*mm/total,wkv_mm_bias_percent=100*(mm+bias)/total,linked=x['link_coverage']['linked'],total=x['link_coverage']['total'],ambiguous=len(x['backward_ambiguities']),crosscheck_disagreements=len(x['link_coverage']['owner_crosscheck_disagreements']))
    assert row['linked']==row['total'] and not row['ambiguous'] and not row['crosscheck_disagreements']
    rows.append(row)
assert len(rows)==8
(r/'kernel_findings.json').write_text(json.dumps(dict(cases=rows,total_activities=sum(x['total'] for x in rows),linked_activities=sum(x['linked'] for x in rows)),indent=2)+'\n')
