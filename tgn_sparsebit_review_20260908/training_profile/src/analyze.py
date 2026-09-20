"""Analyze only complete profiling output; no fabricated values for missing runs."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics as st

p = argparse.ArgumentParser()
p.add_argument('--input', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
rows = [json.loads(s) for s in args.input.read_text().splitlines() if s.strip()]
assert rows[-1]['kind'] == 'complete', 'Incomplete run: retain raw log, do not summarize as completed'
args.output.mkdir(exist_ok=False)
cases = sorted({r['case'] for r in rows if r['kind'] == 'timing'})
summary, phases = [], []
for case in cases:
    times = [r for r in rows if r['kind'] == 'timing' and r['case'] == case]
    checks = next(r for r in rows if r['kind'] == 'training_checks' and r['case'] == case)
    c = next(r for r in rows if r['kind'] == 'coefficient_checks' and r['case'] == case)
    variants = {v: {r['round']: r['wall_ms']/r['steps'] for r in times if r['variant'] == v}
                for v in ['keeper', 'free_graph', 'free_relations']}
    keeper = variants['keeper']
    item = dict(case=case, numeric_checks_pass=checks['pass_all'],
                keeper_oracle_checks_pass=all(c['pass_all'] for c in checks['checks'] if c['variant'] != 'source'),
                source_checks_pass=all(c['pass_all'] for c in checks['checks'] if c['variant'] == 'source'),
                keeper_ms=st.median(keeper.values()),
                keeper_min_ms=min(keeper.values()), keeper_max_ms=max(keeper.values()),
                rounds=len(keeper), n_min=min(s['n'] for s in c['steps']),
                n_max=max(s['n'] for s in c['steps']),
                mean_S_nnz=st.mean(s['S_nnz'] for s in c['steps']),
                mean_C_nnz=st.mean(s['C_nnz'] for s in c['steps']),
                mean_csr_bytes=st.mean(s['csr_bytes'] for s in c['steps']))
    for v in ['free_graph', 'free_relations']:
        ratios = [keeper[k]/variants[v][k] for k in keeper]
        ratio = st.median(ratios)
        fraction = 1-1/ratio
        item[v+'_ms'] = st.median(variants[v].values())
        item[v+'_ratio'] = ratio
        item[v+'_paired_ratios'] = ratios
        item[v+'_removable_fraction'] = fraction
        item[v+'_ideal_2x_stage_full_ratio'] = 1/(1-fraction/2)
        item[v+'_meets_1p10'] = bool(item['keeper_oracle_checks_pass'] and ratio >= 1.10)
    for r in rows:
        if r['kind'] == 'phases' and r['case'] == case:
            if r['variant'] == 'keeper':
                item['instrumented_overhead_ratio'] = r['wall_ms']/r['steps']/item['keeper_ms']
            for name, d in r['phases'].items():
                phases.append(dict(case=case, variant=r['variant'], phase=name,
                    host_ms_per_step=d['host_ms']/r['steps'],
                    timeline_ms_per_step=d['timeline_ms']/r['steps'],
                    fraction_instrumented_wall=d['timeline_ms']/r['wall_ms'],
                    calls_per_step=d['calls']/r['steps']))
    summary.append(item)
result = dict(cases=summary, all_numeric_checks_pass=all(r['numeric_checks_pass'] for r in summary),
              note='Free-preparation oracles, not implementation speedups. Three paired rounds; no cross-session CI.')
(args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
for name, data in [('bounds.csv', summary), ('phases.csv', phases)]:
    with (args.output/name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)
lines = ['| 训练窗口 | keeper ms/步 | 免费建图对照 | 免费全部关系对照 | S 非零数/步 | keeper/oracle 核对 | 作者路径核对 |',
         '|---|---:|---:|---:|---:|---|---|']
for r in summary:
    lines.append(f"| {r['case']} | {r['keeper_ms']:.3f} | {r['free_graph_ratio']:.3f}× | {r['free_relations_ratio']:.3f}× | {r['mean_S_nnz']:.1f} | {'通过' if r['keeper_oracle_checks_pass'] else '失败'} | {'通过' if r['source_checks_pass'] else '失败'} |")
(args.output/'table.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(cases=len(cases), numeric_checks_pass=result['all_numeric_checks_pass'])))
