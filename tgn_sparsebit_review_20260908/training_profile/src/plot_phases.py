import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
rows = [json.loads(s) for s in (root/'remote_evidence/output/full_v5/stdout.jsonl').read_text().splitlines()]
configs = ['wikipedia_b32', 'college_b32', 'wikipedia_b200', 'college_b200']
groups = {
    'Memory read + update': ['memory_read', 'memory_update'],
    'Backward': ['backward'],
    'Sampling + insert': ['negative_and_neighbor_sampling', 'neighbor_insert'],
    'GNN + decoder MLP': ['gnn_embedding', 'decoder_mlp'],
    'Graph + Q/S + coefficients': ['graph_build', 'relation_Q_S_coefficients'],
    'Seven-channel aggregation': ['seven_channel_aggregation'],
}
values = []
for config in configs:
    cases = [r for r in rows if r['kind'] == 'phases' and r['variant'] == 'keeper' and r['case'].startswith(config+'_')]
    v = [statistics.mean(100*sum(r['phases'][k]['timeline_ms'] for k in keys)/r['wall_ms'] for r in cases)
         for keys in groups.values()]
    v.append(100-sum(v))
    values.append(v)
values = np.array(values)
labels = list(groups)+['Other + unassigned gaps']
colors = ['#277DA8', '#7957A2', '#69A8A8', '#BAC7D6', '#E88B38', '#EDC778', '#E7E9EC']
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.spines.left': False, 'axes.spines.bottom': False})
fig, ax = plt.subplots(figsize=(11, 4.6))
left = np.zeros(4)
for i, (label, color) in enumerate(zip(labels, colors)):
    ax.barh(np.arange(4), values[:, i], left=left, height=.58, color=color, label=label)
    if i < 2:
        for j in range(4):
            ax.text(left[j]+values[j, i]/2, j, f'{values[j,i]:.1f}%', ha='center', va='center', color='white', weight='bold')
    left += values[:, i]
ax.set_yticks(range(4), ['Wikipedia / B32', 'CollegeMsg / B32', 'Wikipedia / B200', 'CollegeMsg / B200'])
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel('Share of instrumented training-step wall time (%)')
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_axisbelow(True)
ax.grid(axis='x', alpha=.18)
ax.tick_params(axis='y', length=0, pad=10)
fig.suptitle('Memory and backward dominate the current TNCN keeper', x=.02, ha='left', fontsize=16, weight='bold')
ax.set_title('Mean of three fixed windows per configuration; CUDA event timeline includes host launch gaps.', loc='left', fontsize=9, color='#555555', pad=15)
ax.legend(ncol=3, loc='upper left', bbox_to_anchor=(-.01, -.22), frameon=False, fontsize=9)
fig.subplots_adjust(left=.17, right=.98, top=.77, bottom=.25)
fig.savefig(root/'analysis/training_phases.png', dpi=170, bbox_inches='tight')
fig.savefig(root/'analysis/training_phases.svg', bbox_inches='tight')
(root/'analysis/grouped_phases.json').write_text(json.dumps(dict(configs=configs, labels=labels, percentages=values.tolist()), indent=2)+'\n')
