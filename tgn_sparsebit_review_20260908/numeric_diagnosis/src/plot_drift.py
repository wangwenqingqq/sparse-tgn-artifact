import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
rows = [json.loads(s) for s in (root/'evidence/witnesses_v1/stdout.jsonl').read_text().splitlines()]
out = root/'analysis'; out.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7), sharey=True)
styles = [('source', 'Unmodified source', '#C4563B', '-'),
          ('source_anchor_mlp', 'Align predictor weights only', '#999999', '--'),
          ('source_anchor_frequency', 'Align time frequency only', '#187C80', '-')]
for ax, name, title in zip(axes, ['wikipedia', 'college'], ['Wikipedia / B200', 'CollegeMsg / B200']):
    case = name+'_b200_s20'
    for mode, label, color, ls in styles:
        row = next(r for r in rows if r['kind']=='witness_trajectory' and r['case']==case and r['variant']==mode)
        y = np.array([c['metrics']['embedding']['max_abs'] for c in row['checks']])
        ax.plot(range(20, 28), np.where(y>0, y, np.nan), ls=ls, color=color,
                marker='o', markersize=4, linewidth=2 if mode!='source_anchor_mlp' else 1.5, label=label)
    probe = next(r for r in rows if r['kind']=='transplant' and r['case']==case)
    ax.scatter([probe['probe_index']], [probe['metrics']['embedding']['max_abs']],
               marker='s', s=80, facecolors='none', edgecolors='#212529', linewidths=1.5,
               zorder=5, label='Change one frequency entry only')
    ax.set_title(title, fontsize=12)
    ax.set_yscale('log'); ax.set_xticks(range(20, 28)); ax.set_xlabel('Training-step index')
    ax.grid(axis='y', alpha=.22); ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('Max embedding difference vs keeper')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(.5, .035), frameon=False, fontsize=9)
fig.suptitle('A tiny time-frequency update can drive later divergence', fontsize=14, y=.98)
fig.text(.5, .006, 'Step 20 embeddings are identical. Parameter alignment and transplantation are causal probes, not training methods.',
         ha='center', fontsize=8, color='#555555')
fig.tight_layout(rect=[0, .18, 1, .93])
fig.savefig(out/'training_drift.png', dpi=180)
fig.savefig(out/'training_drift.svg')
