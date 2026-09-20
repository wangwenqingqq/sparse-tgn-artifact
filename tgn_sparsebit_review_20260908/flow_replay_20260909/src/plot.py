import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);a=p.parse_args()
summary=json.loads((a.analysis/'summary.json').read_text())
colors={'copy':'#7f7f7f','coarse':'#d89021','mlp1':'#286ca8','fm1':'#c94853','fm2':'#914cb7','fm4':'#328779'}
labels={'copy':'Keep memory','coarse':'One GRU update','mlp1':'MLP','fm1':'FM / 1 step','fm2':'FM / 2 steps','fm4':'FM / 4 steps'}
fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
for ax,(name,chunk) in zip(axes.ravel(),[(n,k) for n in ['wikipedia','college'] for k in [128,512]]):
    rows=[r for r in summary['rows'] if r['dataset']==name and r['chunk']==chunk and r['family']!='exact']
    for family,color in colors.items():
        rr=[r for r in rows if r['family']==family]
        ax.scatter([r['speed'] for r in rr],[-100*min(r['ap_delta'],r['auc_delta']) for r in rr],
                   color=color,label=labels[family],s=45,alpha=.85,marker='s' if family in ['copy','coarse'] else 'o')
    ax.axhline(1,color='#666666',ls='--',lw=1);ax.axvline(1.1,color='#666666',ls='--',lw=1)
    ax.set_title(f'{name.capitalize()} / {chunk} historical events')
    ax.set_xlabel('Full replay speedup vs optimized teacher');ax.set_ylabel('Larger AP / AUC loss (percentage points)')
    ax.grid(alpha=.18)
axes[0,0].legend(fontsize=8)
fig.suptitle('Flow matching and simpler historical replay approximations\nEach learned-method point is one seed; lower and farther right is better',fontsize=12)
fig.savefig(a.analysis/'speed_quality.png',dpi=180);fig.savefig(a.analysis/'speed_quality.svg');plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for ax,(name,chunk) in zip(axes.ravel(),[(n,k) for n in ['wikipedia','college'] for k in [128,512]]):
    raw=np.load(a.input/f'{name}_k{chunk}_queries.npz')
    for family in ['copy','coarse','mlp1','fm2']:
        keys=[key for key in raw.files if key.endswith('_drift') and key.removesuffix('_drift').split('_')[0]==family]
        values=np.stack([raw[k] for k in keys]);x=(np.arange(values.shape[1])+1)*chunk
        ax.plot(x,np.median(values,axis=0),color=colors[family],label=labels[family])
        if len(values)>1:ax.fill_between(x,values.min(0),values.max(0),color=colors[family],alpha=.14)
    ax.set_title(f'{name.capitalize()} / block {chunk}');ax.set_xlabel('Replayed historical events')
    ax.set_ylabel('Memory error / teacher RMS');ax.grid(alpha=.18)
axes[0,0].legend(fontsize=8)
fig.suptitle('Closed-loop memory drift: median and seed range',fontsize=12)
fig.savefig(a.analysis/'memory_drift.png',dpi=180);fig.savefig(a.analysis/'memory_drift.svg')
