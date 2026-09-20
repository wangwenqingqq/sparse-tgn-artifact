import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parents[1]
r=json.loads((root/'analysis/summary.json').read_text())
fig,ax=plt.subplots(figsize=(10,6.5))
styles=[('direct',-.13,'#177E89','Direct CSR (recommended)'),
        ('fused_bounds',.13,'#8B8E98','Direct CSR + boundary read fusion')]
for variant,offset,color,label in styles:
    values=[c['ratios']['compatible/'+variant] for c in r['cases']]
    mids=np.array([v['median'] for v in values]);lo=np.array([v['bootstrap_95'][0] for v in values]);hi=np.array([v['bootstrap_95'][1] for v in values])
    ax.errorbar(mids,np.arange(12)+offset,xerr=[mids-lo,hi-mids],fmt='o',markersize=5,
                capsize=2,elinewidth=1.3,color=color,label=label)
ax.axvline(1,color='#A84E40',linestyle='--',linewidth=1.1)
ax.set_yticks(range(12),[c['case'].replace('college','CollegeMsg').replace('wikipedia','Wikipedia').replace('_b',' / B').replace('_s',' / step ') for c in r['cases']])
ax.invert_yaxis();ax.set_xlabel('Whole-step speedup over dense compatibility (higher is faster)')
ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.61,.94),ncol=2,frameon=False,fontsize=8.5)
fig.suptitle('Direct CSR reduces conversion cost while preserving source results',fontsize=13,y=.98)
fig.text(.5,.01,'Both variants: 96/96 source-identical steps. Intervals describe nine paired rounds within this GPU3 session.',ha='center',fontsize=8,color='#555555')
fig.tight_layout(rect=[0,.035,1,.89])
fig.savefig(root/'analysis/speedup.png',dpi=180);fig.savefig(root/'analysis/speedup.svg')
