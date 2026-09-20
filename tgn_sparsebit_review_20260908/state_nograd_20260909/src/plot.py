import json
from pathlib import Path
import numpy as np
import statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'analysis'
s=json.loads((OUT/'summary.json').read_text());cases=s['cases'];n=len(cases)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,(ax,bx)=plt.subplots(2,1,figsize=(12,10),gridspec_kw={'height_ratios':[2.5,1]},layout='constrained')
y=np.arange(n)
for pair,label,color,offset in [('direct/direct_nograd','No-grad on direct CSR','#176b87',-.13),('batch_store/batch_nograd','No-grad after batch store','#ca7839',.13)]:
    vals=np.array([r['ratios'][pair]['median'] for r in cases]);ci=np.array([r['ratios'][pair]['bootstrap_95'] for r in cases])
    ax.errorbar(vals,y+offset,xerr=np.stack([vals-ci[:,0],ci[:,1]-vals]),fmt='o',color=color,label=label,capsize=3,lw=1.3,markersize=5)
ax.axvline(1,color='#777777',ls='--',lw=1);ax.set_yticks(y,[r['case'].replace('wikipedia','Wiki').replace('college','College').replace('_',' ') for r in cases]);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
ax.set_xlabel('Baseline / no-grad whole-step time (higher is faster)')
ax.set_title('A. Isolated no-grad effect on complete training steps\n9 paired rounds per window, 8 steps per interval; descriptive 95% bootstrap',loc='left',fontweight='bold')
ax.legend(loc='lower right',frameon=False)
variants=['direct','direct_nograd','batch_store','batch_nograd'];means=[]
for variant in variants:
    means.append(st.mean(r['phases'][variant]['memory_update/state.recompute']['timeline_ms_per_step'] for r in cases))
bars=bx.bar(range(4),means,color=['#8ba7b1','#176b87','#dec3a9','#ca7839'],width=.6)
for bar,value in zip(bars,means):bx.text(bar.get_x()+bar.get_width()/2,value+.015,f'{value:.3f}',ha='center')
bx.set_xticks(range(4),['Direct CSR','Direct + no-grad','Batch store','Batch + no-grad']);bx.set_ylabel('Instrumented ms/step');bx.set_ylim(0,max(means)*1.22)
bx.set_title('B. State recomputation still executes\nSeparate phase replay; arithmetic is retained while autograd recording is disabled',loc='left',fontweight='bold')
fig.savefig(OUT/'nograd_effect.png',dpi=180);fig.savefig(OUT/'nograd_effect.svg');plt.close(fig)
