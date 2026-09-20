import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'analysis'
s=json.loads((OUT/'summary.json').read_text());rows=s['cases'];n=len(rows)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,(ax,bx)=plt.subplots(2,1,figsize=(13,11),gridspec_kw={'height_ratios':[2,1.6]},layout='constrained')
y=np.arange(n);v=np.array([r['speedup']['median'] for r in rows]);ci=np.array([r['speedup']['bootstrap_95'] for r in rows])
labels=[r['display'].replace('wikipedia','Wiki').replace('college','College') for r in rows]
ax.errorbar(v,y,xerr=np.stack([v-ci[:,0],ci[:,1]-v]),fmt='o',color='#176b87',capsize=3,lw=1.4)
ax.axvline(1,color='#888888',ls=':',lw=1);ax.axvline(1.10,color='#b65d30',ls='--',lw=1.5,label='Predeclared 1.10x target')
ax.set_yticks(y,labels);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
ax.set_xlabel('Batch-store baseline / paid event-index prototype time')
ax.set_title(f"A. Measured paid prototype: {s['speedup']['geomean']:.3f}x geometric mean\n9 paired rounds per window; 8 complete steps per interval; descriptive 95% bootstrap",loc='left',fontweight='bold')
ax.legend(frameon=False,loc='best')
paid=np.array([r['prefix']['batch_store']['timeline_ms_per_step'] for r in rows]);free=np.array([r['prefix']['event_index']['timeline_ms_per_step'] for r in rows])
bx.barh(y-.16,paid,height=.3,label='Paid lookup / unpack / cat',color='#4a899d');bx.barh(y+.16,free,height=.3,label='CPU event indices / transfer / gathers',color='#e2aa6d')
bx.set_yticks(y,labels);bx.invert_yaxis();bx.set_xlabel('Prefix ms/step in separate instrumented replay')
bx.set_title('B. Real read costs remain in both arms\nTime encoding, live-memory gathers, GRU, cache writes, backward and Adam remain paid',loc='left',fontweight='bold')
bx.legend(frameon=False,loc='lower right');bx.grid(axis='x',alpha=.2)
fig.savefig(OUT/'paid_event_reads.png',dpi=180);fig.savefig(OUT/'paid_event_reads.svg');plt.close(fig)
