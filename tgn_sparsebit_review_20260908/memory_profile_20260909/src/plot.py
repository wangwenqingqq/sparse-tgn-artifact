from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'analysis'
s=json.loads((OUT/'summary.json').read_text());f=json.loads((OUT/'findings.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig=plt.figure(figsize=(13,12),layout='constrained');gs=fig.add_gridspec(3,1,height_ratios=[1.65,1.1,1.35])
ax=fig.add_subplot(gs[0]);cases=s['cases'];y=np.arange(len(cases))
labels=[r['case'].replace('wikipedia','Wiki').replace('college','College').replace('_',' ') for r in cases]
vals=np.array([r['speedup']['median'] for r in cases]);ci=np.array([r['speedup']['bootstrap_95'] for r in cases])
ax.errorbar(vals,y,xerr=np.stack([vals-ci[:,0],ci[:,1]-vals]),fmt='o',color='#176b87',capsize=3,lw=1.5)
ax.axvline(1,color='#777777',ls='--',lw=1);ax.set_yticks(y,labels);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
ax.set_xlim(.975,1.26);ax.set_xlabel('Direct / batch-store time (higher is faster)')
ax.set_title('A. Whole training steps: 1.104x geometric-mean speedup\n9 paired rounds per window; 8 steps per round; intervals show descriptive 95% bootstrap',loc='left',fontweight='bold')
for yy,v in zip(y,vals):ax.text(1.248,yy,f'{v:.3f}x',ha='right',va='center',fontsize=9)
ax=fig.add_subplot(gs[1]);groups=[('Memory read',['memory_read'],'#4395a8'),('Memory update',['memory_update'],'#e39950'),('Backward',['backward'],'#635a9c'),('Decoder forward',['source_decoder'],'#91ad69'),('Other',None,'#b3b8bd')]
left=np.zeros(2);variants=['direct','batch_store']
for label,keys,color in groups:
 vals=[]
 for v in variants:
  m=f['phases'][v]['macro_ms'];vals.append(sum(m[k] for k in keys) if keys else sum(m.values())-sum(m[k] for k in ['memory_read','memory_update','backward','source_decoder']))
 ax.barh([0,1],vals,left=left,color=color,label=label,height=.5)
 for yy,v,l in zip([0,1],vals,left):
  if v>2:ax.text(l+v/2,yy,f'{v:.2f}',ha='center',va='center',color='white' if label=='Backward' else '#222222',fontsize=9)
 left+=np.array(vals)
ax.set_yticks([0,1],['Direct baseline','Batch-store probe']);ax.invert_yaxis();ax.set_xlim(0,34);ax.set_xlabel('Mean ms/step across 12 windows (sum of top-level phase timelines)')
ax.set_title('B. Separate phase replay: memory write submissions are removable\nDetailed CUDA-event instrumentation adds overhead; these are not the timings in panel A',loc='left',fontweight='bold')
ax.legend(ncol=5,loc='lower center',bbox_to_anchor=(.5,-.5),frameon=False,fontsize=9)
ax=fig.add_subplot(gs[2]);rows=[t for t in f['backward'] if t['case_variant'].endswith('_direct')];left=np.zeros(4)
for name,label,color in [('gnn_embedding','GNN','#4395a8'),('source_decoder','Decoder','#635a9c'),('memory_read','Memory read','#e39950'),('loss','Loss','#b3b8bd')]:
 vals=[100*r['gpu'].get(name,{}).get('gpu_us',0)/sum(x['gpu_us'] for x in r['gpu'].values()) for r in rows]
 ax.barh(np.arange(4),vals,left=left,color=color,label=label,height=.55)
 for yy,v,l in zip(range(4),vals,left):
  if v>5:ax.text(l+v/2,yy,f'{v:.1f}%',ha='center',va='center',fontsize=9,color='white' if name=='source_decoder' else '#222222')
 left+=np.array(vals)
ax.set_yticks(range(4),[r['case_variant'].replace('_direct','').replace('wikipedia','Wiki').replace('college','College').replace('_',' ') for r in rows]);ax.invert_yaxis();ax.set_xlim(0,100)
ax.set_xlabel('Share of attributed backward CUDA activity duration (%)')
ax.set_title('C. Backward attribution: GNN + decoder account for 81-85% of CUDA work\n4 single-step traces; activity durations exclude CPU gaps; memory-update recompute has no loss-backward nodes',loc='left',fontweight='bold')
ax.legend(ncol=4,loc='lower center',bbox_to_anchor=(.5,-.4),frameon=False)
fig.savefig(OUT/'bottleneck.png',dpi=180);fig.savefig(OUT/'bottleneck.svg');plt.close(fig)
