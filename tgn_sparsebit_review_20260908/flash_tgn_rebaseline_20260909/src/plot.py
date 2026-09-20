from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
s=json.loads((ROOT/'analysis/summary.json').read_text())
labels=sorted(s['timing'],key=lambda x:(s['timing'][x]['case']['bs'],s['timing'][x]['case']['k'],s['timing'][x]['case']['layers']))
names=[f"bs={s['timing'][x]['case']['bs']}, K={s['timing'][x]['case']['k']}, L={s['timing'][x]['case']['layers']}" for x in labels]
fig,(a,b)=plt.subplots(1,2,figsize=(14,5),gridspec_kw=dict(width_ratios=[1,1.6]))
ys=np.arange(len(labels));med=np.array([s['timing'][x]['epoch_ms']['median']/1000 for x in labels]);lo=np.array([s['timing'][x]['epoch_ms']['p10']/1000 for x in labels]);hi=np.array([s['timing'][x]['epoch_ms']['p90']/1000 for x in labels])
a.barh(ys,med,color='#275a90',height=.65);a.errorbar(med,ys,xerr=[med-lo,hi-med],fmt='none',ecolor='#16202b',capsize=3)
a.set_yticks(ys,names);a.invert_yaxis();a.set_xlabel('Complete training epoch (seconds)');a.set_title('Source timing: median and p10–p90\n9 epochs/configuration, 3 fresh processes')
for y,v,h in zip(ys,med,hi):a.text(h+.1,y,f'{v:.2f}',va='center',fontsize=9)
a.set_xlim(0,max(hi)*1.17);a.grid(axis='x',alpha=.15);a.set_axisbelow(True)
groups=[('Backward',['backward'],'#c26351'),('Embedding',['embedding'],'#e6b36b'),('Memory update',['memory_update'],'#5b8fbd'),('Mailbox write',['mailbox_write'],'#55a58a'),('Optimizer',['optimizer','zero_grad'],'#9078ad'),('Schedule',['outer_schedule_full','schedule_build','cuda.tcsr_temporal_sample'],'#95a65a')]
left=np.zeros(len(labels));used=[]
for title,keys,color in groups:
 values=np.array([sum(s['profiles'][x]['categories_percent'].get(k,0) for k in keys) for x in labels]);b.barh(ys,values,left=left,label=title,color=color,height=.65);left+=values;used+=keys
b.barh(ys,100-left,left=left,label='Other / unscoped',color='#c5ccd2',height=.65)
b.set_yticks(ys,[]);b.invert_yaxis();b.set_xlim(0,100);b.set_xlabel('Share of instrumented epoch (%)');b.set_title('Separate event profile, epoch 2\nMutually exclusive scopes; includes CPU launch gaps')
b.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3,frameon=False,fontsize=9)
fig.suptitle('FlashTGN rebaseline — diagnostic timing only; self-replay gate FAILED',fontsize=13,fontweight='bold',y=1.03)
fig.text(.05,-.05,'Wikipedia full training split; RTX PRO 6000 Blackwell, GPU0. No optimization speedup or accuracy parity is established.',fontsize=10,color='#555')
fig.tight_layout();fig.savefig(ROOT/'analysis/flash_baseline.png',dpi=180,bbox_inches='tight');fig.savefig(ROOT/'analysis/flash_baseline.svg',bbox_inches='tight');plt.close(fig)
