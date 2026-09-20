from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];s=json.loads((ROOT/'analysis/summary.json').read_text())
rows=[x for x in s['rows'] if not x['port']['source_packed']]
fig,ax=plt.subplots(figsize=(10,4.8));y=np.arange(len(rows));ratio=np.array([x['ratio'] for x in rows]);lo=np.array([x['ratio_ci95'][0] for x in rows]);hi=np.array([x['ratio_ci95'][1] for x in rows])
labels=[f"bs={x['port']['case']['bs']}, L={x['port']['case']['layers']} / {x['port']['module']}" for x in rows]
ax.errorbar(ratio,y,xerr=[ratio-lo,hi-ratio],fmt='o',color='#285c86',capsize=4);ax.axvline(1,color='#777',ls='--',label='Same time');ax.axvline(1.1,color='#be7c32',ls=':',label='Local 1.10 target')
ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel('Source / candidate paired time ratio (higher is faster)');ax.set_title('FlashTGN W_kv: preserve empty-query semantics, skip unused projections\nFixed real inputs; complete local forward + backward; GPU6')
for yy,rr in zip(y,ratio):ax.annotate(f'{rr:.3f}×',(rr,yy),xytext=(7,7),textcoords='offset points',fontsize=9)
ax.legend(loc='best',frameon=False);ax.grid(axis='x',alpha=.15);fig.tight_layout();fig.text(.02,-.03,'Six source-dense cases shown. Six source-packed identity controls are reported separately. No whole-training speedup claim.',fontsize=9)
fig.savefig(ROOT/'analysis/wkv_probe.png',dpi=180,bbox_inches='tight');fig.savefig(ROOT/'analysis/wkv_probe.svg',bbox_inches='tight');plt.close(fig)
