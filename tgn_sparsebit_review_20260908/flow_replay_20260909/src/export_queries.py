"""Export small raw query arrays for local metric and uncertainty reanalysis."""
import argparse
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);args=p.parse_args()
torch.set_num_threads(1)
for name in ['wikipedia','college']:
    for chunk in [128,512]:
        arrays={}
        prefix=f'{name}_k{chunk}_'
        for path in sorted(args.input.glob(prefix+'*_rollout.pt')):
            variant=path.name.removeprefix(prefix).removesuffix('_rollout.pt')
            data=torch.load(path,map_location='cpu',weights_only=False)
            arrays[variant]=data['logits'].numpy()
            labels=data['labels'].numpy()
            if 'labels' in arrays:
                assert np.array_equal(labels,arrays['labels'])
            arrays['labels']=labels
            arrays['block']=np.concatenate([np.full(len(row['labels']),row['block'],dtype=np.int64)
                                            for row in data['rows'] if row['labels'] is not None])
            arrays[variant+'_drift']=np.array([row['drift']['nrmse'] for row in data['rows']])
        np.savez(args.input/f'{name}_k{chunk}_queries.npz',**arrays)
print('Exported 4 small query archives.')
