import json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
assert not torch.cuda.is_initialized()
meta=json.loads((ROOT/'output/capture_v1/artifacts/capture.json').read_text());rows=[]
for port in meta['ports']:
    raw=torch.load(ROOT/'output/capture_v1/artifacts'/port['input_file'],map_location='cpu',weights_only=False)
    empty=~(raw['nbr'].numpy()>=0).any(axis=1)
    expected=raw['source_output'].numpy()
    actual=torch.load(ROOT/'output/qualify_v1/artifacts'/(port['label']+'_candidate.pt'),map_location='cpu',weights_only=False)['output'].numpy()
    diff=np.abs(actual.astype('float64')-expected.astype('float64'));bad=diff>2e-4+2e-4*np.abs(expected)
    row=dict(label=port['label'],source_packed=port['source_packed'],empty_queries=int(empty.sum()),queries=len(empty),failing_output_rows=int(bad.any(axis=1).sum()),failing_output_rows_empty=int(bad[empty].any(axis=1).sum()),failing_output_rows_nonempty=int(bad[~empty].any(axis=1).sum()),max_abs_empty=float(diff[empty].max(initial=0)),max_abs_nonempty=float(diff[~empty].max(initial=0)),retained_projection_rows=int(((raw['nbr'].numpy()>=0)|empty[:,None]).sum()),original_rows=raw['nbr'].numel())
    rows.append(row);print(json.dumps(row),flush=True)
(ROOT/'analysis/empty_queries.json').write_text(json.dumps(dict(cases=rows),indent=2)+'\n')
