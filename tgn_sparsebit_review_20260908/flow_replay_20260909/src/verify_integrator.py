"""Check the actual solver against analytic constant and affine vector fields."""
import ast
import json
from pathlib import Path
import sys
import torch

root=Path(__file__).resolve().parents[1]
tree=ast.parse((root/'src/experiment.py').read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='integrate')
ns={}
exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual_integrate','exec'),ns)
condition=torch.zeros((3,5),dtype=torch.float64)
rows=[]
for coefficient in [0.,.5,-.5]:
    for steps in [1,2,4]:
        calls=[]
        def field(cond,z,t):
            calls.append(float(t[0,0]));return 2.+coefficient*z
        result=ns['integrate'](field,condition,'fm',steps)
        expected=2. if coefficient==0 else 2./coefficient*((1+coefficient/steps)**steps-1)
        assert torch.allclose(result,torch.full_like(result,expected),atol=1e-12,rtol=1e-12)
        assert calls==[j/steps for j in range(steps)]
        rows.append(dict(coefficient=coefficient,steps=steps,evaluations=len(calls),max_error=float((result-expected).abs().max())))
target=Path(sys.argv[1]);assert not target.exists()
target.write_text(json.dumps(dict(pass_all=True,cases=rows),indent=2)+'\n')
print('All 9 CPU analytic-vector-field solver checks passed.')
