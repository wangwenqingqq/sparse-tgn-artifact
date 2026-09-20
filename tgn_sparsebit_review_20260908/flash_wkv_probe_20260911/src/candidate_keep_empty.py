import ast
import copy
from types import SimpleNamespace
import common as a
p=a.p


def build():
    tree=ast.parse(a.c.Path(a.lm.__file__).read_text())
    source=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='temporal_attention')
    fn=copy.deepcopy(source)
    indexes=[i for i,n in enumerate(fn.body) if ast.unparse(n)=='z_proj = layer.W_kv(z_flat)'];assert len(indexes)==1;idx=indexes[0]
    replacement=ast.parse('''
projection_mask = valid_mask | (~valid_mask.view(q, k).any(dim=1)).repeat_interleave(k)
z_valid = z_flat[projection_mask]
projected_valid = layer.W_kv(z_valid)
z_proj = torch.zeros((z_flat.shape[0], 2 * layer.d_embed), device=z_flat.device, dtype=z_flat.dtype)
z_proj[projection_mask] = projected_valid
''').body
    fn.body[idx:idx+1]=replacement
    restored=copy.deepcopy(fn);restored.body[idx:idx+len(replacement)]=[source.body[idx]];assert ast.dump(restored)==ast.dump(source)
    ns=dict(vars(a.lm));exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(a.lm.__file__)+'[valid-only-projection]','exec'),ns)
    return ns['temporal_attention']


CANDIDATE=build()


def prepare(raw):
    data=a.gpu({k:raw[k] for k in ['q','z','nbr','weight','bias','upstream']})
    for name in ['q','z','weight','bias']:data[name].requires_grad_(True)
    layer=SimpleNamespace(W_kv=lambda x:p.nn.functional.linear(x,data['weight'],data['bias']),
        d_embed=data['q'].shape[1],num_heads=raw['heads'],d_k=data['q'].shape[1]//raw['heads'],training=True,attn_act=p.nn.LeakyReLU(.2))
    policy=a.lm.OperatorPolicy(**raw['policy'])
    return data,layer,policy


def run(bundle,arm):
    data,layer,policy=bundle;fn=a.lm.temporal_attention if arm=='source' else CANDIDATE
    output=fn(layer,data['q'],data['z'],data['nbr'],policy)
    gradients=p.autograd.grad(output,[data[k] for k in ['q','z','weight','bias']],grad_outputs=data['upstream'])
    return dict(output=output,**dict(zip(['grad_q','grad_z','grad_weight','grad_bias'],gradients)))
