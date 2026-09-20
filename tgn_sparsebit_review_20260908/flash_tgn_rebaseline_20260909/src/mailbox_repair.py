"""Isolated explicit last-concatenated-position mailbox semantics, not a speedup."""
import ast
import copy
import types
import core as c
from flash_tgn import state as sm


def install(tr):
    tree=ast.parse(c.Path(sm.__file__).read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TemporalState')
    original=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='store_raw_messages')
    fn=copy.deepcopy(original);body=fn.body[-1].body
    assert [ast.unparse(x) for x in body[-2:]]==['self.mail_data[nodes] = mails','self.mail_ts[nodes] = times']
    replacement=ast.parse('''
last = torch.full((len(self.mem_data),), -1, device=nodes.device, dtype=torch.long)
positions = torch.arange(len(nodes), device=nodes.device, dtype=torch.long)
last.scatter_reduce_(0, nodes.long(), positions, reduce='amax', include_self=True)
unique = torch.nonzero(last >= 0, as_tuple=False).flatten()
winners = last[unique]
self.mail_data[unique] = mails[winners]
self.mail_ts[unique] = times[winners]
''').body
    assert ast.dump(ast.Module(body=body[:-2],type_ignores=[]))==ast.dump(ast.Module(body=original.body[-1].body[:-2],type_ignores=[]))
    body[-2:]=replacement
    ns=dict(vars(sm));exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(sm.__file__)+'[explicit-last-write]','exec'),ns)
    tr.state.store_raw_messages=types.MethodType(ns['store_raw_messages'],tr.state)
