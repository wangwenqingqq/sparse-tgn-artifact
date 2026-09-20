"""Keep the original state computation, suppress only its autograd recording."""
from pathlib import Path
import sys
import types
PREV=Path('/home/data/wangxuran/tncn_memory_profile_20260909')
sys.path.insert(0,str(PREV/'src'))
import instrument as i
p,torch,d=i.p,i.torch,i.d
VARIANTS=['direct','direct_nograd','batch_store','batch_nograd']


def install_nograd(tr):
    memory=tr.model['memory']
    original=memory._update_memory
    def update(self,n_id):
        with torch.no_grad():
            return original(n_id)
    memory._update_memory=types.MethodType(update,memory)


def prepare(tr,variant,mode='none'):
    assert variant in VARIANTS
    clock=i.prepare(tr,'batch_store' if variant.startswith('batch') else 'direct',mode)
    if variant.endswith('nograd'):install_nograd(tr)
    return clock


def inspect_state_graph(tr):
    """Diagnostic observer only; never installed in performance measurements."""
    memory=tr.model['memory'];original=memory._update_memory;rows=[]
    def update(self,n_id):
        before=torch.is_grad_enabled()
        result=original(n_id)
        root=self.memory.grad_fn;pending=[] if root is None else [root];seen=set();types_count={}
        while pending:
            fn=pending.pop()
            if fn in seen:continue
            seen.add(fn);name=type(fn).__name__;types_count[name]=types_count.get(name,0)+1
            pending.extend(child for child,_ in fn.next_functions if child is not None)
        rows.append(dict(outer_grad_enabled=before,requires_grad=self.memory.requires_grad,
            grad_fn=type(root).__name__ if root is not None else None,
            reachable_nodes=len(seen),node_types=types_count))
        return result
    memory._update_memory=types.MethodType(update,memory)
    return rows
