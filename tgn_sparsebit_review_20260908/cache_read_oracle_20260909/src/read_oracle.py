"""Oracle for constant message-store reads; all learned computations stay literal."""
import ast
import copy
import importlib
from pathlib import Path
import sys
import types
PREV=Path('/home/data/wangxuran/tncn_memory_profile_20260909')
sys.path.insert(0,str(PREV/'src'))
import instrument as i
p,torch,d=i.p,i.torch,i.d
FIELDS=['src','dst','t','raw_msg']


def source_method():
    module=importlib.import_module(p.TGNMemory.__module__)
    tree=ast.parse(Path(module.__file__).read_text())
    cl=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='TGNMemory')
    fn=copy.deepcopy(next(x for x in cl.body if isinstance(x,ast.FunctionDef) and x.name=='_compute_msg'))
    split=next(j+1 for j,x in enumerate(fn.body) if isinstance(x,ast.Assign) and ast.unparse(x.targets[0])=='raw_msg')
    assert split==6 and ast.unparse(fn.body[split-1].value)=='torch.cat(raw_msg, dim=0)'
    return module,fn,split


def bind(tr,fn,module,extras):
    ns=dict(vars(module),**extras)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(Path(module.__file__))+'[cache-read-oracle]','exec'),ns)
    tr.model['memory']._compute_msg=types.MethodType(ns[fn.name],tr.model['memory'])


class Tape:
    def __init__(self,records=None,verify=False):
        self.records=[] if records is None else records
        self.cursor=0;self.verify=verify

    def record(self,memory,n_id,msg_store,src,dst,t,raw_msg):
        values=dict(zip(FIELDS,[src,dst,t,raw_msg]))
        assert all(not x.requires_grad and x.is_contiguous() for x in values.values())
        direction='s' if msg_store is memory.msg_s_store else 'd'
        assert msg_store is getattr(memory,'msg_'+direction+'_store')
        self.records.append(dict(n_id=n_id.detach().clone(),direction=direction,**values))

    def begin(self,offset):
        self.cursor=offset*4

    def next(self,memory,n_id,msg_store):
        row=self.records[self.cursor];self.cursor+=1
        if self.verify:
            assert msg_store is getattr(memory,'msg_'+row['direction']+'_store')
            assert torch.equal(n_id,row['n_id'])
        return tuple(row[k] for k in FIELDS)

    def cpu_records(self):
        return [{k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in r.items()} for r in self.records]

    def stats(self):
        unique={};logical=0
        for r in self.records:
            for value in r.values():
                if torch.is_tensor(value):
                    logical+=value.numel()*value.element_size();s=value.untyped_storage()
                    if s.nbytes():unique[s.data_ptr()]=s.nbytes()
        return dict(records=len(self.records),logical_bytes=logical,unique_storage_bytes=sum(unique.values()),
            message_rows=sum(r['src'].numel() for r in self.records))


def install_capture(tr,tape):
    module,fn,split=source_method()
    fn.body[split:split]=ast.parse('_record(self,n_id,msg_store,src,dst,t,raw_msg)').body
    bind(tr,fn,module,{'_record':tape.record})


def install_oracle(tr,tape,clock):
    module,fn,split=source_method();tail=fn.body[split:]
    prefix=ast.parse('src,dst,t,raw_msg=_tape.next(self,n_id,msg_store)').body[0]
    if clock.mode=='none':fn.body=[prefix]+tail
    else:
        def phase(stmt,name):
            return ast.With(items=[ast.withitem(context_expr=ast.parse(f"_clock.phase('{name}')",mode='eval').body)],body=[stmt])
        body=[phase(prefix,'message.cached_read')]
        for stmt in tail:
            if isinstance(stmt,ast.Return):body.append(stmt);continue
            target=ast.unparse(stmt.targets[0]) if isinstance(stmt,ast.Assign) else ''
            body.append(phase(stmt,'message.time' if target in ['t_rel','t_enc'] else 'message.compose'))
        body.insert(-1,ast.parse("_clock.count('message_rows',src.numel())").body[0]);fn.body=body
    bind(tr,fn,module,{'_tape':tape,'_clock':clock})


def prepare(tr,variant,tape=None,mode='none'):
    clock=i.prepare(tr,'batch_store',mode)
    if variant=='capture':install_capture(tr,tape)
    elif variant=='free_cache_reads':install_oracle(tr,tape,clock)
    else:assert variant=='batch_store'
    return clock


def step(tr,index,start,tape=None,check=False):
    if tape is not None:tape.begin(index-start)
    result=i.b.step(tr,index,'direct',check=check)
    if tape is not None:assert tape.cursor==(index-start+1)*4
    return result
