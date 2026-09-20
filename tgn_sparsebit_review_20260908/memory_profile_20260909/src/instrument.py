import ast
import contextlib
import copy
import importlib
from pathlib import Path
import sys
import time
import types

PREV=Path('/home/data/wangxuran/tncn_compatible_csr_20260908')
sys.path.insert(0,str(PREV/'src'))
import bridge as b
p,torch,d=b.p,b.torch,b.d


class Scopes:
    def __init__(self,mode='none'):
        assert mode in ['none','events','trace']
        self.mode=mode;self.stack=[];self.rows=[];self.stats={}

    def count(self,name,value):
        key='/'.join(self.stack+[name])
        self.stats[key]=self.stats.get(key,0)+int(value)

    @contextlib.contextmanager
    def phase(self,name):
        if self.mode=='none':
            yield;return
        self.stack.append(name);path='/'.join(self.stack)
        if self.mode=='trace':
            try:
                with torch.profiler.record_function('scope:'+path):yield
            finally:self.stack.pop()
        else:
            a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();start=time.perf_counter_ns()
            try:yield
            finally:
                z.record();self.rows.append((path,(time.perf_counter_ns()-start)/1e6,a,z));self.stack.pop()

    def results(self):
        result={}
        for name,host,a,z in self.rows:
            row=result.setdefault(name,dict(calls=0,host_ms=0.,timeline_ms=0.))
            row['calls']+=1;row['host_ms']+=host;row['timeline_ms']+=a.elapsed_time(z)
        return dict(phases=result,stats=self.stats)


def instrument_memory(tr,clock):
    module=importlib.import_module(p.TGNMemory.__module__)
    tree=ast.parse(Path(module.__file__).read_text())
    cl=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='TGNMemory')
    for method in ['_compute_msg','_get_updated_memory','_update_msg_store','_update_memory']:
        fn=copy.deepcopy(next(x for x in cl.body if isinstance(x,ast.FunctionDef) and x.name==method))
        body=[]
        for stmt in fn.body:
            if isinstance(stmt,ast.Return):body.append(stmt);continue
            target=ast.unparse(stmt.targets[0]) if isinstance(stmt,ast.Assign) else ''
            if method=='_compute_msg':
                name='message.lookup' if target=='data' else 'message.unpack'
                if target in ['src','dst','t','raw_msg']:name='message.concat'
                elif target in ['t_rel','t_enc']:name='message.time'
                elif target=='msg':name='message.compose'
            elif method=='_get_updated_memory':
                name='memory.assoc'
                if target.startswith('(msg_s,'):name='memory.source_messages'
                elif target.startswith('(msg_d,'):name='memory.destination_messages'
                elif target in ['idx','msg','t']:name='memory.aggregate_inputs'
                elif target=='aggr':name='memory.last_aggregate'
                elif target=='memory':name='memory.gru'
                elif target in ['dim_size','last_update']:name='memory.last_time'
            elif method=='_update_msg_store':
                name='store.sort' if target=='(n_id, perm)' else 'store.counts'
                if isinstance(stmt,ast.For):name='store.per_node_gathers'
            else:
                name='state.recompute' if target=='(memory, last_update)' else 'state.write'
            wrapped=ast.With(items=[ast.withitem(context_expr=ast.parse(f"_clock.phase('{name}')",mode='eval').body)],body=[stmt])
            body.append(wrapped)
        if method=='_update_msg_store':
            body+=ast.parse("_clock.count('updated_nodes', n_id.numel())").body
        elif method=='_compute_msg':
            body.insert(-1,ast.parse("_clock.count('message_rows', src.numel())").body[0])
        fn.body=body
        ns=dict(vars(module),_clock=clock)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(Path(module.__file__))+'[scopes]','exec'),ns)
        setattr(tr.model['memory'],method,types.MethodType(ns[method],tr.model['memory']))


def install_batch_store(tr,clock=None):
    clock=clock or Scopes()
    def update(self,src,dst,t,raw_msg,msg_store):
        with clock.phase('store.sort'):
            n_id,perm=src.sort()
        with clock.phase('store.counts'):
            n_id,count=n_id.unique_consecutive(return_counts=True)
        with clock.phase('store.metadata'):
            ids=n_id.tolist();sizes=count.tolist()
        with clock.phase('store.batch_gathers'):
            ordered=[a[perm] for a in (src,dst,t,raw_msg)]
        with clock.phase('store.views_and_dictionary'):
            parts=[a.split(sizes) for a in ordered]
            for i,values in zip(ids,zip(*parts)):
                msg_store[i]=values
        clock.count('updated_nodes',n_id.numel())
    tr.model['memory']._update_msg_store=types.MethodType(update,tr.model['memory'])


def prepare(tr,variant='direct',mode='none'):
    clock=Scopes(mode)
    tr.clock=clock
    if mode!='none':instrument_memory(tr,clock)
    if variant=='batch_store':install_batch_store(tr,clock)
    else:assert variant=='direct'
    b.install(tr,'direct',clock=clock)
    return clock


def expanded_check(tr,result):
    checks=dict(result)
    sizes={}
    for direction in ['s','d']:
        store=getattr(tr.model['memory'],'msg_'+direction+'_store')
        records=[store[i] for i in range(tr.n)]
        checks['cache/'+direction+'/lengths']=torch.tensor([r[0].numel() for r in records],dtype=torch.long)
        unique={};logical=0
        for j,field in enumerate(['src','dst','time','message']):
            tensors=[r[j] for r in records]
            checks['cache/'+direction+'/'+field]=torch.cat(tensors,dim=0).detach().cpu()
            for t in tensors:
                logical+=t.numel()*t.element_size()
                storage=t.untyped_storage()
                if storage.nbytes():unique[storage.data_ptr()]=storage.nbytes()
        sizes[direction]=dict(logical_bytes=logical,unique_storage_bytes=sum(unique.values()),unique_storages=len(unique))
    checks['rng_cpu']=torch.get_rng_state().clone()
    checks['rng_cuda']=torch.cuda.get_rng_state().clone()
    return checks,sizes
