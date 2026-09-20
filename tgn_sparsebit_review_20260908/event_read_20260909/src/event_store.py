"""Paid event-ID stores for the immutable-input, qualified training harness."""
import ast
import time
import types
from pathlib import Path
import sys
import numpy as np
ORACLE=Path('/home/data/wangxuran/tncn_cache_read_oracle_20260909')
sys.path.insert(0,str(ORACLE/'src'))
import read_oracle as source
i,p,torch,d=source.i,source.p,source.torch,source.d
PREV=source.PREV
FIELDS=['src','dst','t','raw_msg']


class Store:
    def __init__(self,direction):
        self.direction=direction
        self.nodes=[]


class EventState:
    def __init__(self,tr,start,clock,capture):
        begin=time.perf_counter_ns()
        self.tr=tr;self.start=start;self.clock=clock;self.capture=capture
        self.data=tr.data
        assert all(not x.requires_grad for x in self.data)
        self.cpu=[x.detach().cpu().numpy().copy() for x in self.data]
        self.stores=[Store('s'),Store('d')]
        self.step_index=None;self.records=[];self.seed=None;self.backup=None
        self.setup=dict(cpu_input_copy_ms=(time.perf_counter_ns()-begin)/1e6,
            additional_cpu_input_bytes=sum(x.nbytes for x in self.cpu),
            shared_existing_gpu_input_bytes=sum(x.numel()*x.element_size() for x in self.data))

    def restore(self,memory,backup):
        memory.memory=backup[0].clone();memory.last_update=backup[1].clone()
        if self.backup is not backup:
            begin=time.perf_counter_ns();stop=self.start*self.tr.bs
            s,d,t,msg=self.cpu
            lookup={}
            for eid in range(stop):
                lookup.setdefault((int(s[eid]),int(d[eid]),int(t[eid]),msg[eid].tobytes()),eid)
            seeds=[]
            for direction,old in zip(['s','d'],backup[2:]):
                rows=[old[node] for node in range(self.tr.n)]
                lengths=[r[0].numel() for r in rows]
                fields=[torch.cat([r[j] for r in rows],dim=0).cpu().numpy() for j in range(4)]
                ids=np.empty(sum(lengths),dtype=np.int64)
                for j in range(len(ids)):
                    a,b=int(fields[0][j]),int(fields[1][j])
                    if direction=='d':a,b=b,a
                    ids[j]=lookup[(a,b,int(fields[2][j]),fields[3][j].tobytes())]
                offsets=np.concatenate([np.zeros(1,dtype=np.int64),np.cumsum(lengths,dtype=np.int64)])
                seeds.append([ids[offsets[node]:offsets[node+1]] for node in range(self.tr.n)])
            self.seed=seeds;self.backup=backup
            self.setup['checkpoint_import_ms']=(time.perf_counter_ns()-begin)/1e6
            self.setup['checkpoint_event_limit']=stop
        for store,seed in zip(self.stores,self.seed):store.nodes=seed.copy()
        memory.msg_s_store,memory.msg_d_store=self.stores

    def read(self,memory,n_id,store):
        assert store in self.stores
        with self.clock.phase('message.index_lookup_concat'):
            nodes=n_id.tolist()
            events=np.concatenate([store.nodes[node] for node in nodes])
        with self.clock.phase('message.index_transfer'):
            index=torch.from_numpy(events).to(n_id.device)
        with self.clock.phase('message.field_gathers'):
            s,d,t,msg=[torch.index_select(value,0,index) for value in self.data]
            if store.direction=='d':s,d=d,s
        if self.capture:
            self.records.append(dict(n_id=n_id.detach().cpu().clone(),direction=store.direction,
                event_ids=torch.from_numpy(events.copy()),
                **{k:v.detach().cpu().clone() for k,v in zip(FIELDS,[s,d,t,msg])}))
        self.clock.count('message_rows',len(events))
        return s,d,t,msg

    def update(self,memory,src,dst,t,raw_msg,store):
        assert store in self.stores and self.step_index is not None
        assert len(src)==self.tr.bs
        with self.clock.phase('store.sort'):
            _,perm=src.sort()
        with self.clock.phase('store.permutation_transfer'):
            events=perm.cpu().numpy()+self.step_index*self.tr.bs
        with self.clock.phase('store.cpu_group_and_write'):
            keys=self.cpu[0 if store.direction=='s' else 1][events]
            bounds=np.concatenate([np.zeros(1,np.int64),np.flatnonzero(keys[1:]!=keys[:-1])+1,np.array([len(keys)],np.int64)])
            for lo,hi in zip(bounds[:-1],bounds[1:]):
                store.nodes[int(keys[lo])]=events[lo:hi]
        self.clock.count('updated_nodes',len(bounds)-1)

    def expanded(self,result):
        checks=dict(result);storage={}
        for store in self.stores:
            lengths=np.array([len(x) for x in store.nodes],dtype=np.int64)
            events=np.concatenate(store.nodes)
            s,d,t,msg=[x[events] for x in self.cpu]
            if store.direction=='d':s,d=d,s
            prefix='cache/'+store.direction+'/'
            checks[prefix+'lengths']=torch.from_numpy(lengths)
            for name,value in zip(['src','dst','time','message'],[s,d,t,msg]):
                checks[prefix+name]=torch.from_numpy(value)
            unique={}
            for arr in store.nodes:
                if not len(arr):continue
                base=arr
                while isinstance(base.base,np.ndarray):base=base.base
                unique[id(base)]=base.nbytes
            storage[store.direction]=dict(logical_cache_bytes=sum(x.nbytes for x in [s,d,t,msg]),
                cpu_index_logical_bytes=events.nbytes,cpu_index_retained_bytes=sum(unique.values()),
                cpu_index_backing_arrays=len(unique),nodes=len(store.nodes),nonempty_nodes=int(np.count_nonzero(lengths)))
        checks['rng_cpu']=torch.get_rng_state().clone()
        checks['rng_cuda']=torch.cuda.get_rng_state().clone()
        return checks,storage


def prepare(tr,variant,start,mode='none',capture=False):
    clock=i.prepare(tr,'batch_store',mode)
    if variant=='batch_store':return clock
    assert variant=='event_index'
    state=EventState(tr,start,clock,capture);tr.event_state=state;memory=tr.model['memory']
    def restore(self,backup):state.restore(self,backup)
    def update(self,src,dst,t,raw_msg,msg_store):state.update(self,src,dst,t,raw_msg,msg_store)
    memory.restore_memory=types.MethodType(restore,memory)
    memory._update_msg_store=types.MethodType(update,memory)
    module,fn,split=source.source_method();tail=fn.body[split:]
    prefix=ast.parse('src,dst,t,raw_msg=_event.read(self,n_id,msg_store)').body
    if mode!='none':
        body=[]
        for stmt in tail:
            if isinstance(stmt,ast.Return):body.append(stmt);continue
            target=ast.unparse(stmt.targets[0]) if isinstance(stmt,ast.Assign) else ''
            name='message.time' if target in ['t_rel','t_enc'] else 'message.compose'
            body.append(ast.With(items=[ast.withitem(context_expr=ast.parse(f"_clock.phase('{name}')",mode='eval').body)],body=[stmt]))
        tail=body
    fn.body=prefix+tail
    source.bind(tr,fn,module,{'_event':state,'_clock':clock})
    return clock


def step(tr,index,check=False):
    if hasattr(tr,'event_state'):tr.event_state.step_index=index
    return i.b.step(tr,index,'direct',check=check)


def expanded_check(tr,result):
    return tr.event_state.expanded(result) if hasattr(tr,'event_state') else i.expanded_check(tr,result)
