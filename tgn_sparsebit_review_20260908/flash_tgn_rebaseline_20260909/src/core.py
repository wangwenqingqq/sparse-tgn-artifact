"""Instrumentation around the supplied FlashTGN entry point; no TNCN imports."""
import ast
import contextlib
import copy
import dataclasses
import functools
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import types

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/home/data/wangxuran/factor_tgn_sptc_20260902/vendor/flash-tgn-artifact')
DATA=Path('/home/data/wangxuran/factor_tgn_sptc_20260902/project/experiments/factor_tgn_20260904_anchored_time_quality_wikipedia_full/data_root')
sys.path[:0]=[str(ROOT/'deps'),str(BASE/'python')]
import numpy as np
import torch
import sklearn
import flash_tgn
import flash_tgn.trainer as tm
import flash_tgn.schedule as sm
from flash_tgn.extension import cuda_extension
from flash_tgn.operator_policy import reset_operator_counters


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()


def emit(kind,**kw):print(json.dumps(dict(kind=kind,**kw)),flush=True)


def guard(initial=False):
    gpu=os.environ['CUDA_VISIBLE_DEVICES']
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True)
    foreign=[line for line in apps.splitlines() if gpu in line and int(line.split(',')[1])!=os.getpid()]
    assert not foreign,foreign
    return apps


def cases():
    return [dict(bs=bs,k=k,layers=layers,label=f'wiki_bs{bs}_k{k}_l{layers}')
            for bs in [200,2000] for k in [20,50] for layers in [1,2]]


class Clock:
    def __init__(self,mode='coarse'):
        assert mode in ['none','coarse','events']
        self.mode=mode;self.stack=[];self.rows=[];self.epoch_id=0;self.tr=None;self.monitor=None

    @contextlib.contextmanager
    def phase(self,name,essential=False):
        if self.mode=='none' or (self.mode=='coarse' and not essential):
            yield;return
        self.stack.append(name);path='/'.join(self.stack)
        a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();start=time.perf_counter_ns()
        try:yield
        finally:
            z.record();self.rows.append((self.epoch_id,path,(time.perf_counter_ns()-start)/1e6,a,z));self.stack.pop()

    def results(self,epoch=None):
        rows={}
        for e,k,host,a,z in self.rows:
            if epoch is not None and e!=epoch:continue
            item=rows.setdefault(k,dict(calls=0,host_ms=0.,timeline_ms=0.))
            item['calls']+=1;item['host_ms']+=host;item['timeline_ms']+=a.elapsed_time(z)
        return rows

    @contextlib.contextmanager
    def epoch(self,number):
        self.epoch_id=int(number)
        before=self.monitor.counts.copy() if self.monitor else {}
        with self.phase('epoch',True):yield
        torch.cuda.synchronize()
        phases=self.results(self.epoch_id)
        result=getattr(self.tr,'observed_epoch_result',None)
        assert result is not None and math.isfinite(result.loss)
        emit('epoch',epoch=self.epoch_id,mode=self.mode,case=self.tr.case['label'],
            source_result=dataclasses.asdict(result),phases=phases,
            cuda_calls={k:v-before.get(k,0) for k,v in self.monitor.counts.items()},
            operator_counts=getattr(flash_tgn,'_flash_tgn_operator_counts',{}).copy(),
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
        assert not self.monitor.errors,self.monitor.errors
        self.rows=[r for r in self.rows if r[0]!=self.epoch_id]


ACTIVE_CLOCK=None


class Monitor:
    def __init__(self):
        self.counts={};self.errors=[];self.samples=[];self.capture_samples=False
        module=cuda_extension();self.originals={}
        for name in dir(module):
            if not (name.startswith('fused_') or name=='tcsr_temporal_sample'):continue
            original=getattr(module,name);self.originals[name]=original
            def call(*args,_name=name,_original=original,**kw):
                clock=ACTIVE_CLOCK
                with clock.phase('cuda.'+_name) if clock else contextlib.nullcontext():
                    try:result=_original(*args,**kw)
                    except Exception as exc:
                        self.errors.append(dict(name=_name,error=repr(exc)));raise
                self.counts[_name]=self.counts.get(_name,0)+1
                if self.capture_samples and _name=='tcsr_temporal_sample':
                    query_nodes,query_times=args[4:6];q=len(query_nodes)
                    positions=torch.linspace(0,q-1,min(q,64),device=query_nodes.device).long().unique()
                    self.samples.append(dict(total_queries=q,k=args[6],positions=positions.cpu(),
                        nodes=query_nodes[positions].cpu(),times=query_times[positions].cpu(),
                        neighbors=result[0][positions].cpu(),events=result[1][positions].cpu(),timestamps=result[2][positions].cpu()))
                return result
            setattr(module,name,call)

    def reset(self):self.counts.clear();self.errors.clear();self.samples.clear()


def install_global_scopes():
    def wrap(original,name):
        @functools.wraps(original)
        def call(*args,**kw):
            with ACTIVE_CLOCK.phase(name) if ACTIVE_CLOCK else contextlib.nullcontext():return original(*args,**kw)
        return call
    original=sm.precompute_epoch_schedule
    sm.precompute_epoch_schedule=wrap(original,'schedule_build')
    tm.precompute_epoch_schedule=sm.precompute_epoch_schedule
    torch.autograd.backward=wrap(torch.autograd.backward,'backward')
    torch.cuda.empty_cache=wrap(torch.cuda.empty_cache,'allocator_empty_cache')


def make(case,epochs=4):
    spec=importlib.util.spec_from_file_location('flash_entry',BASE/'train_flash_tgn.py')
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    tree=ast.parse((BASE/'train_flash_tgn.py').read_text())
    original=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    fn=copy.deepcopy(original)
    assert ast.unparse(fn.body[-1])=='trainer.train()'
    fn.body[-1]=ast.Return(value=ast.Name(id='trainer',ctx=ast.Load()))
    assert ast.dump(ast.Module(body=fn.body[:-1],type_ignores=[]))==ast.dump(ast.Module(body=original.body[:-1],type_ignores=[]))
    ns=dict(vars(entry));exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(BASE/'train_flash_tgn.py')+'[return-trainer]','exec'),ns)
    args=['train_flash_tgn.py','-d','wiki','--data-path',str(DATA),'--gpu','0','--epochs',str(epochs),
        '--bsize',str(case['bs']),'--n-nbrs',str(case['k']),'--n-layers',str(case['layers']),
        '--dim-time','256','--dim-embed','256','--num-heads','2','--dropout','0.1','--seed','42',
        '--precompute-mode','auto','--train-only','--model-dir',str(ROOT/'unused_train_only_models')]
    old=sys.argv;sys.argv=args
    try:tr=ns['main']()
    finally:sys.argv=old
    tr.case=case
    assert tr.graph.edge_feature_mode=='cpu' and tr.graph.num_edges==157474 and tr.graph.num_nodes==9227
    assert tr.graph.dim_edge==172 and tr.graph.dim_node==256
    assert tr.cfg.max_train_batches is None
    return tr


def wrap_method(obj,method,clock,name,essential=False,after=None):
    original=getattr(obj,method)
    def call(*args,**kw):
        with clock.phase(name,essential):result=original(*args,**kw)
        if after is not None:after(result,*args,**kw)
        return result
    setattr(obj,method,call)


def attach(tr,clock,monitor):
    global ACTIVE_CLOCK
    ACTIVE_CLOCK=clock;clock.tr=tr;clock.monitor=monitor;tr.clock=clock
    tr.capture=False;tr.captured=[]
    def save_epoch(result,*args,**kw):tr.observed_epoch_result=result
    wrap_method(tr,'run_epoch',clock,'online_loop',True,save_epoch)
    wrap_method(tr,'_build_schedules',clock,'schedule_full',True)
    wrap_method(tr.state,'reset',clock,'state_reset')
    wrap_method(tr.state,'update_memory_for_batch',clock,'memory_update')
    wrap_method(tr.state,'store_raw_messages',clock,'mailbox_write')
    wrap_method(tr.features,'projected_node_features',clock,'node_features')
    wrap_method(tr.features,'compact_edge_features',clock,'edge_features')
    wrap_method(tr.model,'forward_embed',clock,'embedding')
    wrap_method(tr.model,'predict',clock,'predictor')
    wrap_method(tr.criterion,'forward',clock,'loss')
    wrap_method(tr.optimizer,'zero_grad',clock,'zero_grad')
    wrap_method(tr.optimizer,'step',clock,'optimizer')
    for name,module in tr.model.named_modules():
        if isinstance(module,(torch.nn.Linear,torch.nn.GRUCell)):
            wrap_method(module,'forward',clock,'dense.'+name)
    # This wrapper is the only modification of the actual source train body.
    tree=ast.parse(Path(tm.__file__).read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FlashTGNTrainer')
    original=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='train')
    fn=copy.deepcopy(original)
    loop=next(n for n in fn.body if isinstance(n,ast.For) and ast.unparse(n.target)=='epoch')
    original_body=loop.body
    loop.body=[ast.With(items=[ast.withitem(context_expr=ast.parse('_clock.epoch(epoch)',mode='eval').body)],body=original_body)]
    stripped=copy.deepcopy(fn)
    stripped_loop=next(n for n in stripped.body if isinstance(n,ast.For) and ast.unparse(n.target)=='epoch')
    stripped_loop.body=stripped_loop.body[0].body
    assert ast.dump(stripped)==ast.dump(original),'non-timing source change'
    ns=dict(vars(tm),_clock=clock)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(Path(tm.__file__))+'[epoch-timer]','exec'),ns)
    tr.train=types.MethodType(ns['train'],tr)


def snapshot(tr):
    result={}
    for name,param in tr.model.named_parameters():
        result['parameter/'+name]=param.detach().cpu().clone()
        result['gradient/'+name]=None if param.grad is None else param.grad.detach().cpu().clone()
        for key,value in tr.optimizer.state.get(param,{}).items():
            result['adam/'+name+'/'+key]=value.detach().cpu().clone() if torch.is_tensor(value) else value
    for key in ['mem_data','mem_ts','mail_data','mail_ts']:
        result['state/'+key]=getattr(tr.state,key).detach().cpu().clone()
    result['rng/cpu']=torch.get_rng_state().clone();result['rng/cuda']=torch.cuda.get_rng_state().clone()
    state=np.random.get_state()
    result['rng/numpy_array']=torch.from_numpy(state[1].astype(np.int64))
    result['rng/numpy_other']=(state[0],*state[2:])
    result['rng/python']=random.getstate()
    return result


def checkpoint(tr):
    return dict(model=copy.deepcopy(tr.model.state_dict()),optimizer=copy.deepcopy(tr.optimizer.state_dict()),
        state={key:getattr(tr.state,key).detach().clone() for key in ['mem_data','mem_ts','mail_data','mail_ts']},
        cpu=torch.get_rng_state().clone(),cuda=torch.cuda.get_rng_state().clone(),numpy=copy.deepcopy(np.random.get_state()),python=random.getstate())


def restore(tr,ck):
    tr.model.load_state_dict(ck['model']);tr.optimizer.load_state_dict(copy.deepcopy(ck['optimizer']))
    for key,value in ck['state'].items():setattr(tr.state,key,value.clone())
    torch.set_rng_state(ck['cpu']);torch.cuda.set_rng_state(ck['cuda']);np.random.set_state(ck['numpy']);random.setstate(ck['python'])


def capture_steps(tr):
    original_embed=tr.model.forward_embed
    def embed(*args,**kw):
        result=original_embed(*args,**kw)
        if tr.capture:
            if result.requires_grad:result.retain_grad()
            tr.latest_embedding=result
        return result
    tr.model.forward_embed=embed
    original_step=tr._run_model_step
    def step(*args,**kw):
        result=original_step(*args,**kw)
        if tr.capture:tr.latest_result=result
        return result
    tr._run_model_step=step
    original_write=tr.state.store_raw_messages
    def write(*args,**kw):
        result=original_write(*args,**kw)
        if tr.capture:
            full=snapshot(tr)
            for name,tensor in zip(['loss','positive_logits','negative_logits'],tr.latest_result):full[name]=tensor.detach().cpu().clone()
            full['embedding']=tr.latest_embedding.detach().cpu().clone()
            full['embedding_gradient']=None if tr.latest_embedding.grad is None else tr.latest_embedding.grad.detach().cpu().clone()
            tr.captured.append(full)
        return result
    tr.state.store_raw_messages=write


def compare(actual,expected):
    assert actual.keys()==expected.keys()
    failures=[];different=[];elements=0;worst=[]
    for key,a in actual.items():
        b=expected[key]
        if torch.is_tensor(a):
            assert torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype,(key,a.shape,getattr(b,'shape',None))
            x=a.numpy();y=b.numpy();elements+=x.size
            exact=x.tobytes()==y.tobytes()
            if not exact:different.append(key)
            if np.issubdtype(x.dtype,np.floating):
                diff=np.abs(x.astype(np.float64)-y.astype(np.float64));limit=2e-4+2e-4*np.abs(y.astype(np.float64))
                ok=np.isfinite(x)&np.isfinite(y)&(diff<=limit)
                if not np.all(ok):failures.append(dict(field=key,count=int(np.count_nonzero(~ok)),max_abs=float(diff.max())))
                worst.append((float(diff.max()) if x.size else 0.,key))
            elif not exact:failures.append(dict(field=key,count=int(np.count_nonzero(x!=y))))
        elif a!=b:
            different.append(key);failures.append(dict(field=key,reason='scalar/None/RNG mismatch'))
    return dict(pass_all=not failures,bitwise_equal=not different,fields=len(actual),elements=elements,
        different_fields=different,failures=failures,worst=sorted(worst,reverse=True)[:8])
