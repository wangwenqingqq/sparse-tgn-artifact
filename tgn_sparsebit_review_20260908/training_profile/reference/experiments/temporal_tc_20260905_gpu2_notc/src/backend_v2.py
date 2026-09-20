"""Native GPU ordinary relation/feature control; no Tensor Core kernels."""
import ctypes as c
import os
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
LIB=Path(os.environ['GPU_NT_LIB']);lib=c.CDLL(str(LIB));P=c.c_void_p;I=c.c_int
lib.nt_bits.argtypes=[I,P,P,P,P];lib.nt_bits.restype=I
lib.nt_produce.argtypes=[I,I]+[P]*9+[I,I,P];lib.nt_produce.restype=I
lib.nt_mm.argtypes=[I,I]+[P]*6+[I,I,P];lib.nt_mm.restype=I
def ptr(t):return t.data_ptr() if t is not None else None
def call(name,*args):
    rc=getattr(lib,name)(*args)
    if rc:raise RuntimeError(f'{name}: CUDA error {rc}')
def stream():return torch.cuda.current_stream().cuda_stream

def validate(n,edges,targets):
    if not 1<=n<=2048 or edges.ndim!=2 or targets.ndim!=2 or edges.shape[0]!=2 or targets.shape[0]!=2 or not 1<=targets.shape[1]<=64 or edges.shape[1]>1000000:raise ValueError('bounded shape')
    for t in (edges,targets):
        if t.dtype!=torch.int64 or t.device.type!='cuda':raise TypeError('CUDA int64 required')
        if t.numel() and (t.min().item()<0 or t.max().item()>=n):raise ValueError('ID range')
    if edges.device!=targets.device or edges.device!=torch.device('cuda',torch.cuda.current_device()):raise ValueError('current device mismatch')

class Plan:
    def __init__(self,n,edges,targets,mode='list',checked=False,build_transpose=True):
        if not checked:validate(n,edges,targets)
        if mode not in ['list','bitset']:raise ValueError('mode')
        self.n,self.b=n,targets.shape[1];self.m=7*self.b;device=edges.device
        targets=targets.contiguous();sym=torch.cat([edges,edges.flip(0)],dim=1)
        a=torch.sparse_coo_tensor(sym,torch.ones(sym.shape[1],device=device,dtype=torch.int64),(n,n)).coalesce()
        ac=a.indices()[1].contiguous();av=a.values();ar=a.indices()[0]
        ap=torch.cat([torch.zeros(1,device=device,dtype=torch.int64),torch.bincount(ar,minlength=n).cumsum(0)])
        bits=torch.zeros((n,(n+31)//32),device=device,dtype=torch.int32)
        call('nt_bits',n,ptr(ap),ptr(ac),ptr(bits),stream())
        counts=torch.empty(self.m,device=device,dtype=torch.int64)
        args=[n,self.b,ptr(ap),ptr(ac),ptr(av),ptr(bits),ptr(targets),ptr(counts)]
        call('nt_produce',*args,None,None,None,int(mode=='bitset'),0,stream())
        self.rp=torch.cat([torch.zeros(1,device=device,dtype=torch.int64),counts.cumsum(0)])
        self.nnz=int(self.rp[-1].item()) # Charged host synchronization / exact allocation; not Graph compatible.
        self.col=torch.empty(self.nnz,device=device,dtype=torch.int64);self.iv=torch.empty_like(self.col)
        call('nt_produce',*args,ptr(self.rp),ptr(self.col),ptr(self.iv),int(mode=='bitset'),1,stream())
        self.row=self.perm=self.tr=self.tp=None
        if build_transpose:
            self.row=torch.repeat_interleave(torch.arange(self.m,device=device,dtype=torch.int64),counts,output_size=self.nnz)
            self.perm=torch.argsort(self.col*self.m+self.row);self.tr=self.row[self.perm]
            self.tp=torch.cat([torch.zeros(1,device=device,dtype=torch.int64),torch.bincount(self.col,minlength=n).cumsum(0)])
    def values(self,dtype,decay=False,time_info=None):
        if dtype not in (torch.float32,torch.float64):raise TypeError('FP32/64 only')
        v=self.iv.to(dtype)
        if decay:
            if self.row is None:
                self.row=torch.repeat_interleave(torch.arange(self.m,device=self.iv.device,dtype=torch.int64),self.rp[1:]-self.rp[:-1],output_size=self.nnz)
            last,times=time_info
            if last.requires_grad or times.requires_grad:raise ValueError('time gradient not admitted')
            factor=torch.exp(-(times[self.row%self.b]-last[self.col])/10000)
            v=v*torch.where(self.row<6*self.b,factor,torch.ones_like(factor))
        return v
    def mm(self,v,x,trans=False):
        if x.dtype not in (torch.float32,torch.float64) or x.device!=self.iv.device or x.ndim!=2 or not 1<=x.shape[1]<=128:raise ValueError('feature contract')
        if x.shape[0]!=(self.m if trans else self.n) or v.dtype!=x.dtype or v.device!=x.device or v.numel()!=self.nnz:raise ValueError('mm shape')
        if trans and self.tp is None:raise ValueError('native transpose was not constructed')
        x=x.contiguous();v=v.contiguous();rows=self.n if trans else self.m
        out=torch.empty((rows,x.shape[1]),dtype=x.dtype,device=x.device)
        call('nt_mm',rows,x.shape[1],ptr(self.tp if trans else self.rp),ptr(self.tr if trans else self.col),ptr(self.perm),ptr(v),ptr(x),ptr(out),int(x.dtype==torch.float64),int(trans),stream())
        return out
    def dense(self):
        out=torch.zeros((self.m,self.n),device=self.iv.device,dtype=torch.int64);out[self.row,self.col]=self.iv
        return out.reshape(7,self.b,self.n)
    def consume(self,x,decay=False,time_info=None,library=False):
        v=self.values(x.dtype,decay,time_info)
        if library:
            a=torch.sparse_csr_tensor(self.rp,self.col,v,size=(self.m,self.n))
            y=torch.sparse.mm(a,x)
        else:y=Consume.apply(x,self,v)
        return y.reshape(7,self.b,x.shape[1]).permute(1,0,2).reshape(self.b,-1)

class Consume(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,plan,v):
        ctx.plan=plan;ctx.save_for_backward(v)
        return plan.mm(v,x)
    @staticmethod
    def backward(ctx,grad):
        v,=ctx.saved_tensors
        return ctx.plan.mm(v,grad,True),None,None

def decode(model,x,edges,targets,mode='list',decay=False,time_info=None,library=False):
    p=Plan(len(x),edges,targets,mode,checked=True,build_transpose=not library)
    y=p.consume(x,decay,time_info,library)
    return model.xsmlp(torch.cat([x[targets[0]]*x[targets[1]],y],-1))

def pair(model,x,edges,targets,variant):
    mode='bitset' if variant.startswith('B') else 'list';library=variant.endswith('library')
    if 'shared' in variant:
        p=Plan(len(x),edges,targets,mode,checked=True,build_transpose=not library);y=p.consume(x,library=library)
        features=torch.cat([x[targets[0]]*x[targets[1]],y],-1)
        return model.xsmlp(features[:32]),model.xsmlp(features[32:])
    return decode(model,x,edges,targets[:,:32],mode),decode(model,x,edges,targets[:,32:],mode)
