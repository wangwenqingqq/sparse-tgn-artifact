"""Explicit cuSPARSE diagnostic binding; synchronous, not a timed backend."""
import ctypes as c
import hashlib
from pathlib import Path
import torch
P=c.c_void_p; I=c.c_int; L=c.c_int64; S=c.c_size_t
LIB=Path('/home/data/wangxuran/isaacsim6/env/lib/python3.12/site-packages/nvidia/cu13/lib/libcusparse.so.12')
HEADER=LIB.parent.parent/'include/cusparse.h'
assert hashlib.sha256(LIB.read_bytes()).hexdigest()=='09339f848f60bb1111a61ee0fe91ed0c25132b7ff63298244d7ac14e61b58466'
text=HEADER.read_text()
for literal in ['CUSPARSE_SPMM_CSR_ALG3         = 12','CUSPARSE_SPMM_CSR_ALG2         = 6','CUSPARSE_INDEX_64I = 3','CUSPARSE_ORDER_ROW = 2']:
    assert literal in text,literal
lib=c.CDLL(str(LIB))
SIGNATURES={
 'cusparseGetProperty':[I,c.POINTER(I)],'cusparseCreate':[c.POINTER(P)],'cusparseDestroy':[P],
 'cusparseSetStream':[P,P],'cusparseSetPointerMode':[P,I],
 'cusparseCreateCsr':[c.POINTER(P),L,L,L,P,P,P,I,I,I,I],
 'cusparseDestroySpMat':[P], 'cusparseCreateDnMat':[c.POINTER(P),L,L,L,P,I,I],
 'cusparseDestroyDnMat':[P],
 'cusparseSpMM_bufferSize':[P,I,I,P,P,P,P,P,I,I,c.POINTER(S)],
 'cusparseSpMM':[P,I,I,P,P,P,P,P,I,I,P],
}
for name,args in SIGNATURES.items():
    f=getattr(lib,name);f.argtypes=args;f.restype=I

def call(name,*args):
    rc=getattr(lib,name)(*args)
    if rc:raise RuntimeError(f'{name}: cuSPARSE status {rc}')

def identity():
    version=[]
    for k in range(3):
        v=I();call('cusparseGetProperty',k,c.byref(v));version.append(v.value)
    maps=Path('/proc/self/maps').read_text()
    mapped=sorted({line.split()[-1] for line in maps.splitlines() if 'libcusparse' in line})
    assert mapped==[str(LIB)],mapped
    return dict(library=str(LIB),library_sha256=hashlib.sha256(LIB.read_bytes()).hexdigest(),header=str(HEADER),header_sha256=hashlib.sha256(HEADER.read_bytes()).hexdigest(),version=version,mapped=mapped)

def transpose(p):
    """Paid GPU transpose; input has exact integer values and unique CSR entries."""
    rows=torch.repeat_interleave(torch.arange(p.m,device=p.rp.device,dtype=torch.int64),p.rp[1:]-p.rp[:-1],output_size=p.col.numel())
    keys=p.col*p.m+rows
    order=torch.argsort(keys,stable=True)
    cols=rows[order].contiguous();values=p.iv[order].contiguous()
    counts=torch.bincount(p.col,minlength=p.n)
    rp=torch.cat([torch.zeros(1,device=p.rp.device,dtype=torch.int64),counts.cumsum(0)])
    return rp,cols,values

def mm(rp,col,iv,m,n,x,alg):
    assert alg in (6,12) and x.dtype in (torch.float32,torch.float64)
    assert x.is_cuda and x.ndim==2 and x.shape[0]==n
    assert rp.dtype==col.dtype==iv.dtype==torch.int64
    assert all(t.device==x.device and t.is_contiguous() for t in (rp,col,iv))
    x=x.contiguous();d=x.shape[1];v=iv.to(x.dtype)
    y=torch.empty((m,d),device=x.device,dtype=x.dtype)
    typ=1 if x.dtype==torch.float64 else 0
    scalar=c.c_double if typ else c.c_float
    alpha,beta=scalar(1),scalar(0)
    h,a,b,o=P(),P(),P(),P();workspace=None
    try:
        call('cusparseCreate',c.byref(h));call('cusparseSetStream',h,torch.cuda.current_stream().cuda_stream)
        call('cusparseSetPointerMode',h,0)
        call('cusparseCreateCsr',c.byref(a),m,n,col.numel(),rp.data_ptr(),col.data_ptr(),v.data_ptr(),3,3,0,typ)
        call('cusparseCreateDnMat',c.byref(b),n,d,d,x.data_ptr(),typ,2)
        call('cusparseCreateDnMat',c.byref(o),m,d,d,y.data_ptr(),typ,2)
        args=(h,0,0,c.byref(alpha),a,b,c.byref(beta),o,typ,alg)
        size=S();call('cusparseSpMM_bufferSize',*args,c.byref(size))
        workspace=torch.empty(max(1,size.value),device=x.device,dtype=torch.uint8)
        call('cusparseSpMM',*args,workspace.data_ptr())
        torch.cuda.current_stream().synchronize()
        return y,dict(workspace_bytes=size.value,workspace_allocated_bytes=workspace.numel(),value_bytes=v.numel()*v.element_size(),output_bytes=y.numel()*y.element_size(),algorithm=alg)
    finally:
        # Keep allocations alive through asynchronous completion even on API failure.
        torch.cuda.current_stream().synchronize()
        for desc,fn in [(o,'cusparseDestroyDnMat'),(b,'cusparseDestroyDnMat'),(a,'cusparseDestroySpMat'),(h,'cusparseDestroy')]:
            if desc.value:call(fn,desc)

def run(p,tr,x,target,alg):
    b=p.m//7;d=x.shape[1]
    yc,fw=mm(p.rp,p.col,p.iv,p.m,p.n,x,alg)
    y=yc.reshape(7,b,d).permute(1,0,2).reshape(b,7*d)
    g=2*(y-target)
    gc=g.reshape(b,7,d).permute(1,0,2).reshape(p.m,d)
    dx,bw=mm(*tr,p.n,p.m,gc,alg)
    return dict(Y=y,g=g,dx=dx),dict(forward=fw,backward=bw,transpose_storage_bytes=sum(t.numel()*t.element_size() for t in tr),csr_storage_bytes=sum(t.numel()*t.element_size() for t in (p.rp,p.col,p.iv)))
