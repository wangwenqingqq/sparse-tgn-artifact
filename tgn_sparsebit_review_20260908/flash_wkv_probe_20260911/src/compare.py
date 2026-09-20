"""CPU-only, bounded-memory full tensor comparison at the original fixed gate."""
import numpy as np
import torch


def compare(actual,expected):
    assert actual.keys()==expected.keys();rows={}
    for name,x in actual.items():
        y=expected[name]
        assert torch.is_tensor(x) and torch.is_tensor(y) and x.device.type==y.device.type=='cpu'
        assert x.shape==y.shape and x.dtype==y.dtype,(name,x.shape,y.shape,x.dtype,y.dtype)
        xx=x.contiguous().numpy().reshape(-1);yy=y.contiguous().numpy().reshape(-1)
        nbad=0;different=0;max_abs=0.;max_scaled=0.;exact=True;finite=True
        for start in range(0,len(xx),1<<20):
            a=xx[start:start+(1<<20)];b=yy[start:start+(1<<20)]
            exact &= bool(np.array_equal(a.view(np.uint8),b.view(np.uint8)))
            different+=int(np.count_nonzero(a!=b))
            if a.dtype.kind=='f':
                a=a.astype(np.float64);b=b.astype(np.float64);delta=np.abs(a-b);bound=2e-4+2e-4*np.abs(b)
                good=np.isfinite(a)&np.isfinite(b);finite &= bool(good.all());nbad+=int(np.count_nonzero(~good|(delta>bound)))
                max_abs=max(max_abs,float(delta.max(initial=0)));max_scaled=max(max_scaled,float((delta/bound).max(initial=0)))
            else:nbad+=int(np.count_nonzero(a!=b))
        rows[name]=dict(shape=list(x.shape),dtype=str(x.dtype),elements=x.numel(),bitwise=exact,finite=finite,different=different,failed_elements=nbad,max_abs=max_abs,max_scaled_error=max_scaled,pass_all=nbad==0)
    return dict(fields=rows,pass_all=all(x['pass_all'] for x in rows.values()),bitwise=all(x['bitwise'] for x in rows.values()),elements=sum(x['elements'] for x in rows.values()))
