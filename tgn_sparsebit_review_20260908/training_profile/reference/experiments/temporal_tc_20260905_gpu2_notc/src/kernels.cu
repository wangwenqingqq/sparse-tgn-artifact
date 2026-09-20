// Native ordinary sparse controls. All kernels launch on the caller's stream.
#include <cuda_runtime.h>
#include <stdint.h>
using I=int64_t;
using U=unsigned long long;
__device__ I lookup(const I* rp,const I* ci,const I* av,I u,I v) {
    I lo=rp[u],hi=rp[u+1];
    while(lo<hi) {I m=(lo+hi)/2;if(ci[m]<v)lo=m+1;else hi=m;}
    return lo<rp[u+1]&&ci[lo]==v?av[lo]:0;
}
__global__ void bitmap(int n,const I* rp,const I* ci,uint32_t* p) {
    int u=blockIdx.x,w=(n+31)/32;
    for(I k=rp[u]+threadIdx.x;k<rp[u+1];k+=blockDim.x) {
        I c=ci[k];atomicOr(p+u*w+c/32,1u<<(c%32));
    }
}
__device__ I coefficient(int channel,int z,int u,int v,I e,int w,const uint32_t* p,const uint32_t* q,const U* s) {
    I a=(p[u*w+z/32]>>(z%32))&1u,b=(p[v*w+z/32]>>(z%32))&1u;
    I h=(q[z/32]>>(z%32))&1u,k=(q[w+z/32]>>(z%32))&1u;
    if(channel==0)return z==u?b:0;
    if(channel==1)return z==v?a:0;
    if(channel==2)return a*b;
    if(channel==6)return I(s[z]);
    if(z==u||z==v)return 0;
    if(channel==3)return a*(k-e);
    if(channel==4)return b*(h-e);
    return e>0&&(a||b)?0:h*k+I(s[z]);
}
template<int MODE,bool PACK> __global__ void produce(int n,int b,const I* rp,const I* ci,const I* av,
        const uint32_t* p,const I* targets,I* counts,const I* outptr,I* outcol,I* outval) {
    extern __shared__ U raw[];
    U* s=raw;uint32_t* q=(uint32_t*)(s+n);int w=(n+31)/32;
    int* work=(int*)(q+2*w);__shared__ I e;
    int r=blockIdx.x,t=threadIdx.x,lane=t%32,warp=t/32,u=int(targets[r]),v=int(targets[b+r]);
    for(int z=t;z<n;z+=256)s[z]=0;
    for(int z=t;z<2*w;z+=256)q[z]=0;
    if(t==0)e=lookup(rp,ci,av,u,v);
    __syncthreads();
    if constexpr(MODE==1) {
        for(int z=t;z<2*w;z+=256) {
            I root=z<w?u:v;uint32_t acc=0;
            for(I k=rp[root];k<rp[root+1];++k)acc|=p[ci[k]*w+z%w];
            q[z]=acc;
        }
    } else {
        for(int side=0;side<2;++side) {
            I root=side?v:u;
            for(I k=rp[root]+warp;k<rp[root+1];k+=8) {
                I x=ci[k];
                for(I j=rp[x]+lane;j<rp[x+1];j+=32) {
                    I z=ci[j];atomicOr(q+side*w+z/32,1u<<(z%32));
                }
            }
        }
    }
    for(I k=rp[u]+warp;k<rp[u+1];k+=8) {
        I x=ci[k];
        if((p[v*w+x/32]>>(x%32))&1u)
            for(I j=rp[x]+lane;j<rp[x+1];j+=32)atomicAdd(s+ci[j],U(av[j]));
    }
    __syncthreads();
    for(int channel=0;channel<7;++channel) {
        int row=channel*b+r;
        if constexpr(!PACK) {
            int count=0;
            for(int z=t;z<n;z+=256)count+=coefficient(channel,z,u,v,e,w,p,q,s)!=0;
            for(int delta=16;delta;delta/=2)count+=__shfl_down_sync(0xffffffff,count,delta);
            if(lane==0)work[warp]=count;
            __syncthreads();
            if(t==0) {int sum=0;for(int j=0;j<8;++j)sum+=work[j];counts[row]=sum;}
            __syncthreads();
        } else {
            int base=0;
            for(int tile=0;tile<n;tile+=256) {
                int z=tile+t;I value=z<n?coefficient(channel,z,u,v,e,w,p,q,s):0;
                uint32_t mask=__ballot_sync(0xffffffff,value!=0);
                if(lane==0)work[warp]=__popc(mask);
                __syncthreads();
                if(t==0) {int sum=0;for(int j=0;j<8;++j){work[8+j]=sum;sum+=work[j];}work[16]=sum;}
                __syncthreads();
                if(value) {
                    int rank=__popc(mask&((1u<<lane)-1u));I at=outptr[row]+base+work[8+warp]+rank;
                    outcol[at]=z;outval[at]=value;
                }
                base+=work[16];
                __syncthreads();
            }
        }
    }
}
template<class T,bool TRANS> __global__ void consume(int rows,int d,const I* rp,const I* col,
        const I* perm,const T* value,const T* x,T* y) {
    int row=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;
    if(row>=rows)return;
    for(int f=lane;f<d;f+=32) {
        T acc=0;
        for(I k=rp[row];k<rp[row+1];++k)acc+=value[TRANS?perm[k]:k]*x[col[k]*d+f];
        y[row*d+f]=acc;
    }
}
extern "C" int nt_bits(int n,const I* rp,const I* col,uint32_t* p,void* stream) {
    bitmap<<<n,256,0,(cudaStream_t)stream>>>(n,rp,col,p);return int(cudaGetLastError());
}
extern "C" int nt_produce(int n,int b,const I* rp,const I* ci,const I* av,const uint32_t* p,
        const I* targets,I* counts,const I* op,I* oc,I* ov,int mode,int pack,void* stream) {
    size_t shared=8*n+4*(2*((n+31)/32)+18);cudaStream_t st=(cudaStream_t)stream;
    if(mode==0&&pack==0)produce<0,false><<<b,256,shared,st>>>(n,b,rp,ci,av,p,targets,counts,op,oc,ov);
    if(mode==1&&pack==0)produce<1,false><<<b,256,shared,st>>>(n,b,rp,ci,av,p,targets,counts,op,oc,ov);
    if(mode==0&&pack==1)produce<0,true><<<b,256,shared,st>>>(n,b,rp,ci,av,p,targets,counts,op,oc,ov);
    if(mode==1&&pack==1)produce<1,true><<<b,256,shared,st>>>(n,b,rp,ci,av,p,targets,counts,op,oc,ov);
    return int(cudaGetLastError());
}
extern "C" int nt_mm(int rows,int d,const I* rp,const I* col,const I* perm,const void* value,
        const void* x,void* y,int fp64,int trans,void* stream) {
    cudaStream_t st=(cudaStream_t)stream;int grid=(rows+7)/8;
    if(fp64&&trans)consume<double,true><<<grid,256,0,st>>>(rows,d,rp,col,perm,(const double*)value,(const double*)x,(double*)y);
    if(fp64&&!trans)consume<double,false><<<grid,256,0,st>>>(rows,d,rp,col,perm,(const double*)value,(const double*)x,(double*)y);
    if(!fp64&&trans)consume<float,true><<<grid,256,0,st>>>(rows,d,rp,col,perm,(const float*)value,(const float*)x,(float*)y);
    if(!fp64&&!trans)consume<float,false><<<grid,256,0,st>>>(rows,d,rp,col,perm,(const float*)value,(const float*)x,(float*)y);
    return int(cudaGetLastError());
}
