#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);} } while(0)
constexpr unsigned FULL=0xffffffffu;

__device__ __forceinline__ unsigned hpair(float a,float b){
    return unsigned(__half_as_ushort(__float2half_rn(a))) |
           (unsigned(__half_as_ushort(__float2half_rn(b)))<<16);
}
__device__ __forceinline__ void bmma(int (&d)[4],unsigned a0,unsigned a1,unsigned b){
    asm volatile("mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc "
      "{%0,%1,%2,%3},{%4,%5},{%6},{%0,%1,%2,%3};"
      : "+r"(d[0]),"+r"(d[1]),"+r"(d[2]),"+r"(d[3])
      : "r"(a0),"r"(a1),"r"(b));
}
__device__ __forceinline__ void smma(float (&d)[4],const unsigned *a,const unsigned *b,unsigned e){
    asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%0,%1,%2,%3},%12,0;"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),
        "r"(b[0]),"r"(b[1]),"r"(b[2]),"r"(b[3]),"r"(e));
}
__device__ __forceinline__ void dmma(float (&d)[4],const unsigned *a,const unsigned *b){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}

template<bool Aligned>
__device__ __forceinline__ void make_masks(const unsigned *bits,int words,unsigned (&m)[4]){
    const int lane=threadIdx.x%32,g=lane/4,t=lane%4;
    #pragma unroll
    for(int p=0;p<4;++p){
        const int z=Aligned ? 16*(p/2)+4*(g/2)+2*(p%2)+g%2 : 8*p+g;
        int d[4]={0,0,0,0};
        for(int w=0;w<words;w+=4)
            bmma(d,bits[g*words+w+t],bits[(g+8)*words+w+t],bits[(16+z)*words+w+t]);
        m[p]=(d[0]>0) | ((d[1]>0)<<1) | ((d[2]>0)<<2) | ((d[3]>0)<<3);
    }
}

template<int Mode>
__device__ __forceinline__ void sparse_coefficients(const unsigned *bits,int words,const float *w,
                                                  float (&v)[4][4],unsigned char *qout){
    const int lane=threadIdx.x%32,g=lane/4,t=lane%4;
    unsigned masks[4];
    if constexpr(Mode!=3)make_masks<Mode!=0>(bits,words,masks);
    #pragma unroll
    for(int h=0;h<2;++h){
        unsigned low=0,high=0;
        if constexpr(Mode==0){
            // Shuffle compressed four-bit Boolean packets, never s32 counts.
            const int src=4*g+2*(t%2);
            unsigned a=__shfl_sync(FULL,masks[2*h],src);
            unsigned b=__shfl_sync(FULL,masks[2*h],src+1);
            unsigned c=__shfl_sync(FULL,masks[2*h+1],src);
            unsigned d=__shfl_sync(FULL,masks[2*h+1],src+1);
            low=t<2?a:c;high=t<2?b:d;
        }else if constexpr(Mode!=3){low=masks[2*h];high=masks[2*h+1];}
        #pragma unroll
        for(int rh=0;rh<2;++rh){
            const int row=g+8*rh;
            #pragma unroll
            for(int j=0;j<4;++j){
                const int z=16*h+4*t+j;
                bool q;
                if constexpr(Mode==3){
                    q=false;
                    for(int k=0;k<words;++k){
                        if(bits[row*words+k]&bits[(16+z)*words+k]){q=true;break;}
                    }
                }else q=(((j<2?low:high)>>(2*rh+j%2))&1)!=0;
                if(qout)qout[row*32+z]=q;
                v[2*h+rh][j]=q?w[row*32+z]:0.0f;
            }
        }
    }
}

__device__ __forceinline__ void encode4(const float *v,int plane,unsigned &a,unsigned &e){
    unsigned mask=0;
    #pragma unroll
    for(int j=0;j<4;++j)mask|=(v[j]!=0.0f)<<j;
    if(plane){mask&=mask-1;mask&=mask-1;}
    int i=mask?__ffs(mask)-1:0;
    unsigned rest=mask & (mask-1);
    int j=rest?__ffs(rest)-1:(i==0?1:0);
    float vi=mask?v[i]:0.0f,vj=rest?v[j]:0.0f;
    if(j<i){int z=i;i=j;j=z;float f=vi;vi=vj;vj=f;}
    a=hpair(vi,vj);e=unsigned(i)|(unsigned(j)<<2);
}
__device__ __forceinline__ void sparse_pack(const float (&v)[4][4],unsigned (&a)[2][4],unsigned (&e)[2],bool &two){
    const int t=threadIdx.x%4;
    bool local=false;
    #pragma unroll
    for(int r=0;r<4;++r){int nz=0;
        #pragma unroll
        for(int j=0;j<4;++j)nz+=v[r][j]!=0;
        local|=nz>2;
    }
    two=__any_sync(FULL,local);
    #pragma unroll
    for(int plane=0;plane<2;++plane){
        unsigned nib[4];
        #pragma unroll
        for(int r=0;r<4;++r)encode4(v[r],plane,a[plane][r],nib[r]);
        unsigned lo=(nib[0]<<(4*t)) | (nib[1]<<(16+4*t));
        unsigned hi=(nib[2]<<(4*t)) | (nib[3]<<(16+4*t));
        lo|=__shfl_xor_sync(FULL,lo,1);lo|=__shfl_xor_sync(FULL,lo,2);
        hi|=__shfl_xor_sync(FULL,hi,1);hi|=__shfl_xor_sync(FULL,hi,2);
        e[plane]=t==0?lo:hi;
    }
}
__device__ __forceinline__ void store_y(float *y,int dim,int feature,const float (&d)[4]){
    int lane=threadIdx.x%32,g=lane/4,t=lane%4;
    #pragma unroll
    for(int i=0;i<4;++i){int col=feature+2*t+i%2;
        if(col<dim)y[(g+8*(i/2))*dim+col]=d[i];
    }
}
__device__ __forceinline__ unsigned xpair(const __half *x,int dim,int row,int col){
    if(col>=dim)return 0;
    return unsigned(__half_as_ushort(x[row*dim+col])) |
           (unsigned(__half_as_ushort(x[(row+1)*dim+col]))<<16);
}
__device__ __forceinline__ void sparse_consume(const float (&v)[4][4],const __half *x,float *y,int dim){
    int lane=threadIdx.x%32,g=lane/4,t=lane%4;
    unsigned a[2][4],e[2];bool two;
    sparse_pack(v,a,e,two);
    for(int f=0;f<dim;f+=8){
        unsigned b[4];
        #pragma unroll
        for(int j=0;j<4;++j)b[j]=xpair(x,dim,8*j+2*t,f+g);
        float d[4]={0,0,0,0};smma(d,a[0],b,e[0]);
        if(two)smma(d,a[1],b,e[1]);
        store_y(y,dim,f,d);
    }
}

// Mode 4 is the producer part of the split, explicit-materialization control.
template<int Mode>
__global__ void sparse_path(const unsigned *bits,const float *weights,const __half *x,
                            float *y,float *intermediate,unsigned char *qout,int tiles,int words,int dim){
    int tile=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
    if(tile>=tiles)return;
    float v[4][4];
    sparse_coefficients<Mode>(bits+tile*48*words,words,weights+tile*512,v,qout?qout+tile*512:nullptr);
    if constexpr(Mode==4){
        int g=(threadIdx.x%32)/4,t=threadIdx.x%4;
        #pragma unroll
        for(int r=0;r<4;++r){
            #pragma unroll
            for(int j=0;j<4;++j)intermediate[tile*512+(g+8*(r%2))*32+16*(r/2)+4*t+j]=v[r][j];
        }
    }else sparse_consume(v,x+tile*32*dim,y+tile*16*dim,dim);
}
__global__ void split_consumer(const float *c,const __half *x,float *y,int tiles,int dim){
    int tile=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
    if(tile>=tiles)return;
    int g=(threadIdx.x%32)/4,t=threadIdx.x%4;float v[4][4];
    #pragma unroll
    for(int r=0;r<4;++r){
        #pragma unroll
        for(int j=0;j<4;++j)v[r][j]=c[tile*512+(g+8*(r%2))*32+16*(r/2)+4*t+j];
    }
    sparse_consume(v,x+tile*32*dim,y+tile*16*dim,dim);
}
__global__ void dense_path(const unsigned *bits,const float *weights,const __half *x,
                          float *y,unsigned char *qout,int tiles,int words,int dim){
    int tile=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
    if(tile>=tiles)return;
    int lane=threadIdx.x%32,g=lane/4,t=lane%4;unsigned m[4],a[2][4];
    make_masks<false>(bits+tile*48*words,words,m);
    #pragma unroll
    for(int p=0;p<4;++p){
        #pragma unroll
        for(int rh=0;rh<2;++rh){
            float val[2];
            #pragma unroll
            for(int j=0;j<2;++j){int r=g+8*rh,z=8*p+2*t+j;bool q=(m[p]>>(2*rh+j))&1;
                if(qout)qout[tile*512+r*32+z]=q;
                val[j]=q?weights[tile*512+r*32+z]:0.0f;
            }
            a[p/2][2*(p%2)+rh]=hpair(val[0],val[1]);
        }
    }
    const __half *tx=x+tile*32*dim;
    for(int f=0;f<dim;f+=8){float d[4]={0,0,0,0};
        #pragma unroll
        for(int h=0;h<2;++h){unsigned b[2]={xpair(tx,dim,16*h+2*t,f+g),xpair(tx,dim,16*h+8+2*t,f+g)};dmma(d,a[h],b);}
        store_y(y+tile*16*dim,dim,f,d);
    }
}
__global__ void pack_bytes(const unsigned char *raw,unsigned *packed,int total_words){
    int word=blockIdx.x*(blockDim.x/32)+threadIdx.x/32,lane=threadIdx.x%32;
    if(word>=total_words)return;
    unsigned value=__ballot_sync(FULL,raw[word*32+lane]!=0);
    if(lane==0)packed[word]=value;
}

struct Data {
    int tiles,k,words,dim;
    std::vector<unsigned> bits;
    std::vector<unsigned char> raw,q;
    std::vector<float> w,expected;
    std::vector<__half> x;
    unsigned *db=nullptr;unsigned char *dr=nullptr,*dq=nullptr;
    float *dw=nullptr,*dy=nullptr,*dc=nullptr;__half *dx=nullptr;
    Data(int count,int kval,int d,int type,unsigned seed,bool identity=false):tiles(count),k(kval),words(std::max(4,((kval+127)/128)*4)),dim(d),
      bits(size_t(count)*48*words),raw(bits.size()*32),q(size_t(count)*512),w(size_t(count)*512),expected(size_t(count)*16*d),x(size_t(count)*32*d){
        std::mt19937 gen(seed);
        auto bit=[&](int tile,int row,int z){bits[(size_t(tile)*48+row)*words+z/32]|=1u<<(z%32);};
        for(int tile=0;tile<count;++tile){
            if(type==0){}
            else if(type==1 || type==2){
                for(int r=0;r<16;++r)if(r<k)bit(tile,r,r);
                for(int z=0;z<32;++z)for(int r=0;r<16 && r<k;++r){
                    unsigned mask=type==2?15u:unsigned((tile+r+z/4)%16);
                    if(mask&(1u<<(z%4)))bit(tile,16+z,r);
                }
            }else if(type==3){
                for(int r=0;r<48;++r)for(int z=0;z<k;++z)if(gen()%100<3)bit(tile,r,z);
            }else if(type==4){
                for(int r=0;r<16;++r)if(r<k)bit(tile,r,r);
                constexpr unsigned patterns[6]={3,5,9,6,10,12};
                for(int z=0;z<32;++z)for(int r=0;r<16 && r<k;++r)
                    if(patterns[(tile+r+z/4)%6]&(1u<<(z%4)))bit(tile,16+z,r);
            }else if(k>0){for(int r=0;r<48;++r)bit(tile,r,k-1);}
            for(int r=0;r<16;++r)for(int z=0;z<32;++z){
                bool yes=false;for(int j=0;j<words;++j)yes|=(bits[(size_t(tile)*48+r)*words+j]&bits[(size_t(tile)*48+16+z)*words+j])!=0;
                q[tile*512+r*32+z]=yes;
                int value=int(gen()%9)-4;
                if(type==4 || type==2)value=(gen()%2?-1:1)*(1+gen()%4);
                w[tile*512+r*32+z]=float(value);
            }
            for(int z=0;z<32;++z)for(int f=0;f<d;++f)
                x[(size_t(tile)*32+z)*d+f]=__float2half_rn(identity?float(z==f):float(int(gen()%17)-8)*0.25f);
            for(int r=0;r<16;++r)for(int f=0;f<d;++f){double value=0;
                for(int z=0;z<32;++z)value+=double(q[tile*512+r*32+z])*w[tile*512+r*32+z]*__half2float(x[(size_t(tile)*32+z)*d+f]);
                expected[(size_t(tile)*16+r)*d+f]=float(value);
            }
        }
        for(size_t j=0;j<bits.size();++j)for(int s=0;s<32;++s)raw[j*32+s]=(bits[j]>>s)&1;
        CK(cudaMalloc(&db,bits.size()*4));CK(cudaMalloc(&dr,raw.size()));CK(cudaMalloc(&dq,q.size()));
        CK(cudaMalloc(&dw,w.size()*4));CK(cudaMalloc(&dx,x.size()*2));CK(cudaMalloc(&dy,expected.size()*4));CK(cudaMalloc(&dc,w.size()*4));
        CK(cudaMemcpy(db,bits.data(),bits.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dr,raw.data(),raw.size(),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dw,w.data(),w.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dx,x.data(),x.size()*2,cudaMemcpyHostToDevice));
    }
    ~Data(){cudaFree(db);cudaFree(dr);cudaFree(dq);cudaFree(dw);cudaFree(dx);cudaFree(dy);cudaFree(dc);}
};
const char *names[]={"natural_sparse","aligned_sparse","natural_dense","simt_sparse","aligned_split"};
std::string archive_dir;
void launch(Data &a,int mode,bool pack,bool debug=false,cudaStream_t stream=0){
    if(pack)pack_bytes<<<(a.bits.size()+7)/8,256,0,stream>>>(a.dr,a.db,int(a.bits.size()));
    int blocks=(a.tiles+3)/4;unsigned char *q=debug?a.dq:nullptr;
    if(mode==0)sparse_path<0><<<blocks,128,0,stream>>>(a.db,a.dw,a.dx,a.dy,a.dc,q,a.tiles,a.words,a.dim);
    else if(mode==1)sparse_path<1><<<blocks,128,0,stream>>>(a.db,a.dw,a.dx,a.dy,a.dc,q,a.tiles,a.words,a.dim);
    else if(mode==2)dense_path<<<blocks,128,0,stream>>>(a.db,a.dw,a.dx,a.dy,q,a.tiles,a.words,a.dim);
    else if(mode==3)sparse_path<3><<<blocks,128,0,stream>>>(a.db,a.dw,a.dx,a.dy,a.dc,q,a.tiles,a.words,a.dim);
    else{
        sparse_path<4><<<blocks,128,0,stream>>>(a.db,a.dw,a.dx,a.dy,a.dc,q,a.tiles,a.words,a.dim);
        split_consumer<<<blocks,128,0,stream>>>(a.dc,a.dx,a.dy,a.tiles,a.dim);
    }
    CK(cudaGetLastError());
}
bool check(Data &a,int mode,bool pack,int id,cudaStream_t stream=0){
    CK(cudaMemset(a.dy,0xff,a.expected.size()*4));CK(cudaMemset(a.dq,0xff,a.q.size()));
    launch(a,mode,pack,true,stream);CK(cudaStreamSynchronize(stream));
    std::vector<float> y(a.expected.size());std::vector<unsigned char> q(a.q.size());
    CK(cudaMemcpy(y.data(),a.dy,y.size()*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(q.data(),a.dq,q.size(),cudaMemcpyDeviceToHost));
    double maxerr=0;size_t bad=0;for(size_t j=0;j<y.size();++j){
        double e=std::abs(double(y[j])-a.expected[j]);maxerr=std::max(maxerr,e);
        if(!std::isfinite(y[j]) || e>1e-5){if(bad++<3)fprintf(stderr,"FAIL id=%d %s output=%zu got=%g want=%g\n",id,names[mode],j,y[j],a.expected[j]);}
    }
    size_t qbad=0;for(size_t j=0;j<q.size();++j)qbad+=q[j]!=a.q[j];
    size_t pbad=0;if(pack){std::vector<unsigned> b(a.bits.size());CK(cudaMemcpy(b.data(),a.db,b.size()*4,cudaMemcpyDeviceToHost));for(size_t j=0;j<b.size();++j)pbad+=b[j]!=a.bits[j];}
    if(!archive_dir.empty()){
        std::ofstream out(archive_dir+"/case_"+std::to_string(id)+".bin",std::ios::binary);
        unsigned header[6]={0x4c41594fu,unsigned(a.tiles),unsigned(a.k),unsigned(a.words),unsigned(a.dim),unsigned(mode)};
        out.write((char*)header,sizeof(header));out.write((char*)a.bits.data(),a.bits.size()*4);
        out.write((char*)a.w.data(),a.w.size()*4);out.write((char*)a.x.data(),a.x.size()*2);
        out.write((char*)q.data(),q.size());out.write((char*)y.data(),y.size()*4);
        if(!out){fprintf(stderr,"Archive write failed\n");exit(4);}
    }
    printf("check,%d,%s,%d,%d,%d,%zu,%zu,%zu,%.9g\n",id,names[mode],a.tiles,a.k,a.dim,bad,qbad,pbad,maxerr);
    return bad==0&&qbad==0&&pbad==0;
}
int correctness(bool small){
    int id=0,fail=0;
    if(small){
        for(int type=0;type<6;++type){Data a(5,129,9,type,1200+type);for(int mode=0;mode<5;++mode)fail+=!check(a,mode,true,id++);}
    }else{
        for(int type=0;type<6;++type)for(int k:{0,1,15,16,31,127,128,129,257}){
            Data a(17,k,32,type,20260908+id,true);for(int mode=0;mode<5;++mode)fail+=!check(a,mode,true,id++);
        }
        for(int d:{1,7,8,9,16,31,33,100}){Data a(7,512,d,3,901+d);for(int mode=0;mode<5;++mode)fail+=!check(a,mode,false,id++);}
        cudaStream_t streams[2];CK(cudaStreamCreate(&streams[0]));CK(cudaStreamCreate(&streams[1]));
        for(int i=0;i<40;++i){Data a(1+i%9,128+(i%2),1+i%33,i%6,6100+i);fail+=!check(a,i%5,i%2,id++,streams[i%2]);}
        CK(cudaStreamDestroy(streams[0]));CK(cudaStreamDestroy(streams[1]));
    }
    printf("SUMMARY checks=%d failures=%d\n",id,fail);return fail?3:0;
}
int benchmark(){
    std::mt19937 order_rng(20260908);cudaEvent_t begin,end;CK(cudaEventCreate(&begin));CK(cudaEventCreate(&end));
    printf("class,k,dim,tiles,packing,round,mode,us_per_call\n");
    for(int type:{4,2})for(int k:{128,512})for(int d:{8,32,100})for(int tiles:{32,256,2048}){
        Data a(tiles,k,d,type,7300+type+k+d+tiles);
        for(int mode=0;mode<5;++mode)for(int j=0;j<3;++j)launch(a,mode,true);
        CK(cudaDeviceSynchronize());
        for(int pack=0;pack<2;++pack)for(int round=0;round<7;++round){
            std::array<int,5> order={0,1,2,3,4};std::shuffle(order.begin(),order.end(),order_rng);
            for(int mode:order){constexpr int repeats=20;
                CK(cudaEventRecord(begin));for(int j=0;j<repeats;++j)launch(a,mode,pack);
                CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));float ms;CK(cudaEventElapsedTime(&ms,begin,end));
                printf("%s,%d,%d,%d,%d,%d,%s,%.9g\n",type==4?"two_of_four":"dense",k,d,tiles,pack,round,names[mode],double(ms)*1000/repeats);
            }
        }
        fflush(stdout);
    }
    CK(cudaEventDestroy(begin));CK(cudaEventDestroy(end));return 0;
}
int main(int argc,char **argv){
    CK(cudaSetDevice(0));cudaDeviceProp p;CK(cudaGetDeviceProperties(&p,0));
    fprintf(stderr,"DEVICE name=%s sm=%d%d runtime=%d\n",p.name,p.major,p.minor,CUDART_VERSION);
    std::string mode=argc>1?argv[1]:"check";
    if(const char *p=getenv("LAYOUT_ARCHIVE_DIR"))archive_dir=p;
    if(mode=="bench")return benchmark();
    return correctness(mode=="small");
}
