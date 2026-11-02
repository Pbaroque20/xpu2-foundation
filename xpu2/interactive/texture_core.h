// Private bounded software texture filter. CPU/CUDA share this source.
// Filtering algorithms adapted from OpenImageIO 3.1.17.0 texturesys.cpp,
// texture_pvt.h and fmath.h. Copyright Contributors to the OpenImageIO project.
// SPDX-License-Identifier: Apache-2.0. See vendor/oiio/LICENSE.md.
#pragma once
#include "surface_core.h"
#include "bc6h_core.h"
#if !defined(__CUDA_ARCH__)
#include <atomic>
#include <thread>
#endif
namespace xpu2::interactive::tex {
using surface::V3;
#if !defined(__CUDA_ARCH__)
// A host thread reading a tile that may be let go under memory pressure (host_texture.h's relieve) holds a Guard while
// it does: a tile is freed only once no reader that could have seen it remains. A reader stands in its own slot, the
// frame's epoch as it entered; a page is cleared, the epoch moves on, and those who entered before it are waited for.
namespace spill {
constexpr unsigned Slots=512;
struct alignas(64) Reader {std::atomic<unsigned long long> in{0};std::atomic<bool> taken{false};};
inline Reader readers[Slots];
inline std::atomic<unsigned long long> epoch{1};
struct Seat {
 Reader* reader=nullptr;
 Seat(){for(;;std::this_thread::yield())for(auto& r:readers){bool free=false;if(r.taken.compare_exchange_strong(free,true)){reader=&r;return;}}}
 ~Seat(){reader->taken.store(false);}
};
inline Reader& mine(){thread_local Seat seat;return *seat.reader;}
struct Guard {
 Reader& r;
 Guard():r(mine()){r.in.store(epoch.load(std::memory_order_relaxed),std::memory_order_seq_cst);}
 ~Guard(){r.in.store(0,std::memory_order_release);}
 Guard(const Guard&)=delete;
};
}
#endif
constexpr unsigned Tile=32,MaxLevels=16;
enum Status:unsigned {Invalid=0,Ready=1,Miss=2};
// tileBase numbers this level's tiles in a scene-wide tile space, so a miss can
// name the tile it needs (async_gpu.h); zero where nothing listens for misses.
// format: 0, three floats a texel; 1, three bytes, an 8-bit file's values (texel decodes each as OpenImageIO does, v
// times 1/255: host_texture.h checks it is so before it stores bytes), a quarter of the memory.
// load, loader: on the host, a level whose tiles are decoded when first read (host_texture.h's Pyramid, level 0): a page
// not there yet is loaded by load(loader, level, page) and read then. Never on the device (its own descriptors leave it
// null; the GPU's misses go to its tile cache).
struct Mip {unsigned width=0,height=0;const V3* const* pages=nullptr;unsigned tileBase=0;unsigned format=0;
 void (*load)(const void*,unsigned,unsigned)=nullptr;const void* loader=nullptr;};
// Bytes a texel of a level of this format.
X2_HD inline unsigned texelBytes(unsigned format){return format==1?3u:format==2?1u:unsigned(sizeof(V3));}
// Bytes a tile of count texels takes. Format 2, BC6H (texture overflow mode; bc6h_core.h): a whole tile's blocks, whatever its size,
// its texels padded to 32 x 32.
X2_HD inline unsigned tileBytes(unsigned format,unsigned count){return format==2?bc6h::TileBytes:count*texelBytes(format);}
// A sink receives the scene-wide id of each missing tile, and of each tracked tile
// that is read (for the fine-detail cache's eviction order). NoSink ignores both.
// Partial shading: where the sink takes coarse(), a texel whose tile is missing is read from its parent a level up
// where that level is resident (the GPU's levels 1 and up always are: the half-resolution picture and smaller), by
// the sink out of line (parent); the sample is then provisional and is rendered again in full (async_gpu.cu,
// async_render.h).
struct Image;
struct NoSink {X2_HD void operator()(unsigned) const {} X2_HD void touch(unsigned) const {} X2_HD bool coarse() const {return false;}
 X2_HD unsigned parent(const Image&,unsigned,unsigned,unsigned,V3&) const {return 2u;}};
struct Image {unsigned count=0;Mip levels[MaxLevels]{};};
// mode: 0 anisotropic (MoonRay's primary paths), 1 trilinear mips of the closest texels (glossy and mirror
// paths), 2 one mip level's closest texel (diffuse paths).
// swapCross: MoonRay XPU's lookups (CPP_oiioTexture and CPP_oiioUdimTexture, which its vector shading calls) pass
// OIIO dsdx, dsdy, dtdx, dtdy where it takes dsdx, dtdx, dsdy, dtdy: the footprint transposed. Its scalar path, which
// displacement takes, passes them in order.
struct Binding {const Image* image=nullptr;float sx=1,sy=1,ox=0,oy=0,bias=0;unsigned wrap=1,gamma=0,mode=0,swapCross=0;};
struct Maps {Binding diffuse,emission;};
struct Footprint {float u,v,dudx,dvdx,dudy,dvdy;};
X2_HD inline float clamp(float x,float lo,float hi){return fminf(fmaxf(x,lo),hi);}
// OIIO host approximations also used on CUDA (not CUDA fast intrinsics).
X2_HD inline float atan(float y,float x){float a=fabsf(x),b=fabsf(y);float k=b==0?0.f:(a==b?1.f:(b>a?a/b:b/a));float s=1.f-(1.f-k),t=s*s;
 float r=s*(0.430165678f*t+1.f)/((0.0579354987f*t+0.763007998f)*t+1.f);if(b>a)r=1.5707963267948966f-r;if(copysignf(1.f,x)<0)r=surface::Pi-r;return copysignf(r,y);}
X2_HD inline float exp(float a){float x=clamp(a*1.4426950408889634f,-126,126);int m=int(x);x-=m;x=1.f-(1.f-x);float r=1.33336498402e-3f;
 r=x*r+9.810352697968e-3f;r=x*r+5.551834031939e-2f;r=x*r+0.2401793301105f;r=x*r+0.693144857883f;r=x*r+1.f;return ldexpf(r,m);}
X2_HD inline int address(int x,int n,bool wrap){if(wrap){int v=x%n;return v<0?v+n:v;}return x<0?0:x>=n?n-1:x;}
// A resident page's texel.
X2_HD inline Status texel(const V3* page,unsigned width,unsigned tx,unsigned x,unsigned y,V3& out,unsigned format=0){
 if(format==2){const unsigned lx=x%Tile,ly=y%Tile;float rgb[3];
  bc6h::texel(reinterpret_cast<const unsigned*>(reinterpret_cast<const unsigned char*>(page)+((ly/4u)*bc6h::BlocksAcross+lx/4u)*bc6h::BlockBytes),(ly%4u)*4u+lx%4u,rgb);
  out=V3{rgb[0],rgb[1],rgb[2]};return Ready;}
 unsigned stride=width-tx*Tile;if(stride>Tile)stride=Tile;const unsigned i=(y%Tile)*stride+x%Tile;
 if(format==1){const unsigned char* b=reinterpret_cast<const unsigned char*>(page)+3u*i;const float k=1.0f/255.0f;
  out=V3{float(b[0])*k,float(b[1])*k,float(b[2])*k};return Ready;}
 out=page[i];return surface::finite(out)?Ready:Invalid;  // negative values as MoonRay keeps them
}
// Partial shading: a texel's parent a level up (texel x, y of level - 1 halved), where that level's page is here.
X2_HD inline Status parentTexel(const Image& image,unsigned level,unsigned x,unsigned y,V3& out){
 const Mip& u=image.levels[level];if(!u.pages||!u.width||!u.height)return Miss;
 const unsigned px=x<u.width?x:u.width-1,py=y<u.height?y:u.height-1,ux=px/Tile;
 const V3* parent=u.pages[(py/Tile)*((u.width+Tile-1)/Tile)+ux];if(!parent)return Miss;
 return texel(parent,u.width,ux,px,py,out,u.format);
}
template<class Sink>X2_HD inline Status fetch(const Image& image,unsigned level,int x,int y,bool wrap,V3& out,Sink& sink){
 const Mip& m=image.levels[level];
 if(!m.width||!m.height)return Invalid;if(!m.pages)return Miss;x=address(x,int(m.width),wrap);y=address(y,int(m.height),wrap);
 unsigned tx=unsigned(x)/Tile,ty=unsigned(y)/Tile,nx=(m.width+Tile-1)/Tile;const V3* page=m.pages[ty*nx+tx];
#if !defined(__CUDA_ARCH__)
 // A host level loaded on demand: its page decoded now, then read as any other, under a guard as it may be let go.
 if(m.load){
  spill::Guard guard;const V3* held=m.pages[ty*nx+tx];
  while(!held){m.load(m.loader,level,ty*nx+tx);held=m.pages[ty*nx+tx];}
  if(m.tileBase)sink.touch(m.tileBase+ty*nx+tx);
  return texel(held,m.width,tx,unsigned(x),unsigned(y),out,m.format);
 }
#endif
 if(!page){
  sink(m.tileBase+ty*nx+tx);
  // Partial shading: the texel's parent a level up, read by the sink out of line (it is rarely needed).
  return sink.coarse()&&level+1<image.count?Status(sink.parent(image,level+1,unsigned(x)/2,unsigned(y)/2,out)):Miss;
 }
 if(m.tileBase)sink.touch(m.tileBase+ty*nx+tx);
 return texel(page,m.width,tx,unsigned(x),unsigned(y),out,m.format);
}
template<class Sink>X2_HD inline Status bilinear(const Image& image,unsigned level,float u,float v,bool wrap,V3& out,Sink& sink){
 const Mip& m=image.levels[level];
 float s=u*m.width-.5f,t=v*m.height-.5f;if(!surface::finite(s)||!surface::finite(t)||fabsf(s)>1.e8f||fabsf(t)>1.e8f)return Invalid;
 int x=int(floorf(s)),y=int(floorf(t));s-=x;t-=y;V3 a,b,c,d;Status q;
 if((q=fetch(image,level,x,y,wrap,a,sink))!=Ready||(q=fetch(image,level,x+1,y,wrap,b,sink))!=Ready||(q=fetch(image,level,x,y+1,wrap,c,sink))!=Ready||(q=fetch(image,level,x+1,y+1,wrap,d,sink))!=Ready)return q;
 out=surface::add(surface::mul(surface::add(surface::mul(a,1-s),surface::mul(b,s)),1-t),surface::mul(surface::add(surface::mul(c,1-s),surface::mul(d,s)),t));return Ready;
}
// sample_closest: the texel nearest (s, t); an exact half-texel tie takes the lower one.
template<class Sink>X2_HD inline Status closest(const Image& image,unsigned level,float u,float v,bool wrap,V3& out,Sink& sink){
 const Mip& m=image.levels[level];
 float s=u*m.width-.5f,t=v*m.height-.5f;if(!surface::finite(s)||!surface::finite(t)||fabsf(s)>1.e8f||fabsf(t)>1.e8f)return Invalid;
 int x=int(floorf(s)),y=int(floorf(t));if(s-x>.5f)++x;if(t-y>.5f)++y;return fetch(image,level,x,y,wrap,out,sink);
}
// texture_lookup_trilinear_mipmap with closest texels (bilinear ones in mode 3, MoonRay's TrilinearIsotropic for
// displacement): adjust_width's clamp of degenerate derivatives, the
// conservative filter width (the larger of s's and t's), compute_miplevels (one level for MipModeOneLevel).
template<class Sink>X2_HD inline Status trilinear(const Binding& b,float s,float t,float dx,float tx,float dy,float ty,V3& out,Sink& sink){
 const float eps=1.0e-8f,eps2=eps*eps;float dxlen2=dx*dx+tx*tx,dylen2=dy*dy+ty*ty;
 if(dxlen2<eps2){if(dylen2<eps2){dx=eps;dy=0;tx=0;ty=eps;}else{float sc=eps/sqrtf(dylen2);dx=ty*sc;tx=-dy*sc;}}
 else if(dylen2<eps2){float sc=eps/sqrtf(dxlen2);dy=-tx*sc;ty=dx*sc;}
 const float filt=fmaxf(fmaxf(fabsf(dx),fabsf(dy)),fmaxf(fabsf(tx),fabsf(ty)));
 const auto& image=*b.image;int levels[2]={-1,-1};float lw[2]={0,0};
 for(unsigned m=0;m<image.count;++m){auto mip=image.levels[m];float w=filt*(mip.width<mip.height?mip.width:mip.height);if(w<=1){levels[0]=int(m)-1;levels[1]=m;lw[1]=clamp(2*w-1,0,1);lw[0]=1-lw[1];break;}}
 if(levels[1]<0){levels[0]=levels[1]=int(image.count)-1;lw[0]=1;lw[1]=0;}
 else if(levels[0]<0){levels[0]=levels[1]=0;lw[0]=1;lw[1]=0;}
 else if(b.mode==2){levels[0]=levels[1];lw[0]=1;lw[1]=0;}
 V3 result{};for(int k=0;k<2;++k){if(lw[k]==0)continue;V3 value;auto status=b.mode==3?bilinear(image,unsigned(levels[k]),s,t,b.wrap!=0,value,sink):closest(image,unsigned(levels[k]),s,t,b.wrap!=0,value,sink);if(status!=Ready)return status;result=surface::add(result,surface::mul(value,lw[k]));}
 if(b.gamma)result={powf(fmaxf(result.x,0),2.2f),powf(fmaxf(result.y,0),2.2f),powf(fmaxf(result.z,0),2.2f)};
 if(!surface::finite(result))return Invalid;out=result;return Ready;
}
// Admitted subset: stored 2D RGB mips, anisotropy 4, bilinear, zero blur,
// periodic/clamp, deterministic mip blending, explicit scale/offset/gamma.
template<class Sink>X2_HD inline Status lookup(const Binding& b,Footprint f,V3& out,Sink& sink){
 if(!b.image||!b.image->count||b.image->count>MaxLevels)return Invalid;
 float s=b.sx*f.u+b.ox,t=1-(b.sy*f.v+b.oy),bias=1+b.bias;
 float dx=f.dudx*b.sx*bias,tx=-f.dvdx*b.sx*bias,dy=f.dudy*b.sy*bias,ty=-f.dvdy*b.sy*bias;
 if(b.swapCross){const float c=tx;tx=dy;dy=c;}
 if(!surface::finite(s)||!surface::finite(t)||!surface::finite(dx)||!surface::finite(tx)||!surface::finite(dy)||!surface::finite(ty))return Invalid;
 // Far coordinates (a UV set tiled a thousand times and more): into the first repeat where the texture repeats, else
 // near its edge, which its clamped addressing reads all the same (MoonRay's OIIO takes any); nearer ones as they are.
 if(fabsf(s)>1024)s=b.wrap?s-floorf(s):clamp(s,-1,2);if(fabsf(t)>1024)t=b.wrap?t-floorf(t):clamp(t,-1,2);
 if(b.mode)return trilinear(b,s,t,dx,tx,dy,ty,out,sink);
 double A=tx*tx+ty*ty,B=-2.0*(dx*tx+dy*ty),C=dx*dx+dy*dy,root=hypot(A-C,B);
 float major=fminf(sqrtf(fmaxf(float((A+C+root)*.5),0.f)),1000.f),minor=fminf(sqrtf(fmaxf(float((A+C-root)*.5),0.f)),1000.f);
 float theta=atan(float(B),float(A-C))*.5f+surface::Pi*.5f;
 float aspect=minor>0?clamp(major/minor,1,1.e6f):(major>0?1.e6f:1.f);
 if(aspect>4){aspect=4;major=.5f*(major+minor*4);minor=major/4;}
 int levels[2]={-1,-1};float lw[2]={0,0};const auto& image=*b.image;
 for(unsigned m=0;m<image.count;++m){auto mip=image.levels[m];float w=minor*(mip.width<mip.height?mip.width:mip.height);if(w<=1){levels[0]=int(m)-1;levels[1]=m;lw[1]=clamp(2*w-1,0,1);lw[0]=1-lw[1];break;}}
 if(levels[1]<0){levels[0]=levels[1]=int(image.count)-1;lw[0]=1;lw[1]=0;}
 else if(levels[0]<0){levels[0]=levels[1]=0;lw[0]=1;lw[1]=0;auto m=image.levels[0];unsigned r=m.width>m.height?m.width:m.height;if(minor*r<.5f)aspect=clamp(major*r*2,1,4);}
 int n=int(2*aspect-1);if(n<1)n=1;if(n>7)return Invalid;float weight[7],pos[7];
 float LL=2*(major-minor),sm=cosf(theta)*LL*.5f,tm=sinf(theta)*LL*.5f;
 if(n==1){weight[0]=1;pos[0]=0;}else if(n==2){weight[0]=weight[1]=.5f;pos[0]=-.5f;pos[1]=.5f;}else{float sum=0,scale=major/LL;for(int i=0;i<n;++i){pos[i]=2*(i+.5f)*(1.f/n)-1;float x=pos[i]*scale;weight[i]=exp(-2*x*x);sum+=weight[i];}for(int i=0;i<n;++i)weight[i]/=sum;}
 V3 result{};for(int k=0;k<2;++k){if(lw[k]==0)continue;V3 acc{};for(int i=0;i<n;++i){V3 value;auto status=bilinear(image,unsigned(levels[k]),s+pos[i]*sm,t+pos[i]*tm,b.wrap!=0,value,sink);if(status!=Ready)return status;acc=surface::add(acc,surface::mul(value,weight[i]));}result=surface::add(result,surface::mul(acc,lw[k]));}
 if(b.gamma)result={powf(fmaxf(result.x,0),2.2f),powf(fmaxf(result.y,0),2.2f),powf(fmaxf(result.z,0),2.2f)};
 if(!surface::finite(result))return Invalid;out=result;return Ready;
}
X2_HD inline Status lookup(const Binding& b,Footprint f,V3& out){NoSink sink;return lookup(b,f,out,sink);}
// Shared planar hit derivatives; use full-plan SPP, not a progressive batch size.
X2_HD inline bool footprint(surface::V3 a,surface::V3 b,surface::V3 c,const surface::Ray& ray,float u,float v,unsigned spp,Footprint& out,bool spatial=false){
 using namespace surface;V3 e1=add(b,mul(a,-1)),e2=add(c,mul(a,-1));
 if(spatial){V3 n=cross(e1,e2);unsigned axis=fabsf(n.x)>fabsf(n.y)?0:1;if(fabsf(n.z)>(axis==0?fabsf(n.x):fabsf(n.y)))axis=2;
  float e1a=axis==0?e1.y:axis==1?e1.z:e1.x,e1b=axis==0?e1.z:axis==1?e1.x:e1.y,e2a=axis==0?e2.y:axis==1?e2.z:e2.x,e2b=axis==0?e2.z:axis==1?e2.x:e2.y;
  float det=e1a*e2b-e1b*e2a,den=dot(n,ray.direction);if(!finite(det)||det==0||!finite(den)||den==0||!spp)return false;float plane=dot(n,add(a,mul(ray.origin,-1)));V3 primary=mul(ray.direction,plane/den);
  float scale=1.f/fminf(sqrtf(float(spp)),2.f),uv[4];V3 directions[2]={ray.dx,ray.dy};for(int i=0;i<2;++i){V3 d=directions[i];if(scale<1&&!normalize(add(ray.direction,mul(add(d,mul(ray.direction,-1)),scale)),d))return false;float denominator=dot(n,d);if(!finite(denominator)||denominator==0)return false;V3 delta=add(mul(d,plane/denominator),mul(primary,-1));
   float pa=axis==0?delta.y:axis==1?delta.z:delta.x,pb=axis==0?delta.z:axis==1?delta.x:delta.y,x=(pa*e2b-pb*e2a)/det,y=(e1a*pb-e1b*pa)/det;uv[i*2]=x+y;uv[i*2+1]=y;}
  out={u+v,v,uv[0],uv[1],uv[2],uv[3]};return true;
 }
 float det=e1.x*e2.y-e1.y*e2.x;if(!finite(det)||det<=0||!spp)return false;
 float scale=1.f/fminf(sqrtf(float(spp)),2.f);float uv[4];V3 directions[2]={ray.dx,ray.dy};for(int i=0;i<2;++i){auto d=directions[i];if(scale<1&&!normalize(add(ray.direction,mul(add(d,mul(ray.direction,-1)),scale)),d))return false;if(d.z>=0||ray.direction.z>=0)return false;
 auto p=add(mul(d,-ray.origin.z/d.z),mul(ray.direction,ray.origin.z/ray.direction.z));float x=(p.x*e2.y-p.y*e2.x)/det,y=(e1.x*p.y-e1.y*p.x)/det;uv[i*2]=x+y;uv[i*2+1]=y;}
 out={u+v,v,uv[0],uv[1],uv[2],uv[3]};return true;
}
}
