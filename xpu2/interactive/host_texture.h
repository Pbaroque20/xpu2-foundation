// Private Windows host-only ImageMap subset. No GPU level-0 copy or hardware filtering.
#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <OpenImageIO/texture.h>
#include <OpenImageIO/imageio.h>
#include <filesystem>
#include <memory>
#include <atomic>
#include <mutex>
#include <cstdint>
#include <cstring>
#include <string>
#include <deque>
#include <thread>
#include <chrono>
#include <cstdio>
#include <psapi.h>
#include <OpenImageIO/imagecache.h>
#include <stdexcept>
#include "surface_core.h"
#include "texture_core.h"
#include "multi_device.h"
#include <vector>
// ISPC Texture Compressor's BC6H encoder (xpu2_bc6h.cc), weak: a program built without it just does not compress.
extern "C" {
struct rgba_surface;
struct bc6h_enc_settings;
void CompressBlocksBC6H(const rgba_surface* src,uint8_t* dst,bc6h_enc_settings* settings) __attribute__((weak));
void GetProfile_bc6h_fast(bc6h_enc_settings* settings) __attribute__((weak));
}
namespace xpu2::interactive::texture {
inline void need(bool v,const char* s){if(!v)throw std::runtime_error(s);}
inline std::shared_ptr<OIIO::TextureSystem> system(){
 auto s=OIIO::TextureSystem::create(false);need(bool(s),"OIIO texture system unavailable");
 for(auto n:{"automip","autotile","accept_untiled","accept_unmipped"})need(s->attribute(n,0),"OIIO option");
 need(s->attribute("gray_to_rgb",1)&&s->attribute("max_memory_MB",1024.0f),"OIIO budget option");return s;
}
inline void admittedPath(const std::filesystem::path& path,const std::filesystem::path& root){
 need(path.is_absolute()&&root.is_absolute()&&path.extension()==".tx","Frozen absolute TX required");
 auto rel=std::filesystem::weakly_canonical(path).lexically_relative(std::filesystem::canonical(root));
 need(!rel.empty()&&!rel.is_absolute()&&*rel.begin()!="..","Texture outside frozen asset root");
}
// MOONRAY_MODO_OPAQUE_ALPHA (BasicTexture, UdimTexture): a missing channel is filled with one (opaque), else with
// OpenImageIO's fill, zero. Read once, as MoonRay reads it.
inline bool modoOpaqueAlpha(){static const bool on=[]{const char* v=std::getenv("MOONRAY_MODO_OPAQUE_ALPHA");return v&&v[0]=='1';}();return on;}
// A texel as MoonRay's TextureSampler looks up four channels (gray_to_rgb): a grey image's channel in red, green and
// blue, a grey and alpha image's (alpha channel 1) grey in all three and its alpha fourth, and any channel the file
// lacks the fill.
inline void texelChannels(const float* d,int n,int alphaChannel,surface::V3& c,float& a){
 const float fill=modoOpaqueAlpha()?1.0f:0.0f;
 if(n==1){c={d[0],d[0],d[0]};a=fill;}
 else if(n==2){if(alphaChannel==1){c={d[0],d[0],d[0]};a=d[1];}else{c={d[0],d[1],fill};a=fill;}}
 else{c={d[0],d[1],d[2]};a=n>3?d[3]:fill;}}
// alpha: the texture's alpha as each texel's three channels (ImageMap's alpha_only; texelChannels' where the file has
// none).
// Whether an 8-bit value can be kept as its byte: OpenImageIO's conversion of each of the 256 to float is the byte times
// 1/255 (tex::texel's decoding), and 0 and 1 (a missing channel's fill) are bytes 0 and 255. Checked once.
inline bool bytesExact(){static const bool exact=[]{unsigned char in[256];float out[256];for(int i=0;i<256;++i)in[i]=(unsigned char)i;
  if(!OIIO::convert_pixel_values(OIIO::TypeDesc::UINT8,in,OIIO::TypeDesc::FLOAT,out,256))return false;
  for(int i=0;i<256;++i)if(out[i]!=float(i)*(1.0f/255.0f))return false;return 255.0f*(1.0f/255.0f)==1.0f;}();return exact;}
// A value as a half float, rounded to the nearest (ties to even), where it is within their range (halfFits).
inline bool halfFits(float v){return surface::finite(v)&&std::fabs(v)<=65504.0f;}
inline uint16_t halfRounded(float v){uint32_t u;std::memcpy(&u,&v,4);const uint16_t sign=uint16_t((u>>16)&0x8000u);u&=0x7FFFFFFFu;
 if(u>=0x38800000u)return uint16_t(sign|((u+0x0FFFu+((u>>13)&1u)-0x38000000u)>>13));
 return uint16_t(sign|unsigned(std::nearbyint(std::fabs(v)*16777216.0f)));}
// Where it is one exactly (a half-float file's values, decoded by OpenImageIO, are).
inline bool halfExactly(float v,uint16_t& h){if(!halfFits(v))return false;h=halfRounded(v);return tex::halfValue(h)==v;}
// One texel as a level of this format holds it (tex::texel decodes it again to the same values), checked so: a value a format
// cannot hold exactly is an error, never a rounding, but for a float file's values in half floats (round).
inline void storeTexel(const surface::V3& c,unsigned format,unsigned char* to,bool round=false){
 if(tex::oneChannel(format))need(c.x==c.y&&c.y==c.z,"Grey texel not grey");
 const float v[3]={c.x,c.y,c.z};const int n=tex::oneChannel(format)?1:3;
 if(format==1||format==5){for(int k=0;k<n;++k){const long b=std::lround(v[k]*255.0f);need(b>=0&&b<=255&&float(b)*(1.0f/255.0f)==v[k],"8-bit texel not its byte");to[k]=(unsigned char)b;}}
 else if(format==3||format==6){for(int k=0;k<n;++k){uint16_t h=0;
   if(round){need(halfFits(v[k]),"Texel past half floats' range");h=halfRounded(v[k]);}else need(halfExactly(v[k],h),"Half-float texel not a half float");
   std::memcpy(to+2*k,&h,2);}}
 else{need(surface::finite(c),"Invalid TX sample");std::memcpy(to,v,size_t(n)*4);}}
// A texel as it reads back from half floats rounded so.
inline surface::V3 halfRoundedTexel(const surface::V3& c){return {tex::halfValue(halfRounded(c.x)),tex::halfValue(halfRounded(c.y)),tex::halfValue(halfRounded(c.z))};}
// The image cache the finest levels' tiles are read through (thread-safe; half a gigabyte of the files' own tiles).
inline std::shared_ptr<OIIO::ImageCache> lazyCache(){static std::shared_ptr<OIIO::ImageCache> c=[]{auto s=OIIO::ImageCache::create(true);
  s->attribute("max_memory_MB",512.0f);return s;}();return c;}
// Whether the finest level is read when first used (XPU2_SESSION_EAGER_TEXTURES=1: every level decoded at once).
inline bool lazyTextures(){static const bool on=[]{const char* v=std::getenv("XPU2_SESSION_EAGER_TEXTURES");return !(v&&v[0]=='1');}();return on;}
// Texture overflow mode: the largest side a texture has for the rest of the session (0: as stored), where a scene's load ran
// out of memory on the machine or the card: finer levels are neither decoded nor sent. The session halves it until the
// scene loads (nextOverflowSide) and says so in the preview.
inline std::atomic<unsigned>& overflowSide(){static std::atomic<unsigned> side{0};return side;}
inline std::atomic<unsigned>& largestTexture(){static std::atomic<unsigned> side{0};return side;}
constexpr unsigned SmallestOverflowSide=256;
// Its first step, where the encoder is there: every texture as BC6H, a byte a texel, at the size it has: lossy, but the detail stays.
inline std::atomic<bool>& overflowCompress(){static std::atomic<bool> on{false};return on;}
inline bool compressorThere(){return CompressBlocksBC6H&&GetProfile_bc6h_fast;}
inline bool nextOverflowSide(){
 if(!overflowCompress().load()&&compressorThere()&&largestTexture().load()){overflowCompress().store(true);return true;}
 const unsigned now=overflowSide().load(),from=now?now:largestTexture().load();
 if(from<=SmallestOverflowSide)return false;
 unsigned next=SmallestOverflowSide;while(next*2<from)next*=2;
 overflowSide().store(next);return true;}
inline bool canOverflow(){const unsigned now=overflowSide().load();
 return (!overflowCompress().load()&&compressorThere()&&largestTexture().load())||(now?now:largestTexture().load())>SmallestOverflowSide;}
inline std::string overflowWords(){const unsigned side=overflowSide().load();std::string words;
 if(overflowCompress().load())words="compressed";
 if(side)words+=(words.empty()?"limited to ":" and limited to ")+std::to_string(side)+" pixels a side";
 return words;}
// A tile's texels (tw x th, any size up to a tile) as BC6H blocks, padded to a whole tile by its edge. Unsigned: negative values come
// out as 0 (a map with them stays as it is: Pyramid::compressed).
inline void encodeTile(const surface::V3* rgb,unsigned tw,unsigned th,unsigned char* out){
 struct Surface{uint8_t* ptr;uint32_t width,height,stride;};
 struct Settings{bool slow_mode,fast_mode;int refineIterations_1p,refineIterations_2p,fastSkipTreshold;};
 auto half=[](float f)->uint16_t{if(!(f>0))return 0;uint32_t u;std::memcpy(&u,&f,4);const int e=int((u>>23)&255)-112;const uint32_t m=(u>>13)&1023;
  if(e>=31)return 0x7BFF;if(e<=0)return uint16_t(f*16777216.0f);return uint16_t((uint32_t(e)<<10)|m);};
 uint16_t texels[tex::Tile*tex::Tile*4];
 for(unsigned y=0;y<tex::Tile;++y)for(unsigned x=0;x<tex::Tile;++x){const surface::V3& c=rgb[size_t(std::min(y,th-1))*tw+std::min(x,tw-1)];
  uint16_t* to=&texels[(y*tex::Tile+x)*4];to[0]=half(c.x);to[1]=half(c.y);to[2]=half(c.z);to[3]=half(1.0f);}
 Settings settings;GetProfile_bc6h_fast(reinterpret_cast<bc6h_enc_settings*>(&settings));
 Surface surface{reinterpret_cast<uint8_t*>(texels),tex::Tile,tex::Tile,tex::Tile*8};
 CompressBlocksBC6H(reinterpret_cast<const rgba_surface*>(&surface),out,reinterpret_cast<bc6h_enc_settings*>(&settings));
}
// A decoded tile's block, in the heap of the node that decoded it (multi_device.h): each node's tiles in its own memory,
// read there by its workers without crossing to the other socket. With one node, the process's heap, as before.
struct NodeFree {HANDLE heap=nullptr;void operator()(unsigned char* p)const{if(p)HeapFree(heap,0,p);}};
using Block=std::unique_ptr<unsigned char[],NodeFree>;
constexpr unsigned MaxNodes=64;
inline HANDLE nodeHeap(unsigned node){
 if(!multi::numa()||node>=MaxNodes)return GetProcessHeap();
 static std::atomic<HANDLE> heaps[MaxNodes]{};
 HANDLE h=heaps[node].load(std::memory_order_acquire);if(h)return h;
 HANDLE made=HeapCreate(0,0,0);if(!made)return GetProcessHeap();
 HANDLE none=nullptr;if(!heaps[node].compare_exchange_strong(none,made)){HeapDestroy(made);return none;}
 return made;}
inline Block nodeBlock(size_t size,unsigned node){HANDLE h=nodeHeap(node);auto* p=static_cast<unsigned char*>(HeapAlloc(h,0,size));if(!p)throw std::bad_alloc();return Block(p,NodeFree{h});}
// The decoded tiles each node holds, in bytes.
inline std::atomic<size_t>& nodeTileBytes(unsigned node){static std::atomic<size_t> held[MaxNodes]{};return held[std::min(node,MaxNodes-1)];}
// Set by the pressure watch where tiles were let go, all of them, and memory did not come back (RAM still short).
inline std::atomic<bool>& pressureShort(){static std::atomic<bool> s{false};return s;}
struct Pyramid;
// Every texture whose tiles can be let go; a list never freed, as the pressure watch outlives what ends at exit.
inline std::mutex& spillLock(){static auto* m=new std::mutex;return *m;}
inline std::vector<Pyramid*>& spillList(){static auto* l=new std::vector<Pyramid*>;return *l;}
struct Pyramid {
 tex::Image image;
 // Its tiles, each in the pyramid's format (tex::Mip::format): an 8-bit file's in bytes, a half-float file's in half floats, a
 // grey one's in one channel (format below); XPU2_SESSION_FLOAT_TEXTURES=1 keeps every file in three floats.
 std::vector<std::vector<unsigned char>> byteTiles;unsigned format=0;
 std::vector<std::vector<const surface::V3*>> pages;std::atomic<uint64_t> bytes{0};
 // The finest level on demand (lazyTextures): its tiles decoded when first read (loadTile), each into a block of its own.
 std::string file;int channels=0,alphaChannel=-1,fileBase=0;bool alphaOnly=false;std::mutex lazyLock;
 // Loaded tiles, oldest first: under memory pressure the oldest are let go (letGo), and read again if asked for again.
 struct LazyTile {Block block;unsigned level,tile;size_t size;unsigned node;};
 std::deque<LazyTile> lazyTiles;size_t lazyBytes=0;std::atomic<unsigned> lazyLoaded{0},lazyFlat{0},lazyWide{0};
 // One texel of the file as kept: channels mapped, alpha as colour.
 surface::V3 kept(const float* d)const{surface::V3 c;float a;texelChannels(d,channels,alphaChannel,c,a);if(alphaOnly)c={a,a,a};return c;}
 // A finest-level tile decoded now (tex::Mip::load): the same values the whole level gave, through OIIO's conversion.
 static void loadTile(const void* self,unsigned level,unsigned tile){auto* p=const_cast<Pyramid*>(static_cast<const Pyramid*>(self));
  // Decoded outside the lock (tiles of one texture decode side by side); the first to finish publishes its page.
  auto& page=p->pages[level][tile];if(page)return;
  // XPU2_SESSION_FAIL_TILE=N: the Nth tile read fails, to test what follows one.
  static std::atomic<long long> untilFailure{[]{const char* v=std::getenv("XPU2_SESSION_FAIL_TILE");return v?std::atoll(v):0ll;}()};
  if(untilFailure.load(std::memory_order_relaxed)>0&&untilFailure.fetch_sub(1)==1)throw std::runtime_error("Texture tile read: "+p->file+", tile "+std::to_string(tile)+": failed for a test (XPU2_SESSION_FAIL_TILE)");
  const tex::Mip& m=p->image.levels[level];const unsigned nx=(m.width+tex::Tile-1)/tex::Tile,tx=tile%nx,ty=tile/nx;
  const unsigned tw=std::min(tex::Tile,m.width-tx*tex::Tile),th=std::min(tex::Tile,m.height-ty*tex::Tile);
  std::vector<float> decoded(size_t(tw)*th*p->channels);
  if(!lazyCache()->get_pixels(OIIO::ustring(p->file),0,int(level)+p->fileBase,int(tx*tex::Tile),int(tx*tex::Tile+tw),int(ty*tex::Tile),int(ty*tex::Tile+th),0,1,
                              OIIO::TypeDesc::FLOAT,decoded.data()))
   throw std::runtime_error("Texture tile read: "+p->file+", tile "+std::to_string(tile)+": "+lazyCache()->geterror());
  std::vector<surface::V3> held(size_t(tw)*th);bool flat=true,fits=true;
  for(size_t at=0;at<held.size();++at){held[at]=p->kept(&decoded[at*p->channels]);const surface::V3& c=held[at];
   flat=flat&&c.x==held[0].x&&c.y==held[0].y&&c.z==held[0].z;if(p->rounds)fits=fits&&halfFits(c.x)&&halfFits(c.y)&&halfFits(c.z);}
  // A tile with a value past half floats' range held in floats (tex::widePage), the level's other tiles in half floats.
  const bool wide=p->rounds&&!fits;const unsigned format=wide?tex::wideFormat(p->format):p->format;
  // A flat tile, all its texels one value (a mask's black, a normal map's untouched blue): that value alone, its page marked so
  // (tex::flatPage), on the card too. Its value is the one its texels would read: checked to be one its format holds, or rounded
  // as they would be.
  surface::V3 value=held[0];
  if(flat&&!p->compressed){unsigned char check[12];storeTexel(value,format,check,p->rounds);if(p->rounds&&!wide)value=halfRoundedTexel(value);}
  const size_t size=flat?sizeof(surface::V3):p->compressed?size_t(tex::tileBytes(2,0)):size_t(tw)*th*tex::texelBytes(format);
  const unsigned node=multi::currentNode();Block block=nodeBlock(size,node);
  if(flat)std::memcpy(block.get(),&value,sizeof(surface::V3));
  else if(p->compressed)encodeTile(held.data(),tw,th,block.get());
  else for(size_t at=0;at<held.size();++at)storeTexel(held[at],format,block.get()+at*tex::texelBytes(format),p->rounds);
  std::lock_guard<std::mutex> g(p->lazyLock);if(page)return;
  std::atomic_thread_fence(std::memory_order_release);
  page=reinterpret_cast<const surface::V3*>(reinterpret_cast<uintptr_t>(block.get())|(flat?1u:wide?2u:0u));
  p->lazyTiles.push_back({std::move(block),level,tile,size,node});nodeTileBytes(node)+=size;p->lazyBytes+=size;p->bytes+=size;p->lazyLoaded++;if(flat)p->lazyFlat++;else if(wide)p->lazyWide++;}
 // The decoded tiles it holds, in bytes: all, or those on one node.
 size_t lazyHeld(int node=-1){std::lock_guard<std::mutex> g(lazyLock);if(node<0)return lazyBytes;size_t n=0;for(const auto& t:lazyTiles)if(int(t.node)==node)n+=t.size;return n;}
 // The oldest tiles (of one node, or of any), up to want bytes: their pages cleared, their blocks handed over to free once
 // no reader has them.
 size_t letGo(size_t want,std::vector<Block>& out,int node=-1){
  std::lock_guard<std::mutex> g(lazyLock);size_t freed=0;
  for(auto it=lazyTiles.begin();freed<want&&it!=lazyTiles.end();){if(node>=0&&int(it->node)!=node){++it;continue;}auto& t=*it;
   if(tex::flatPage(pages[t.level][t.tile]))lazyFlat--;else if(tex::widePage(pages[t.level][t.tile]))lazyWide--;pages[t.level][t.tile]=nullptr;freed+=t.size;
   nodeTileBytes(t.node)-=t.size;out.push_back(std::move(t.block));it=lazyTiles.erase(it);lazyLoaded--;}
  lazyBytes-=freed;bytes-=freed;return freed;}
 // The oldest tiles of node from, up to want bytes, copied into node to's memory (the calling thread on to: multi::OnNode),
 // their pages pointed at the copies and the old blocks handed over to free once no reader has them. A reader holding a
 // page reads the same texels in either.
 size_t moveTo(unsigned from,unsigned to,size_t want,std::vector<Block>& out){
  std::lock_guard<std::mutex> g(lazyLock);size_t moved=0;
  for(auto& t:lazyTiles){if(moved>=want)break;if(t.node!=from)continue;
   auto& page=pages[t.level][t.tile];const uintptr_t tag=reinterpret_cast<uintptr_t>(page)&3u;
   Block copy=nodeBlock(t.size,to);std::memcpy(copy.get(),t.block.get(),t.size);
   std::atomic_thread_fence(std::memory_order_release);
   page=reinterpret_cast<const surface::V3*>(reinterpret_cast<uintptr_t>(copy.get())|tag);
   out.push_back(std::move(t.block));t.block=std::move(copy);nodeTileBytes(from)-=t.size;nodeTileBytes(to)+=t.size;t.node=to;moved+=t.size;}
  return moved;}
 ~Pyramid();
 // Texture overflow mode: this pyramid's levels as BC6H (tex::Mip format 2). Not a map with negative values, BC6H's unsigned.
 bool compressed=false;
 // What the file holds (reportTextures): 0 8-bit, 1 half, 2 float, 3 another; and, read once from a decoded level, whether its
 // three channels are one and how much of it is flat.
 unsigned sourceFormat=3;mutable bool scanned=false,grey=false;mutable double flatShare=0;
 // A float file held in half floats, each value rounded to the nearest: a relative error of at most 1/2048, a twentieth of an
 // 8-bit image's step; tiles or coarser levels with a value past their range stay in floats. XPU2_SESSION_FLOAT_TEXTURES=1
 // keeps every file in floats, as MoonRay reads them.
 bool rounds=false;
 Pyramid(const Pyramid&)=delete;
 explicit Pyramid(const std::string& name,bool alpha=false):file(name),alphaOnly(alpha){auto in=OIIO::ImageInput::open(name);need(bool(in),"Pyramid read");pages.reserve(tex::MaxLevels);
  // Whether it is to be compressed, by a small level's values.
  if(overflowCompress().load()&&compressorThere()){compressed=true;
   for(int l=0;in->seek_subimage(0,l);++l){const auto small=in->spec();if(std::max(small.width,small.height)>64)continue;
    std::vector<float> values(size_t(small.width)*small.height*small.nchannels);
    if(in->read_image(0,l,0,small.nchannels,OIIO::TypeDesc::FLOAT,values.data()))for(float v:values)if(v<0){compressed=false;break;}
    break;}
   need(in->seek_subimage(0,0),"Pyramid read");}
  for(unsigned level=0;level<tex::MaxLevels&&in->seek_subimage(0,level);++level){auto spec=in->spec();need(spec.get_int_attribute("sampleborder",0)==0&&spec.get_int_attribute("oiio:sampleborder",0)==0,"Border-sampled TX unsupported");
   unsigned w=spec.width,h=spec.height,nx=(w+tex::Tile-1)/tex::Tile,ny=(h+tex::Tile-1)/tex::Tile;
   // Texture overflow mode: the levels over its side left out, the first one under it this pyramid's finest.
   if(const unsigned side=overflowSide().load();side&&std::max(w,h)>side&&!image.count){fileBase=int(level)+1;continue;}
   static const bool floats=[]{const char* v=std::getenv("XPU2_SESSION_FLOAT_TEXTURES");return v&&v[0]=='1';}();
   channels=spec.nchannels;alphaChannel=spec.alpha_channel;
   sourceFormat=spec.format==OIIO::TypeDesc::UINT8?0u:spec.format==OIIO::TypeDesc::HALF?1u:spec.format==OIIO::TypeDesc::FLOAT?2u:3u;
   // Held as the file holds it: bytes, half floats or floats, one channel where it is grey (texelChannels gives grey to all three:
   // one channel, grey and alpha, or the alpha alone), each read back as the same value (storeTexel's check).
   const bool grey=channels==1||(channels==2&&alphaChannel==1)||alpha;
   format=floats?0u:compressed?2u:sourceFormat==0&&bytesExact()?(grey?5u:1u):sourceFormat==1||sourceFormat==2?(grey?6u:3u):grey?4u:0u;
   rounds=sourceFormat==2&&(format==3||format==6);
   if(!image.count&&lazyTextures()&&spec.tile_width>0){
    // The finest level read when first used (loadTile): its pages empty for now.
    pages.emplace_back(size_t(nx)*ny,nullptr);
    image.levels[image.count]={w,h,pages.back().data(),0,format,&Pyramid::loadTile,this};++image.count;continue;}
   std::vector<float> decoded(size_t(w)*h*spec.nchannels);
   need(in->read_image(0,int(level),0,spec.nchannels,OIIO::TypeDesc::FLOAT,decoded.data()),"Stored mip decode");pages.emplace_back(nx*ny,nullptr);auto& page=pages.back();
   if(compressed){
    // Its tiles side by side: a level of some thousands of tiles, as the scene is made ready.
    const size_t base=byteTiles.size();byteTiles.resize(base+size_t(nx)*ny);for(size_t k=base;k<byteTiles.size();++k)byteTiles[k].resize(tex::tileBytes(2,0));
    std::atomic<unsigned> next{0};const int nc=spec.nchannels,ac=spec.alpha_channel;const bool onlyAlpha=alpha;
    std::mutex failing;std::exception_ptr failed;
    auto work=[&]{try{std::vector<surface::V3> held;for(unsigned i;(i=next++)<nx*ny;){const unsigned x=i%nx,y=i/nx,tw=std::min(tex::Tile,w-x*tex::Tile),th=std::min(tex::Tile,h-y*tex::Tile);
      held.resize(size_t(tw)*th);
      for(unsigned j=0;j<th;++j)for(unsigned q=0;q<tw;++q){size_t idx=(size_t(y*tex::Tile+j)*w+x*tex::Tile+q)*nc;surface::V3 c;float a;texelChannels(&decoded[idx],nc,ac,c,a);if(onlyAlpha)c={a,a,a};held[size_t(j)*tw+q]=c;}
      encodeTile(held.data(),tw,th,byteTiles[base+i].data());}}
     catch(...){std::lock_guard<std::mutex> lock(failing);if(!failed)failed=std::current_exception();next=nx*ny;}};
    const unsigned threads=std::max(1u,std::thread::hardware_concurrency()/2);std::vector<std::thread> pool;
    try{if(size_t(nx)*ny>64)for(unsigned k=1;k<threads;++k)pool.emplace_back(work);}catch(const std::system_error&){}
    work();for(auto& t:pool)t.join();if(failed)std::rethrow_exception(failed);
    for(unsigned i=0;i<nx*ny;++i){page[i]=reinterpret_cast<const surface::V3*>(byteTiles[base+i].data());bytes+=byteTiles[base+i].size();}
    image.levels[image.count]={w,h,page.data(),0,2u};++image.count;continue;}
   // A coarser level of a float file with a value past half floats' range stays in floats.
   bool fits=true;if(rounds)for(float v:decoded)if(!halfFits(v)){fits=false;break;}
   const unsigned levelFormat=fits?format:tex::wideFormat(format),size=tex::texelBytes(levelFormat);
   for(unsigned y=0;y<ny;++y)for(unsigned x=0;x<nx;++x){unsigned tw=std::min(tex::Tile,w-x*tex::Tile),th=std::min(tex::Tile,h-y*tex::Tile);
    // Each texel in the pyramid's format (an 8-bit file's values are its bytes times 1/255, its fill 0 or 1). Negative values are
    // kept, as MoonRay's texture sampler keeps them (a signed displacement or vector map).
    byteTiles.emplace_back(size_t(tw)*th*size);auto& tile=byteTiles.back();
    for(unsigned j=0;j<th;++j)for(unsigned i=0;i<tw;++i){size_t idx=(size_t(y*tex::Tile+j)*w+x*tex::Tile+i)*spec.nchannels;
     storeTexel(kept(&decoded[idx]),levelFormat,&tile[size_t(j*tw+i)*size],rounds);}
    page[y*nx+x]=reinterpret_cast<const surface::V3*>(tile.data());bytes+=tile.size();
   }image.levels[image.count]={w,h,page.data(),0,levelFormat};++image.count;
  }need(image.count&&image.levels[image.count-1].width==1&&image.levels[image.count-1].height==1,"Software mip level bound");
  // Realtime's view: the pyramid from its first level no wider or taller than realtimeSide() on.
  const unsigned side=realtimeSide();
  while(side&&realtimeFrom+1<image.count&&std::max(image.levels[realtimeFrom].width,image.levels[realtimeFrom].height)>side)++realtimeFrom;
  realtimeImage.count=image.count-realtimeFrom;
  for(unsigned k=0;k<realtimeImage.count;++k)realtimeImage.levels[k]=image.levels[realtimeFrom+k];
  std::lock_guard<std::mutex> listed(spillLock());spillList().push_back(this);
 }
 // The largest side a texture has in realtime's frames (XPU2_SESSION_RT_TEXTURE_SIDE, default 1024, was 1280; 0: as rendered).
 // Perceptually enough at realtime's sizes, and its finer levels are then never decoded nor sent to the card.
 static unsigned realtimeSide(){static const unsigned side=[]{const char* v=std::getenv("XPU2_SESSION_RT_TEXTURE_SIDE");
   return v?unsigned(std::max(0,std::atoi(v))):1024u;}();return side;}
 tex::Image realtimeImage;unsigned realtimeFrom=0;
 // The pyramid a plan reads: realtime's from its first level of realtimeSide() or less, else the whole.
 const tex::Image& view(bool realtime)const{return realtime&&realtimeFrom?realtimeImage:image;}
};
inline Pyramid::~Pyramid(){std::lock_guard<std::mutex> listed(spillLock());auto& l=spillList();l.erase(std::remove(l.begin(),l.end(),this),l.end());}
// The decoded tiles held now, in bytes.
inline size_t tilesHeld(){std::lock_guard<std::mutex> listed(spillLock());size_t n=0;for(auto* p:spillList())n+=p->lazyHeld();return n;}
// Lets go of up to want bytes of decoded tiles, each texture's oldest, in proportion to what it holds, and returns what
// went. A tile asked for again is decoded again from its file (the disk is its record; the card's copy, where it has
// one, stays). Freed once every reader that may have seen one has left it (tex::spill).
// Blocks freed once every reader that may have seen one has left it (tex::spill).
inline void bury(std::vector<Block>& grave){
 if(grave.empty())return;
 std::atomic_thread_fence(std::memory_order_seq_cst);
 const unsigned long long stamp=tex::spill::epoch.fetch_add(1,std::memory_order_seq_cst);
 for(auto& r:tex::spill::readers){for(;;std::this_thread::yield()){const auto in=r.in.load(std::memory_order_seq_cst);if(!in||in>stamp)break;}}
 grave.clear();
}
inline size_t relieve(size_t want,int node=-1){
 std::vector<Block> grave;size_t freed=0;
 {std::lock_guard<std::mutex> listed(spillLock());size_t held=0;for(auto* p:spillList())held+=p->lazyHeld(node);if(!held)return 0;
  for(auto* p:spillList()){const size_t mine=p->lazyHeld(node);if(mine)freed+=p->letGo(size_t(double(want)*double(mine)/double(held))+1,grave,node);}}
 if(grave.empty())return 0;
 bury(grave);HeapCompact(node<0?GetProcessHeap():nodeHeap(unsigned(node)),0);return freed;
}
// NUMA (multi_device.h): a node short of memory first moves its oldest tiles to the node with the most room, read from
// there across the sockets, far quicker than decoding them again from their files; only what no node has room for is
// let go. A node's reserve is a tenth of its share of the machine's RAM, 1 GB at least; XPU2_SIMULATE_NODE_MB=a,b,...
// holds each node's tiles to that many MB instead (0: no limit).
inline size_t migrate(unsigned from,unsigned to,size_t want){
 std::vector<Block> grave;size_t moved=0;
 {multi::OnNode there(int(to));
  std::lock_guard<std::mutex> listed(spillLock());size_t held=0;for(auto* p:spillList())held+=p->lazyHeld(int(from));if(!held)return 0;
  for(auto* p:spillList()){const size_t mine=p->lazyHeld(int(from));if(mine)moved+=p->moveTo(from,to,size_t(double(want)*double(mine)/double(held))+1,grave);}}
 bury(grave);if(moved)HeapCompact(nodeHeap(from),0);return moved;
}
inline size_t nodeCap(unsigned node){return size_t(multi::simulatedEntry("XPU2_SIMULATE_NODE_MB",node))<<20;}
inline size_t nodeReserve(){MEMORYSTATUSEX s{};s.dwLength=sizeof s;GlobalMemoryStatusEx(&s);
 return std::max<size_t>(size_t(s.ullTotalPhys/std::max<size_t>(1,multi::nodes().size())/10),1ull<<30);}
// What a node asks to be let go of or moved now, in bytes.
inline size_t nodePressure(unsigned node){
 if(std::getenv("XPU2_SIMULATE_NODE_MB")){const size_t cap=nodeCap(node),held=nodeTileBytes(node).load();return cap&&held>cap?held-cap:0;}
 ULONGLONG free=0;if(!GetNumaAvailableMemoryNodeEx(USHORT(node),&free))return 0;
 const size_t reserve=nodeReserve();return free<reserve?reserve-size_t(free):0;}
// The room a node has for tiles from another, in bytes: what it has past twice its reserve.
inline size_t nodeRoom(unsigned node){
 if(std::getenv("XPU2_SIMULATE_NODE_MB")){const size_t cap=nodeCap(node),held=nodeTileBytes(node).load();return !cap?(size_t(1)<<40):held<cap?cap-held:0;}
 ULONGLONG free=0;if(!GetNumaAvailableMemoryNodeEx(USHORT(node),&free))return 0;
 const size_t keep=2*nodeReserve();return free>keep?size_t(free)-keep:0;}
// Each node short of memory: its tiles moved to the node with the most room, the rest let go. What was moved and let go.
inline void relieveNodes(size_t& moved,size_t& freed,unsigned& lastTo){
 if(!multi::numa())return;
 for(unsigned node:multi::nodes()){size_t want=nodePressure(node);if(!want)continue;
  unsigned to=node;size_t room=0;for(unsigned other:multi::nodes())if(other!=node){const size_t r=nodeRoom(other);if(r>room){room=r;to=other;}}
  if(room){const size_t m=migrate(node,to,std::min(want,room));moved+=m;want-=std::min(want,m);lastTo=to;}
  if(want)freed+=relieve(want,int(node));}
}
// What memory pressure asks to be let go now, in bytes: the machine's free RAM below its reserve (a tenth of it, 2 GB at
// least), or the session's committed memory past 92% of its ceiling (ceiling 0: none; commit is not RAM: the graphics driver
// commits 8 to 11 GB as it launches, and at 85% that alone let every tile go through a car's render); XPU2_SESSION_SPILL_MB
// holds the tiles to that many MB whatever the machine has.
inline size_t pressure(size_t ceiling){
 if(const char* v=std::getenv("XPU2_SESSION_SPILL_MB")){const size_t cap=size_t(std::max(0,std::atoi(v)))<<20,held=tilesHeld();return held>cap?held-cap:0;}
 size_t want=0;MEMORYSTATUSEX status{};status.dwLength=sizeof status;
 if(GlobalMemoryStatusEx(&status)){const size_t reserve=std::max<size_t>(size_t(status.ullTotalPhys/10),2ull<<30);
  if(status.ullAvailPhys<reserve)want=reserve-size_t(status.ullAvailPhys);}
 PROCESS_MEMORY_COUNTERS_EX c{};c.cb=sizeof c;
 if(ceiling&&K32GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&c),sizeof c)&&c.PrivateUsage>ceiling/100*92)
  want=std::max(want,size_t(c.PrivateUsage)-ceiling/100*88);
 return want;
}
// Watches for memory pressure for the rest of the process, four times a second, and says when it lets tiles go. A session
// past its ceiling used to go back to MoonRay XPU, and a machine out of RAM stalled, with tiles on hand it could read again.
inline void watchPressure(size_t ceiling){
 std::thread([ceiling]{
  auto said=std::chrono::steady_clock::now()-std::chrono::seconds(60),next=said;size_t since=0;
  // The tiles kept whatever the pressure, the render's working set: below it they would only be read again, and again. Letting go
  // that helped nothing is not repeated at once: the wait doubles, from 2 s to a minute, until the pressure has passed. And tiles
  // that could not make much of what is asked for are left alone. Past that the texture overflow mode is what can help.
  const size_t floor=std::getenv("XPU2_SESSION_SPILL_MB")?0:(1ull<<30);int wait=2;
  size_t nodeMoved=0,nodeFreed=0;unsigned nodeTo=0;auto nodeSaid=said;
  for(;;std::this_thread::sleep_for(std::chrono::milliseconds(250))){
   relieveNodes(nodeMoved,nodeFreed,nodeTo);
   if((nodeMoved||nodeFreed)&&std::chrono::steady_clock::now()-nodeSaid>std::chrono::seconds(5)){
    std::printf("MoonRay XPU2: NUMA node short of memory: %zu MB of decoded texture tiles moved to node %u, %zu MB let go\n",nodeMoved>>20,nodeTo,nodeFreed>>20);
    std::fflush(stdout);nodeSaid=std::chrono::steady_clock::now();nodeMoved=nodeFreed=0;}
   const size_t want=pressure(ceiling);if(!want){wait=2;continue;}
   const auto now=std::chrono::steady_clock::now();
   if(floor&&now<next)continue;
   const size_t held=tilesHeld(),spare=held>floor?held-floor:0;
   // At its floor, or with too little to matter, RAM short all the same.
   if(!spare||(floor&&spare<want/4)){pressureShort().store(true);next=now+std::chrono::seconds(wait);wait=std::min(60,wait*2);continue;}
   const size_t freed=relieve(std::min(want,spare));since+=freed;
   next=now+std::chrono::seconds(wait);wait=std::min(60,wait*2);
   if(since&&now-said>std::chrono::seconds(5)){
    MEMORYSTATUSEX status{};status.dwLength=sizeof status;GlobalMemoryStatusEx(&status);PROCESS_MEMORY_COUNTERS_EX c{};c.cb=sizeof c;K32GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&c),sizeof c);
    std::printf("MoonRay XPU2: memory pressure (%.1f GB of RAM free, %.1f GB committed of a %.1f GB ceiling): %zu MB of decoded texture tiles let go, %zu MB kept\n",double(status.ullAvailPhys)/(1ull<<30),double(c.PrivateUsage)/(1ull<<30),double(ceiling)/(1ull<<30),since>>20,tilesHeld()>>20);std::fflush(stdout);said=now;since=0;}
  }}).detach();
}
struct File {
 std::shared_ptr<OIIO::TextureSystem> system;OIIO::ustring name;HANDLE lock=INVALID_HANDLE_VALUE;bool eightBit=false;unsigned levels=0;uint64_t decodedBytes=0;std::shared_ptr<const Pyramid> pyramid;
 bool alpha=false;  // its alpha as its colour (ImageMap's alpha_only), a texture of its own beside the file's colour
 File(const std::filesystem::path& path,const std::filesystem::path& root,std::shared_ptr<OIIO::TextureSystem> s,bool alphaOnly=false):system(std::move(s)),name(path.string()),alpha(alphaOnly){
  admittedPath(path,root);
  // Retain a read lease: no writer or delete can change the file under OIIO's cache.
  lock=CreateFileW(path.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
  need(lock!=INVALID_HANDLE_VALUE,"Cannot lease immutable texture");
  try{LARGE_INTEGER bytes{};need(GetFileSizeEx(lock,&bytes)&&bytes.QuadPart>0&&bytes.QuadPart<=(1ll<<30),"Texture file budget");
   auto in=OIIO::ImageInput::open(path.string());need(bool(in),"TX open failed");
   int w=0,h=0;for(int level=0;level<32;++level){if(!in->seek_subimage(0,level))break;const auto& spec=in->spec();
    need(spec.depth==1&&spec.x==0&&spec.y==0&&spec.z==0&&spec.full_x==0&&spec.full_y==0&&spec.full_width==spec.width&&spec.full_height==spec.height&&spec.width>0&&spec.height>0&&spec.tile_width>0&&spec.tile_height>0&&spec.nchannels>=1&&spec.nchannels<=4&&spec.channelformats.empty()&&!spec.deep,"Unsupported TX layout");
    if(level==0){eightBit=spec.format==OIIO::TypeDesc::UINT8;need(eightBit||spec.format==OIIO::TypeDesc::HALF||spec.format==OIIO::TypeDesc::FLOAT,"Unsupported TX format");}
    else need(spec.width==std::max(1,w/2)&&spec.height==std::max(1,h/2),"Invalid stored mip pyramid");
    w=spec.width;h=spec.height;if(level==0){unsigned seen=largestTexture().load();while(unsigned(std::max(w,h))>seen&&!largestTexture().compare_exchange_weak(seen,unsigned(std::max(w,h)))){}}
    if(const unsigned side=overflowSide().load();!side||unsigned(std::max(w,h))<=side)decodedBytes+=overflowCompress().load()?uint64_t(w)*h:uint64_t(w)*h*spec.nchannels*4;
    need(decodedBytes<=(2ull<<30),"Decoded texture budget");++levels;
   }need(levels>0&&w==1&&h==1,"Complete stored mip pyramid required");need(!in->seek_subimage(1,0),"Multipart texture unsupported");
   OIIO::ImageSpec spec;need(system->get_imagespec(name,spec),"OIIO texture admission failed");
  }catch(...){CloseHandle(lock);lock=INVALID_HANDLE_VALUE;throw;}
 }
 ~File(){if(lock!=INVALID_HANDLE_VALUE)CloseHandle(lock);}File(const File&)=delete;
};
// Where a scene's textures stand in memory, for the log (the session prints it at each load and frame), and what each lossless
// way of holding them would save. Greyness and flatness are read from a level of at most 1024 pixels, the first decoded, once
// a texture: an estimate of its finest level, flat tiles more so.
struct TextureReport {
 size_t files=0,eightBit=0,half=0,floats=0,grey=0,compressed=0,oneChannel=0,rounded=0;unsigned largest=0;
 uint64_t finest=0,coarser=0,decoded=0,tiles=0,tilesDecoded=0,flatTiles=0,wideTiles=0;
 uint64_t saveGrey=0,saveHalf=0,saveFlat=0;double scanSeconds=0;
};
inline void scanPyramid(const Pyramid& p){
 if(p.scanned)return;p.scanned=true;
 const tex::Image& image=p.image;unsigned k=(image.count&&image.levels[0].load)?1u:0u;
 while(k+1<image.count&&std::max(image.levels[k].width,image.levels[k].height)>1024)++k;
 if(k>=image.count||image.levels[k].format==2)return;
 const tex::Mip& m=image.levels[k];const unsigned nx=(m.width+tex::Tile-1)/tex::Tile,ny=(m.height+tex::Tile-1)/tex::Tile;
 bool grey=true;uint64_t flat=0;
 for(unsigned i=0;i<nx*ny;++i){
  const unsigned tw=std::min(tex::Tile,m.width-(i%nx)*tex::Tile),th=std::min(tex::Tile,m.height-(i/nx)*tex::Tile),n=tw*th;bool same=true;
  const unsigned tx=i%nx,x0=tx*tex::Tile,y0=(i/nx)*tex::Tile;surface::V3 first{};
  for(unsigned q=0;q<n;++q){surface::V3 c;tex::texel(m.pages[i],m.width,tx,x0+q%tw,y0+q/tw,c,m.format);if(!q)first=c;
   if(c.x!=c.y||c.y!=c.z)grey=false;if(c.x!=first.x||c.y!=first.y||c.z!=first.z)same=false;if(!grey&&!same)break;}
  if(same)++flat;}
 p.grey=grey;p.flatShare=double(flat)/double(nx*ny);
}
inline TextureReport reportTextures(const std::vector<std::shared_ptr<File>>& files,bool realtime){
 TextureReport r;const auto began=std::chrono::steady_clock::now();
 for(const auto& f:files){if(!f||!f->pyramid)continue;const Pyramid& p=*f->pyramid;++r.files;
  if(p.sourceFormat==0)++r.eightBit;else if(p.sourceFormat==1)++r.half;else ++r.floats;
  scanPyramid(p);if(p.grey)++r.grey;if(p.compressed)++r.compressed;if(tex::oneChannel(p.format))++r.oneChannel;if(p.rounds)++r.rounded;
  const tex::Image& view=p.view(realtime);if(!view.count)continue;
  const tex::Mip& m=view.levels[0];r.largest=std::max(r.largest,std::max(m.width,m.height));
  const unsigned nx=(m.width+tex::Tile-1)/tex::Tile,ny=(m.height+tex::Tile-1)/tex::Tile;
  const auto levelBytes=[](const tex::Mip& l){return l.format==2?uint64_t((l.width+tex::Tile-1)/tex::Tile)*((l.height+tex::Tile-1)/tex::Tile)*tex::tileBytes(2,0):uint64_t(l.width)*l.height*tex::texelBytes(l.format);};
  const uint64_t level=levelBytes(m);
  uint64_t coarse=0;for(unsigned l=1;l<view.count;++l)coarse+=levelBytes(view.levels[l]);
  const uint64_t held=const_cast<Pyramid&>(p).lazyHeld();
  r.finest+=level;r.coarser+=coarse;r.tiles+=uint64_t(nx)*ny;
  if(m.load){r.decoded+=held;for(unsigned i=0;i<nx*ny;++i)if(m.pages[i]){++r.tilesDecoded;if(tex::flatPage(m.pages[i]))++r.flatTiles;else if(tex::widePage(m.pages[i]))++r.wideTiles;}}
  // What is still to be had: grey maps held in three channels, half-float ones in floats, flat tiles of a level read whole.
  if(p.grey&&!tex::oneChannel(p.format)&&p.format!=2)r.saveGrey+=(level+coarse)/3*2;
  if(p.sourceFormat==1&&(p.format==0||p.format==4))r.saveHalf+=(p.format==4?(level+coarse):(p.grey?(level+coarse)/3:level+coarse))/2;
  if(!m.load)r.saveFlat+=uint64_t(double(level)*p.flatShare);
 }
 r.scanSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();return r;
}
struct Binding {
 std::shared_ptr<File> file;float sx=1,sy=1,ox=0,oy=0,mipBias=0;bool wrap=true,gamma=false;
 tex::Binding descriptor(bool realtime=false)const{return {file->pyramid?&file->pyramid->view(realtime):nullptr,sx,sy,ox,oy,mipBias,wrap?1u:0u,gamma&&file->eightBit?1u:0u};}
 surface::V3 lookup(float u,float v,float dudx,float dvdx,float dudy,float dvdy)const{
  OIIO::TextureOpt options;options.anisotropic=4;options.mipmode=OIIO::TextureOpt::MipModeAniso;options.interpmode=OIIO::TextureOpt::InterpBilinear;
  options.swrap=options.twrap=wrap?OIIO::TextureOpt::WrapPeriodic:OIIO::TextureOpt::WrapClamp;
  float out[4];float bias=1+mipBias;
  // ImageMap's primary-hit derivative scaling and T flip, including its cross-axis convention.
  need(file->system->texture(file->name,options,sx*u+ox,1-(sy*v+oy),dudx*sx*bias,-dvdx*sx*bias,dudy*sy*bias,-dvdy*sy*bias,4,out),"Texture lookup failed; invalidate whole epoch");
  surface::V3 c{out[0],out[1],out[2]};if(gamma&&file->eightBit){c={powf(std::max(c.x,0.f),2.2f),powf(std::max(c.y,0.f),2.2f),powf(std::max(c.z,0.f),2.2f)};}
  need(surface::finite(c)&&c.x>=0&&c.y>=0&&c.z>=0,"Invalid texture radiance");return c;
 }
};
}
