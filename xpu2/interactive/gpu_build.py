"""Private NVRTC build using the local pinned toolchain, with input inventory."""
import ctypes as C,os
from pathlib import Path
from run_sample_core import bind
from reference_runtime import sha
HERE=Path(__file__).resolve().parent
PARITY_FLAGS=['--std=c++17','--gpu-architecture=compute_75','--fmad=false','--prec-div=true','--prec-sqrt=true','--ftz=false']
PARITY_HEADERS=('gpu_jobs.h','triangle_hit.h','texture_core.h','resume_core.h','surface_core.h','sample_stream.h','path_stage.h')
# NanoVDB, for VDB volumes on the GPU (volume_grid.h): its three headers from the toolchain, and <cuda/std/limits>, which
# it asks for and XPU2's NVRTC has not, as nvrtc_limits.h. Handed to NVRTC by include name, as the program's own headers.
def rtc_headers(source):
    inc=Path(source)/'toolchain/msys64/ucrt64/include'
    return {**{n:inc/n for n in ('nanovdb/NanoVDB.h','nanovdb/util/Util.h','nanovdb/math/Math.h')},'cuda/std/limits':HERE/'nvrtc_limits.h'}
# Defaults build the parity pipeline's program; the asynchronous lane passes its own unit, headers and flags.
def compile_ptx(source,out,unit='gpu_trace.cu',header_names=PARITY_HEADERS,base_flags=PARITY_FLAGS):
 nvdir=source/'toolchain/xpu/cuda_nvrtc-windows-x86_64-12.8.93-archive/bin'
 cuda=source/'toolchain/xpu/cuda_cudart-windows-x86_64-12.8.90-archive/include'
 optix=source/'toolchain/xpu/optix-dev/include'
 handle=os.add_dll_directory(str(nvdir));builtins=C.WinDLL(str(nvdir/'nvrtc-builtins64_128.dll'));nv=C.WinDLL(str(nvdir/'nvrtc64_120_0.dll'))
 ptr=C.c_void_p;strings=C.POINTER(C.c_char_p);program=ptr()
 headers={n:(HERE/n).read_bytes() for n in header_names}
 external=rtc_headers(source);headers.update({n:p.read_bytes() for n,p in external.items()})
 held={str(p):sha(p) for base in (cuda,optix) for p in base.rglob('*.h')};held.update({str(p):sha(p) for p in external.values()})
 flags=[*base_flags,'-I'+str(cuda),'-I'+str(optix)]
 bind(nv,'nvrtcCreateProgram',[C.POINTER(ptr),C.c_char_p,C.c_char_p,C.c_int,strings,strings])(C.byref(program),(HERE/unit).read_bytes(),unit.encode(),len(headers),(C.c_char_p*len(headers))(*headers.values()),(C.c_char_p*len(headers))(*(n.encode() for n in headers)))
 try:
  fn=nv.nvrtcCompileProgram;fn.argtypes=[ptr,C.c_int,strings];fn.restype=C.c_int;result=fn(program,len(flags),(C.c_char_p*len(flags))(*(f.encode() for f in flags)))
  def get(kind):
   n=C.c_size_t();bind(nv,'nvrtcGet'+kind+'Size',[ptr,C.POINTER(C.c_size_t)])(program,C.byref(n));data=C.create_string_buffer(n.value);bind(nv,'nvrtcGet'+kind,[ptr,ptr])(program,data);return data.value
  (out/'nvrtc.log').write_bytes(get('ProgramLog'))
  if result:raise RuntimeError('NVRTC compilation failed; see nvrtc.log')
  file=out/(Path(unit).stem+'.ptx');file.write_bytes(get('PTX'))
  if any(sha(Path(k))!=v for k,v in held.items()):raise RuntimeError('Toolchain header drift')
  return file,{'flags':flags,'headers':held,'ptx_sha256':sha(file),'nvrtc_sha256':sha(nvdir/'nvrtc64_120_0.dll')}
 finally:bind(nv,'nvrtcDestroyProgram',[C.POINTER(ptr)])(C.byref(program));handle.close()
