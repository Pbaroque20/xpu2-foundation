// XPU2's unit of ISPC Texture Compressor (upstream/ISPCTextureCompressor, Copyright 2017 Intel Corporation, MIT; see
// THIRD_PARTY_NOTICES.md): its C++ side, over kernel objects native_build.py makes with the toolchain's ISPC, for SSE2 (which every
// 64-bit processor has) up to AVX2, the best the processor has chosen as it runs.
#include "ispc_texcomp.cpp"

// ISPC's code probes a large stack by MSVC's name; MinGW has the same routine under its own.
__asm__(".globl __chkstk; __chkstk: jmp ___chkstk_ms");
