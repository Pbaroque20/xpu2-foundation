"""Compile xpu2/sss_missproof.cu to PTX with NVRTC (for XPU2_SSS_GPU_PTX)."""
import argparse
import ctypes
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nvrtc', type=Path, required=True, help='NVRTC bin folder (nvrtc64_120_0.dll)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arch', default='compute_86')
    a = p.parse_args()
    os.add_dll_directory(str(a.nvrtc.resolve()))
    os.environ['PATH'] = str(a.nvrtc.resolve()) + os.pathsep + os.environ['PATH']   # NVRTC loads its builtins by name
    nv = ctypes.CDLL(str(a.nvrtc.resolve() / 'nvrtc64_120_0.dll'))
    src = (ROOT / 'xpu2/sss_missproof.cu').read_bytes()
    prog = ctypes.c_void_p()
    if nv.nvrtcCreateProgram(ctypes.byref(prog), src, b'sss_missproof.cu', 0, None, None):
        raise SystemExit('nvrtcCreateProgram failed')
    opts = (ctypes.c_char_p * 2)(('--gpu-architecture=' + a.arch).encode(), b'--std=c++14')
    r = nv.nvrtcCompileProgram(prog, 2, opts)
    size = ctypes.c_size_t()
    nv.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
    log = ctypes.create_string_buffer(size.value)
    nv.nvrtcGetProgramLog(prog, log)
    if r:
        raise SystemExit('NVRTC: ' + log.value.decode(errors='replace'))
    nv.nvrtcGetPTXSize(prog, ctypes.byref(size))
    ptx = ctypes.create_string_buffer(size.value)
    nv.nvrtcGetPTX(prog, ptx)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_bytes(ptx.value)
    print('wrote', a.output, len(ptx.value), 'bytes')


if __name__ == '__main__':
    main()
