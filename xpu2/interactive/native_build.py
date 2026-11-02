"""Isolated native probe compilation using the installed Windows RDL toolchain."""
from pathlib import Path
import shlex
import subprocess
import sys
from fallback import environment
from reference_runtime import sha
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from xpu2_rdl_index_probe import block

def bc6h_kernels(source,out):
    """ISPC Texture Compressor's kernels (upstream/ISPCTextureCompressor, for texture overflow mode), by the toolchain's ISPC, for
    SSE2 (which every 64-bit processor has) up to AVX2, the best the processor has chosen as it runs. The objects, and the flags
    xpu2_bc6h.cc needs."""
    source,out=Path(source).resolve(),Path(out).resolve()
    out.mkdir(exist_ok=False)
    tools=source/'toolchain/msys64/ucrt64/bin'
    kernel=source/'upstream/ISPCTextureCompressor/ispc_texcomp/kernel.ispc'
    done=subprocess.run([str(tools/'ispc.exe'),'-O2',str(kernel),'-o',str(out/'kernel.o'),'-h',str(out/'kernel_ispc.h'),
                         '--target=sse2-i32x4,sse4-i32x4,avx1-i32x8,avx2-i32x8','--opt=fast-math','--pic','--woff'],
                        env=environment(tools,out),capture_output=True,text=True,errors='replace',creationflags=subprocess.CREATE_NO_WINDOW)
    if done.returncode:raise RuntimeError('ISPC failed: '+done.stderr[-800:])
    return sorted(out.glob('kernel*.o')),['-I'+str(out),'-I'+str(kernel.parent)]


def build_units(source,units,out,extra_libraries=(),extra_flags=()):
    """Several units: each preprocessed and compiled at once (-c), then linked into probe.exe."""
    source,out=Path(source).resolve(),Path(out).resolve()
    units=[Path(u).resolve() for u in units]
    out.mkdir(exist_ok=False)
    compiler=source/'toolchain/msys64/ucrt64/bin/g++.exe'
    root=source/'build/native-avx'
    ninja=(root/'build.ninja').read_text(encoding='utf-8')
    c=block(ninja,'build scene-rdl2/cmd/rdl2_cmd/rdl2_print/CMakeFiles/rdl2_print.dir/rdl2_print.cc.obj:')
    l=block(ninja,'build bin/rdl2_print.exe:')
    flags=shlex.split(' '.join(c[k] for k in ('DEFINES','FLAGS','INCLUDES')))
    flags+=list(extra_flags)
    flags+=['-I'+str(compiler.parent.parent/'include'),'-fno-fast-math','-ffp-contract=off']
    libs=[str((root/v).resolve()) if not v.startswith('-') else v for v in shlex.split(l['LINK_LIBRARIES'])]
    libs += [str(compiler.parent.parent/'lib'/n) for n in ('libOpenImageIO.dll.a','libOpenImageIO_Util.dll.a')]
    libs += [str(v) if str(v).startswith('-') else str(Path(v).resolve()) for v in extra_libraries]
    held={v:sha(Path(v)) for v in libs if not v.startswith('-')}
    env=environment(compiler.parent,out)
    def run_all(commands,label):
        procs=[(name,subprocess.Popen([str(compiler),*flags,*cmd],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                                      creationflags=subprocess.CREATE_NO_WINDOW)) for name,cmd in commands]
        failed=[]
        for name,proc in procs:
            text,_=proc.communicate(timeout=1800)
            (out/(name+'.log')).write_text(text or '',encoding='utf-8')
            if proc.returncode:failed.append(name+': '+(text or '')[-500:])
        if failed:raise RuntimeError(label+' failed: '+' | '.join(failed))
    pre={u:out/(u.stem+'.ii') for u in units}
    obj={u:out/(u.stem+'.o') for u in units}
    run_all([(u.stem+'-preprocess',['-E',str(u),'-o',str(pre[u])]) for u in units],'preprocess')
    run_all([(u.stem+'-compile',['-c',str(pre[u]),'-o',str(obj[u])]) for u in units],'compile')
    exe=out/'probe.exe'
    link=subprocess.run([str(compiler),*flags,*[str(obj[u]) for u in units],'-o',str(exe),*libs],env=env,capture_output=True,text=True,
                        timeout=900,creationflags=subprocess.CREATE_NO_WINDOW)
    (out/'link.log').write_text(link.stdout+link.stderr,encoding='utf-8')
    if link.returncode:raise RuntimeError('link failed: '+link.stderr[-500:])
    if any(sha(Path(p))!=v for p,v in held.items()):raise RuntimeError('Link library drift')
    return exe,{'libraries':held,'compiler_sha256':sha(compiler),'preprocessed_sha256':{u.name:sha(pre[u]) for u in units},'executable_sha256':sha(exe)}

def build(source,unit,out,extra_libraries=(),extra_flags=()):
    source,out,unit=map(lambda x:Path(x).resolve(),(source,out,unit))
    out.mkdir(exist_ok=False)
    compiler=source/'toolchain/msys64/ucrt64/bin/g++.exe'
    root=source/'build/native-avx'
    ninja=(root/'build.ninja').read_text(encoding='utf-8')
    c=block(ninja,'build scene-rdl2/cmd/rdl2_cmd/rdl2_print/CMakeFiles/rdl2_print.dir/rdl2_print.cc.obj:')
    l=block(ninja,'build bin/rdl2_print.exe:')
    flags=shlex.split(' '.join(c[k] for k in ('DEFINES','FLAGS','INCLUDES')))
    flags+=list(extra_flags)
    flags+=['-I'+str(compiler.parent.parent/'include'),'-fno-fast-math','-ffp-contract=off']
    libs=[str((root/v).resolve()) if not v.startswith('-') else v for v in shlex.split(l['LINK_LIBRARIES'])]
    libs += [str(compiler.parent.parent/'lib'/n) for n in ('libOpenImageIO.dll.a','libOpenImageIO_Util.dll.a')]
    libs += [str(v) if str(v).startswith('-') else str(Path(v).resolve()) for v in extra_libraries]
    held={v:sha(Path(v)) for v in libs if not v.startswith('-')}
    pre,exe=out/'probe.ii',out/'probe.exe'
    for name,cmd in [('preprocess',['-E',str(unit),'-o',str(pre)]),('compile',[str(pre),'-o',str(exe),*libs])]:
        r=subprocess.run([str(compiler),*flags,*cmd],env=environment(compiler.parent,out),capture_output=True,
                         text=True,timeout=900,creationflags=subprocess.CREATE_NO_WINDOW)
        (out/(name+'.log')).write_text(r.stdout+r.stderr,encoding='utf-8')
        if r.returncode:raise RuntimeError(name+' failed: '+r.stderr[-500:])
    if any(sha(Path(p))!=v for p,v in held.items()):raise RuntimeError('Link library drift')
    return exe,{'libraries':held,'compiler_sha256':sha(compiler),'preprocessed_sha256':sha(pre),'executable_sha256':sha(exe)}
