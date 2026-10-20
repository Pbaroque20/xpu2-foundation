"""Merged random-walk subsurface waves with the GPU miss proof (xpu2/sss_gpu.h).

Builds on xpu2/rndr_sss_merged.py. In the Waves phase, when XPU2_SSS_GPU=1 and a
wave has at least XPU2_SSS_GPU_MIN rays, the wave's rays go to the GPU first;
rays the GPU proves cannot hit anything skip Embree and count as misses. After a
miss the walk reads only the ray's origin, direction and its own sampled
distance, none of which Embree changes on a miss, so the result is unchanged; the
intersection-ray statistic is still counted. Everything the GPU cannot rule out
(and every ray when the scene is unsupported) goes to Embree as before.
XPU2_SSS_GPU_VERIFY=1 traces the proven rays with Embree too and counts any hit.
"""
from .native_gpu_paths import once
from . import rndr_sss_merged as merged

FILES = merged.FILES
HEADERS = merged.HEADERS + ('sss_gpu.h',)
NEW_SOURCES = ('sss_gpu.cc',)

INTERSECT = '''            for (size_t k = 0; k < all.size(); ++k) {
                if (pendingKind[k]) {
                    xpu2merged::Walk& w = *all[k];
                    geomTls->mSubsurfaceTraceSet = w.job->traceSet;
                    hit[k] = scene->intersectRay(pbrTls->mTopLevelTls, w.sssRay, w.isectOut, w.job->lobeType) ? 1 : 0;
                }
            }
'''

GPU_INTERSECT = '''            // GPU miss proof for large waves: candidate 0 means Embree would miss.
            candidate.assign(all.size(), 1);
            if (xpu2sssgpu::enabled() && pending >= size_t(xpu2sssgpu::threshold())) {
                gpuRays.clear(); gpuIndex.clear(); gpuMaterial.clear(); gpuTraceSet.clear();
                for (size_t k = 0; k < all.size(); ++k) {
                    if (pendingKind[k]) {
                        const Ray& r = all[k]->sssRay;
                        const float rec[8] = {r.org.x, r.org.y, r.org.z, r.tnear, r.dir.x, r.dir.y, r.dir.z, r.tfar};
                        gpuRays.insert(gpuRays.end(), rec, rec + 8);
                        gpuIndex.push_back(k);
                        gpuMaterial.push_back(r.ext.materialID);
                        gpuTraceSet.push_back(all[k]->job->traceSet);
                    }
                }
                gpuProof.resize(gpuIndex.size());
                if (xpu2sssgpu::prove(scene->getEmbreeAccelerator(), gpuRays.data(), gpuMaterial.data(), gpuTraceSet.data(),
                                      int(gpuIndex.size()), gpuProof.data())) {
                    for (size_t i = 0; i < gpuIndex.size(); ++i) candidate[gpuIndex[i]] = gpuProof[i];
                }
            }
            const bool gpuVerify = xpu2sssgpu::verify();
            uint64_t verifyChecked = 0, verifyHits = 0;
            for (size_t k = 0; k < all.size(); ++k) {
                if (pendingKind[k]) {
                    xpu2merged::Walk& w = *all[k];
                    if (!candidate[k] && !gpuVerify) {
                        // As Scene::intersectRay would count it.
                        if (w.sssRay.getStart() < w.sssRay.getEnd()) {
                            pbrTls->mStatistics.incCounter(STATS_INTERSECTION_RAYS);
                        }
                        hit[k] = 0;
                        continue;
                    }
                    geomTls->mSubsurfaceTraceSet = w.job->traceSet;
                    hit[k] = scene->intersectRay(pbrTls->mTopLevelTls, w.sssRay, w.isectOut, w.job->lobeType) ? 1 : 0;
                    if (!candidate[k]) {
                        ++verifyChecked; verifyHits += hit[k];
                        static std::atomic<int> printed{0};
                        if (hit[k] && printed++ < 20) {
                            std::fprintf(stderr, "XPU2_SSS_GPU_MISSED material %d traceSet %p geom %d prim %d t %.9g\\n",
                                         w.sssRay.ext.materialID, (const void*)w.job->traceSet,
                                         int(w.sssRay.geomID), int(w.sssRay.primID), w.sssRay.tfar);
                        }
                    }
                }
            }
            if (verifyChecked) {
                xpu2sssgpu::noteVerify(verifyChecked, verifyHits);
            }
'''

DECLS = '''        std::vector<int> pendingKind(all.size(), 0);
        std::vector<char> hit(all.size(), 0);
'''


def patch(sources):
    out = merged.patch(sources)
    t = out['pbr/integrator/PathIntegratorSubsurface.cc']
    t = once(t, INTERSECT, GPU_INTERSECT)
    t = once(t, DECLS, DECLS + '        std::vector<unsigned char> candidate, gpuProof;\n'
                              '        std::vector<float> gpuRays;\n        std::vector<size_t> gpuIndex;\n'
                              '        std::vector<int> gpuMaterial;\n        std::vector<const void*> gpuTraceSet;\n')
    t = t.replace('#include "sss_wavefront.h"\n', '#include "sss_wavefront.h"\n#include "sss_gpu.h"\n', 1)
    out['pbr/integrator/PathIntegratorSubsurface.cc'] = t
    return out
