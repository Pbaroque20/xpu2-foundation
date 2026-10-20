"""Random-walk subsurface: one wavefront over all lanes of a subsurface call.

Builds on xpu2/rndr_sss_wavefront.py (per-lane wavefront, modes 1 and 2) and adds
XPU2_SSS_WAVEFRONT=merged (3) and merged-shadow (4):

CPP_computeRadianceSubsurface, when every active lane is random-walk subsurface,
runs three phases through computeRadiancePathTraceSubsurface (selected by a
one-shot thread-local phase, so nested calls are unaffected):
  Prepare (per lane, lane order): the function's setup and walk creation, kept
          in a per-lane job;
  Waves   (once): every walk of every lane steps together; each ray is traced
          with its own lane's subsurface trace set;
  Finish  (per lane, lane order): the Bsdf state, AOV initialisation, exits in
          walk order and the original epilogue, then the original AOV queueing
          and result writes.
merged-shadow runs the original lane loop for real, then the merged path on
scratch copies (Subpixel, PathVertex, sequence ids, a separate Bsdf, AOV buffers,
no queueing) and compares lane by lane.
"""
from .native_gpu_paths import once
from . import rndr_sss_wavefront as perlane

FILES = ('pbr/integrator/PathIntegratorSubsurface.cc', 'pbr/integrator/PathIntegratorUtil.cc')
HEADERS = ('sss_wavefront.h',)

TYPES = r'''
namespace xpu2merged {
struct LaneJob;
struct Walk {
    LaneJob* job = nullptr;
    int s = 0;
    IntegratorSample2D directionSamples;
    IntegratorSample1D channelSamples, distanceSamples, dwivediDirectionSamples;
    scene_rdl2::math::Color pt, tr, channelSamplingWeights, pdfDistance, pdfDistanceDwivedi, sigmaTPrime;
    scene_rdl2::math::ReferenceFrame inReferenceFrame;
    scene_rdl2::math::Vec3f dir, org, pOut, nOut, dNdx, dNdy;
    scene_rdl2::math::Vec2f uDir;
    float tfar = 0.0f, recipChannelSum = 0.0f, NdotDir = 0.0f, phaseSamplingWeight = 1.0f;
    int chIndex = 0;
    int state = 3;            // 0 zero-scatter next, 2 multi-scatter next, 3 done
    size_t nScatter = 0;
    bool hadScatterEvent = false, choseDwivediSampling = false, reachSurface = false, exited = false;
    Ray sssRay;
    Intersection isectOut;
};
struct LaneJob {
    bool active = false;
    Subpixel* sp = nullptr;
    const PathVertex* pv = nullptr;
    const RayDifferential* ray = nullptr;
    const Intersection* isect = nullptr;
    const VolumeSubsurface* vs = nullptr;
    float rayEpsilon = 0.0f;
    scene_rdl2::math::Color sigmaT, sigmaS, dwivediV0, fresnel;
    int nSubsurfaceSample = 0;
    float time = 0.0f;
    const scene_rdl2::rdl2::Material* sssMaterial = nullptr;
    scene_rdl2::rdl2::EvalNormalFunc evalNormalFunc = nullptr;
    bool evalNormalBinding = false;
    const scene_rdl2::rdl2::TraceSet* traceSet = nullptr;
    unsigned sssSampleID = 0;
    unsigned int samplesEvaluated = 0;
    const scene_rdl2::rdl2::Material* isectMaterial = nullptr;
    BsdfLobe::Type lobeType = BsdfLobe::Type(0);
    std::vector<Walk> walks;
};
struct Batch {
    std::unique_ptr<LaneJob> lanes[16];
};
} // namespace xpu2merged

void* xpu2sssCreateBatch() { return new xpu2merged::Batch(); }
void xpu2sssDestroyBatch(void* batch) { delete static_cast<xpu2merged::Batch*>(batch); }

'''

PHASES = r'''
    const int xpu2Phase = xpu2sss::takePhase();
    const int xpu2Lane = xpu2sss::phaseLane;
    if (xpu2Phase == xpu2sss::Prepare) {
        auto* xpu2B = static_cast<xpu2merged::Batch*>(xpu2sss::batch);
        auto job = std::make_unique<xpu2merged::LaneJob>();
        job->active = !(mBssrdfSamples < 1 || !mEnableSSS);
        if (job->active) {
            EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_SSS_INTEGRATION);
            job->sp = &sp; job->pv = &pv; job->ray = &ray; job->isect = &isect; job->vs = &volumeSubsurface;
            job->rayEpsilon = rayEpsilon;
            const scene_rdl2::math::Color albedo = volumeSubsurface.getAlbedo();
            job->sigmaT = volumeSubsurface.getSigmaT();
            job->sigmaS = job->sigmaT * albedo;
            job->dwivediV0 = volumeSubsurface.getDwivediV0();
            job->lobeType = BsdfLobe::Type(BsdfLobe::REFLECTION | BsdfLobe::DIFFUSE);
            job->nSubsurfaceSample = (pv.nonMirrorDepth == 0) ? mBssrdfSamples : 1;
            job->time = ray.time;
            job->sssMaterial = volumeSubsurface.getMaterial();
            job->evalNormalFunc = volumeSubsurface.getEvalNormalFunc();
            job->evalNormalBinding = job->sssMaterial && job->evalNormalFunc;
            auto geomTls = pbrTls->mTopLevelTls->mGeomTls.get();
            job->traceSet = volumeSubsurface.getTraceSet();
            geomTls->mSubsurfaceTraceSet = job->traceSet;
            scene_rdl2::math::Vec3f nIn = job->evalNormalBinding ?
                job->evalNormalFunc(job->sssMaterial, pbrTls->mTopLevelTls->mShadingTls.get(),
                    shading::State(&isect)) :
                    isect.getN();
            job->fresnel = evalFresnel(volumeSubsurface.getTransmissionFresnel(), nIn, -ray.dir);
            job->sssSampleID = sequenceID;
            pbrTls->mStatistics.addToCounter(STATS_SSS_SAMPLES, job->nSubsurfaceSample);
            job->isectMaterial = isect.getMaterial();
            const int nSubsurfaceSample = job->nSubsurfaceSample;
            const unsigned sssSampleID = job->sssSampleID;
            job->walks.resize(nSubsurfaceSample);
            for (int s = 0; s < nSubsurfaceSample; ++s) {
                xpu2merged::Walk& w = job->walks[s];
                w.job = job.get();
                w.s = s;
                w.directionSamples = IntegratorSample2D(SequenceIDIntegrator(
                    sp.mPixel, sp.mSubpixelIndex, SequenceType::BssrdfGlobal, SequenceType::VolumePhase, s, sssSampleID));
                w.channelSamples = IntegratorSample1D(SequenceIDIntegrator(
                    sp.mPixel, sp.mSubpixelIndex, SequenceType::BssrdfLocalChannel, SequenceType::VolumeScattering, s, sssSampleID));
                w.distanceSamples = IntegratorSample1D(SequenceIDIntegrator(
                    sp.mPixel, sp.mSubpixelIndex, SequenceType::BssrdfGlobal, SequenceType::VolumeScattering, s, sssSampleID));
                w.dwivediDirectionSamples = IntegratorSample1D(SequenceIDIntegrator(
                    sp.mPixel, sp.mSubpixelIndex, SequenceType::BssrdfGlobal, SequenceType::VolumeScattering, s, sssSampleID));
                w.pt = volumeSubsurface.getScale() * pv.pathThroughput * job->fresnel / nSubsurfaceSample;
                if (isBlack(w.pt)) {
                    continue;
                }
                w.directionSamples.getSample(&w.uDir[0], pv.nonMirrorDepth);
                const float creaseAttenuation = volumeSubsurface.getCreaseAttenuation();
                if (creaseAttenuation == 0.0f) {
                    w.inReferenceFrame = scene_rdl2::math::ReferenceFrame(isect.getN());
                    w.dir = -w.inReferenceFrame.localToGlobal(sampleLocalHemisphereCosine(w.uDir[0], w.uDir[1]));
                } else {
                    float k1, k2;
                    scene_rdl2::math::Vec3f v1;
                    isect.computePrincipalCurvatures(k1, k2, &v1, nullptr);
                    w.inReferenceFrame = scene_rdl2::math::ReferenceFrame(isect.getN(), v1);
                    if (k1 > 0.0f) {
                        float p = creaseAttenuation * 0.01f;
                        float sinThetaMax1 = 1.0f / (k1*p + 1.0f);
                        float sinThetaMax2 = 1.0f;
                        if (k2 > 0.0f) {
                            sinThetaMax2 = 1.0f / (k2*p + 1.0f);
                        }
                        w.dir = -w.inReferenceFrame.localToGlobal(sampleLocalSphericalEllipticCapCosine(w.uDir[0], w.uDir[1],
                                                                                                      sinThetaMax1, sinThetaMax2));
                    } else {
                        w.dir = -w.inReferenceFrame.localToGlobal(sampleLocalHemisphereCosine(w.uDir[0], w.uDir[1]));
                    }
                }
                w.org = isect.getP() - isect.getNg() * scene_rdl2::math::min(rayEpsilon, sHitEpsilonStart);
                w.state = 0;
            }
        }
        xpu2B->lanes[xpu2Lane] = std::move(job);
        return scene_rdl2::math::Color(0.0f);
    }
    if (xpu2Phase == xpu2sss::Waves) {
        auto* xpu2B = static_cast<xpu2merged::Batch*>(xpu2sss::batch);
        EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_SSS_INTEGRATION);
        const Scene *scene = MNRY_VERIFY(pbrTls->mFs->mScene);
        auto geomTls = pbrTls->mTopLevelTls->mGeomTls.get();
        scene_rdl2::alloc::Arena *arena = pbrTls->mArena;
        SCOPED_MEM(arena);
        constexpr size_t maxScatterEvents = 256;
        const float wBackwardSampling = 0.9f;
        std::vector<xpu2merged::Walk*> all;
        for (auto& job : xpu2B->lanes) {
            if (job && job->active) {
                for (auto& w : job->walks) all.push_back(&w);
            }
        }
        auto setupRay = [&](xpu2merged::Walk& w) {
            const xpu2merged::LaneJob& j = *w.job;
            w.sssRay = Ray(w.org, w.dir, 0.0f, w.tfar, j.time);
            w.sssRay.ext.geomTls = (void*)geomTls;
            if (j.isectMaterial) {
                const shading::Material *materialExt = &j.isectMaterial->get<const shading::Material>();
                w.sssRay.ext.materialID = materialExt->getMaterialId();
            }
        };
        auto exitPoint = [&](xpu2merged::Walk& w) {
            const xpu2merged::LaneJob& j = *w.job;
            const PathVertex& pv = *j.pv;
            w.pOut = w.isectOut.getP();
            if (j.evalNormalBinding && w.isectOut.getMaterial() == j.isectMaterial) {
                w.isectOut.invertIntersectionNormals();
                geom::initIntersectionPhase2(w.isectOut, pbrTls->mTopLevelTls,
                        pv.mirrorDepth, pv.glossyDepth, pv.diffuseDepth,
                        isSubsurfaceAllowed(pv.subsurfaceDepth), pv.minRoughness,
                        w.sssRay.dir);
                RayDifferential rayDiff(*j.ray, w.isectOut.getEpsilonHint(), w.sssRay.tfar);
                w.isectOut.transferAndComputeDerivatives(pbrTls->mTopLevelTls, &rayDiff, j.sp->mTextureDiffScale);
                w.nOut = j.evalNormalFunc(j.sssMaterial, pbrTls->mTopLevelTls->mShadingTls.get(), shading::State(&w.isectOut));
            } else {
                geom::initIntersectionPhase2(w.isectOut, pbrTls->mTopLevelTls,
                        pv.mirrorDepth, pv.glossyDepth, pv.diffuseDepth,
                        isSubsurfaceAllowed(pv.subsurfaceDepth), pv.minRoughness,
                        -w.sssRay.dir);
                w.nOut = -w.isectOut.getN();
            }
            w.dNdx = w.isectOut.getdNdx();
            w.dNdy = w.isectOut.getdNdy();
        };
        auto finishMulti = [&](xpu2merged::Walk& w) {
            if (w.hadScatterEvent) {
                w.job->samplesEvaluated += 1;
            }
            w.exited = w.reachSurface;
            w.state = 3;
        };
        std::vector<int> pendingKind(all.size(), 0);
        std::vector<char> hit(all.size(), 0);
        for (;;) {
            size_t pending = 0;
            for (size_t k = 0; k < all.size(); ++k) {
                xpu2merged::Walk& w = *all[k];
                const xpu2merged::LaneJob& j = *w.job;
                const PathVertex& pv = *j.pv;
                pendingKind[k] = 0;
                if (w.state == 0) {
                    if (isBlack(w.pt)) { w.state = 3; continue; }
                    float uChannel, uDistance;
                    w.channelSamples.getSample(&uChannel, pv.nonMirrorDepth);
                    w.distanceSamples.getSample(&uDistance, pv.nonMirrorDepth);
                    scene_rdl2::math::Color zeroScatterSigmaT = j.vs->getZeroScatterSigmaT();
                    w.recipChannelSum = 1.0f / (w.pt.r + w.pt.g + w.pt.b);
                    int channel = (uChannel <     w.pt.r      * w.recipChannelSum) ? 0 :
                                  (uChannel < (w.pt.r + w.pt.g) * w.recipChannelSum) ? 1 : 2;
                    w.tfar = -scene_rdl2::math::log(1.0f - uDistance) / zeroScatterSigmaT[channel];
                    setupRay(w);
                    pendingKind[k] = 1;
                    ++pending;
                } else if (w.state == 2) {
                    if (w.nScatter >= maxScatterEvents || isBlack(w.pt)) { finishMulti(w); continue; }
                    w.phaseSamplingWeight = (w.nScatter > 1) ? 0.1f : 1.0f;
                    float uChannel;
                    w.channelSamples.getSample(&uChannel, pv.nonMirrorDepth);
                    w.chIndex = sampleChannel(w.pt, uChannel, w.channelSamplingWeights);
                    float uDistance;
                    w.distanceSamples.getSample(&uDistance, pv.nonMirrorDepth);
                    w.NdotDir = dot(w.dir, w.inReferenceFrame.getN());
                    w.sigmaTPrime = j.sigmaT*(1.0f - w.NdotDir / j.dwivediV0);
                    w.tfar = sampleDistanceMultiScatter(j.sigmaT, w.sigmaTPrime,
                        w.channelSamplingWeights, w.chIndex, uDistance, w.NdotDir, j.dwivediV0,
                        w.choseDwivediSampling, w.tr, w.pdfDistance, w.pdfDistanceDwivedi);
                    setupRay(w);
                    pendingKind[k] = 2;
                    ++pending;
                }
            }
            if (!pending) {
                break;
            }
            {
                auto& rep = xpu2sss::report();
                ++rep.mergedWaves; rep.mergedRays += pending;
                const int b = xpu2sss::Report::bin(pending);
                ++rep.waveCount[b]; rep.waveRays[b] += pending;
            }
            for (size_t k = 0; k < all.size(); ++k) {
                if (pendingKind[k]) {
                    xpu2merged::Walk& w = *all[k];
                    geomTls->mSubsurfaceTraceSet = w.job->traceSet;
                    hit[k] = scene->intersectRay(pbrTls->mTopLevelTls, w.sssRay, w.isectOut, w.job->lobeType) ? 1 : 0;
                }
            }
            for (size_t k = 0; k < all.size(); ++k) {
                xpu2merged::Walk& w = *all[k];
                const xpu2merged::LaneJob& j = *w.job;
                const PathVertex& pv = *j.pv;
                if (pendingKind[k] == 1) {
                    if (hit[k] && isValidIntersection(w.isectOut, *j.vs)) {
                        float t1 = w.sssRay.tfar;
                        scene_rdl2::math::Color zeroScatterSigmaT = j.vs->getZeroScatterSigmaT();
                        scene_rdl2::math::Color tr01 = exp(-zeroScatterSigmaT * t1);
                        float exitProbability = (w.pt.r * tr01.r + w.pt.g * tr01.g + w.pt.b * tr01.b) * w.recipChannelSum;
                        scene_rdl2::math::Color tr = exp(-zeroScatterSigmaT * t1);
                        w.pt *= tr / exitProbability;
                        exitPoint(w);
                        w.job->samplesEvaluated += 1;
                        w.exited = true;
                        w.state = 3;
                    } else {
                        w.state = 2;
                        w.nScatter = 0;
                    }
                } else if (pendingKind[k] == 2) {
                    if (hit[k] && isValidIntersection(w.isectOut, *j.vs)) {
                        if (w.hadScatterEvent == false) {
                            finishMulti(w);
                            continue;
                        }
                        w.tr = exp(-j.sigmaT * w.sssRay.tfar);
                        w.reachSurface = true;
                        scene_rdl2::math::Color trp = exp(-w.sigmaTPrime*w.sssRay.tfar);
                        float normalization = 1.0f / (w.pt[0] + w.pt[1] + w.pt[2]);
                        const float psw = w.phaseSamplingWeight;
                        float pdfSurf =
                                w.pt[0] * normalization * (psw*w.tr[0] +(1.0f-psw)*trp[0])+
                                w.pt[1] * normalization * (psw*w.tr[1] +(1.0f-psw)*trp[1])+
                                w.pt[2] * normalization * (psw*w.tr[2] +(1.0f-psw)*trp[2]);
                        w.pt *= w.tr / pdfSurf;
                        exitPoint(w);
                        finishMulti(w);
                    } else {
                        w.hadScatterEvent = true;
                        w.org = w.sssRay.org + w.tfar * w.sssRay.dir;
                        w.pt *= w.tr*j.sigmaS;
                        w.pt *= scene_rdl2::math::sOneOverFourPi;
                        w.directionSamples.getSample(&w.uDir[0], pv.nonMirrorDepth);
                        float backwardDirectionSample;
                        w.dwivediDirectionSamples.getSample(&backwardDirectionSample, pv.nonMirrorDepth);
                        sampleDirection(w.uDir, backwardDirectionSample, w.chIndex,
                            w.phaseSamplingWeight, wBackwardSampling,
                            w.sssRay, j.dwivediV0, w.inReferenceFrame, w.choseDwivediSampling, w.NdotDir, w.dir);
                        float pdfSampling = calculateOneSamplePDF(w.NdotDir, w.sssRay.dir, w.dir,
                            j.dwivediV0, j.vs->getDwivediNormPDF(),
                            w.channelSamplingWeights, w.phaseSamplingWeight, w.pdfDistance, w.pdfDistanceDwivedi);
                        w.pt /= pdfSampling;
                        ++w.nScatter;
                    }
                }
            }
        }
        return scene_rdl2::math::Color(0.0f);
    }
    if (xpu2Phase == xpu2sss::Finish) {
        auto* xpu2B = static_cast<xpu2merged::Batch*>(xpu2sss::batch);
        xpu2merged::LaneJob& job = *xpu2B->lanes[xpu2Lane];
        scene_rdl2::math::Color radiance(0.0f);
        if (!job.active) {
            return radiance;
        }
        float *backupAovs = nullptr;
        const int aovNumChannels = pbrTls->mFs->mAovSchema->numChannels();
        scene_rdl2::alloc::Arena *arena = pbrTls->mArena;
        SCOPED_MEM(arena);
        if (aovs != nullptr) {
            backupAovs = arena->allocArray<float>(aovNumChannels);
            memcpy(backupAovs, aovs, aovNumChannels * sizeof(float));
        }
        EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_SSS_INTEGRATION);
        pbrTls->mTopLevelTls->mGeomTls.get()->mSubsurfaceTraceSet = job.traceSet;
        const int nSubsurfaceSample = job.nSubsurfaceSample;
        unsigned int samplesEvaluated = job.samplesEvaluated;
        const Fresnel* transmissionFresnel = nullptr;
        for (int s = 0; s < nSubsurfaceSample; ++s) {
            xpu2merged::Walk& w = job.walks[s];
            if (!w.exited) {
                continue;
            }
            ssAov += w.pt;
            const bool includeCosineTerm = true, isEntering = true;
            shading::BsdfSlice sliceLocal(w.nOut, -ray.getDirection(),
                                          includeCosineTerm, isEntering,
                                          pbrTls->mFs->mShadowTerminatorFix);
            const bool isReflection = true;
            LambertBsdfLobe lobeLocal(w.nOut, scene_rdl2::math::sWhite, isReflection);
            lobeLocal.setLightSet(nullptr);
            radiance += computeRadianceSubsurfaceSample(
                pbrTls, bsdf, sp, pv, ray, w.dNdx, w.dNdy,
                w.pt, transmissionFresnel, lightSet, lobeLocal, sliceLocal, w.pOut, w.nOut,
                nSubsurfaceSample, 1, s, doIndirect,  rayEpsilon, shadowRayEpsilon,
                sequenceID, sequenceID, true, aovs, isect, Rdl2LightSetList());
        }
        @EPILOGUE@
    }
    if (xpu2sss::mode() >= 3) {
        // Merged modes: calls outside the batch phases run MoonRay's original loop.
        return xpu2Original(sp, sequenceID, ssAov, aovs);
    }
'''

CPP_BEFORE = r'''
    // ---- XPU2 merged-lane random-walk subsurface (modes 3 and 4) ------------------------------
    const int xpu2Mode = xpu2sss::mode();
    shading::Bssrdf* xpu2Bssrdf[VLEN] = {};
    shading::VolumeSubsurface* xpu2Vs[VLEN] = {};
    bool xpu2Use[VLEN] = {};
    bool xpu2Eligible = false;
    if (xpu2Mode >= 3) {
        xpu2Eligible = true;
        for (unsigned i = 0; i < VLEN; ++i) {
            if (!isActive(lanemask, i)) continue;
            xpu2Bssrdf[i] = shading::createBSSRDF(arena, bssrdfv, i);
            xpu2Vs[i] = shading::createVolumeSubsurface(arena, volumeSubsurfacev, i);
            if (!xpu2Bssrdf[i] && !xpu2Vs[i]) continue;
            if (xpu2Bssrdf[i]) xpu2Eligible = false;
            xpu2Use[i] = true;
        }
        if (!xpu2Eligible) ++xpu2sss::report().mergedFallback;
    }
    // Runs one merged call. Real mode passes the RayState's Subpixel/PathVertex, the real
    // sequence ids and the real bsdf, and queues AOVs; shadow mode passes scratch copies.
    auto xpu2RunMerged = [&](shading::Bsdf& bsdfUse, Subpixel** sps, PathVertex** pvs, uint32_t* seqs,
                             scene_rdl2::math::Color* radOut, scene_rdl2::math::Color* ssOut,
                             float* aovBuf, bool queue, std::vector<float>* aovCapture) {
        ++xpu2sss::report().mergedCalls;
        void* batch = xpu2sssCreateBatch();
        xpu2sss::batch = batch;
        int anyLane = -1;
        for (unsigned i = 0; i < VLEN; ++i) {
            if (!xpu2Use[i]) continue;
            RayState *rs = &baseRayState[rayStateIndices[i]];
            const mcrt_common::RayDifferential &ray = rs->mRay;
            const shading::Intersection &isect = *rs->mAOSIsect;
            pvs[i]->subsurfaceDepth += 1;
            scene_rdl2::math::Color ssUnused(0.0f);
            xpu2sss::phase = xpu2sss::Prepare; xpu2sss::phaseLane = int(i);
            pathIntegrator->computeRadiancePathTraceSubsurface(
                    pbrTls, bsdfUse, *sps[i], *pvs[i], ray, isect, *xpu2Vs[i], *lightSet,
                    doIndirect[i], rayEpsilon[i], shadowRayEpsilon[i], seqs[i], ssUnused, aovBuf);
            if (anyLane < 0) anyLane = int(i);
        }
        if (anyLane >= 0) {
            RayState *rs = &baseRayState[rayStateIndices[anyLane]];
            scene_rdl2::math::Color ssUnused(0.0f);
            uint32_t seqUnused = seqs[anyLane];
            xpu2sss::phase = xpu2sss::Waves; xpu2sss::phaseLane = anyLane;
            pathIntegrator->computeRadiancePathTraceSubsurface(
                    pbrTls, bsdfUse, *sps[anyLane], *pvs[anyLane], rs->mRay, *rs->mAOSIsect, *xpu2Vs[anyLane],
                    *lightSet, doIndirect[anyLane], rayEpsilon[anyLane], shadowRayEpsilon[anyLane], seqUnused,
                    ssUnused, aovBuf);
        }
        for (unsigned i = 0; i < VLEN; ++i) {
            if (!xpu2Use[i]) continue;
            bsdfUse.setLabelIds(materialLabelIds, lpeMaterialLabelIds, geomLabelIds[i]);
            bsdfUse.addBssrdf(xpu2Bssrdf[i]);
            bsdfUse.setVolumeSubsurface(xpu2Vs[i]);
            RayState *rs = &baseRayState[rayStateIndices[i]];
            const mcrt_common::RayDifferential &ray = rs->mRay;
            const shading::Intersection &isect = *rs->mAOSIsect;
            if (aovBuf) {
                aovSchema.initFloatArray(aovBuf);
            }
            scene_rdl2::math::Color ssAov = scene_rdl2::math::Color(0);
            scene_rdl2::math::Color radiance = scene_rdl2::math::Color(0);
            xpu2sss::phase = xpu2sss::Finish; xpu2sss::phaseLane = int(i);
            radiance += pathIntegrator->computeRadiancePathTraceSubsurface(
                    pbrTls, bsdfUse, *sps[i], *pvs[i], ray, isect, *xpu2Vs[i], *lightSet,
                    doIndirect[i], rayEpsilon[i], shadowRayEpsilon[i], seqs[i], ssAov, aovBuf);
            if (aovCapture && aovBuf) aovCapture[i].assign(aovBuf, aovBuf + aovNumChannels);
            if (aovBuf && queue) {
                aovAddToBundledQueue(pbrTls, aovSchema, isect, ray,
                    AOV_TYPE_LIGHT_AOV | AOV_TYPE_VISIBILITY_AOV, aovBuf, rs->mSubpixel.mPixel,
                    rs->mDeepDataHandle);
            }
            radOut[i] = radiance;
            ssOut[i] = ssAov;
        }
        xpu2sss::batch = nullptr;
        xpu2sssDestroyBatch(batch);
    };
    if (xpu2Mode == 3 && xpu2Eligible) {
        Subpixel* sps[VLEN] = {}; PathVertex* pvs[VLEN] = {};
        for (unsigned i = 0; i < VLEN; ++i) {
            if (!xpu2Use[i]) continue;
            RayState *rs = &baseRayState[rayStateIndices[i]];
            sps[i] = &rs->mSubpixel; pvs[i] = &rs->mPathVertex;
        }
        scene_rdl2::math::Color rad[VLEN], ss[VLEN];
        xpu2RunMerged(bsdf, sps, pvs, sequenceID, rad, ss, aovs, true, nullptr);
        for (unsigned i = 0; i < VLEN; ++i) {
            if (!xpu2Use[i]) continue;
            results[i] += rad[i].r;
            results[VLEN + i] += rad[i].g;
            results[VLEN * 2 + i] += rad[i].b;
            ssAovResults[i] += ss[i].r;
            ssAovResults[VLEN + i] += ss[i].g;
            ssAovResults[VLEN * 2 + i] += ss[i].b;
        }
        pbrTls->startIspcAccumulator();
        return;
    }
    // merged-shadow: snapshots before the original loop, captures inside it.
    const bool xpu2Shadow = xpu2Mode == 4 && xpu2Eligible;
    Subpixel xpu2SpStart[VLEN]; PathVertex xpu2PvStart[VLEN]; uint32_t xpu2SeqStart[VLEN] = {};
    scene_rdl2::math::Color xpu2RadO[VLEN], xpu2SsO[VLEN];
    std::vector<float> xpu2AovO[VLEN];
    if (xpu2Shadow) {
        for (unsigned i = 0; i < VLEN; ++i) {
            if (!xpu2Use[i]) continue;
            RayState *rs = &baseRayState[rayStateIndices[i]];
            xpu2SpStart[i] = rs->mSubpixel; xpu2PvStart[i] = rs->mPathVertex; xpu2SeqStart[i] = sequenceID[i];
        }
    }
    // ---- end of XPU2 merged-lane setup -------------------------------------------------------
'''

CPP_CAPTURE = r'''
        if (xpu2Shadow) {
            xpu2RadO[i] = radiance; xpu2SsO[i] = ssAov;
            if (aovs) xpu2AovO[i].assign(aovs, aovs + aovNumChannels);
        }
'''

CPP_AFTER = r'''
    if (xpu2Shadow) {
        // Replay the merged path on scratch copies and compare lane by lane.
        Subpixel spS[VLEN]; PathVertex pvS[VLEN]; Subpixel* sps[VLEN] = {}; PathVertex* pvs[VLEN] = {};
        uint32_t seqS[VLEN];
        for (unsigned i = 0; i < VLEN; ++i) {
            seqS[i] = xpu2SeqStart[i];
            if (!xpu2Use[i]) continue;
            spS[i] = xpu2SpStart[i]; pvS[i] = xpu2PvStart[i]; sps[i] = &spS[i]; pvs[i] = &pvS[i];
        }
        std::vector<float> aovScratch(aovs ? aovNumChannels : 0);
        std::vector<float> aovCap[VLEN];
        scene_rdl2::math::Color rad[VLEN], ss[VLEN];
        shading::Bsdf bsdfScratch;
        xpu2RunMerged(bsdfScratch, sps, pvs, seqS, rad, ss, aovs ? aovScratch.data() : nullptr, false, aovCap);
        for (unsigned i = 0; i < VLEN; ++i) {
            if (!xpu2Use[i]) continue;
            RayState *rs = &baseRayState[rayStateIndices[i]];
            const Subpixel& a = rs->mSubpixel; const Subpixel& b = spS[i];
            auto sameBits = [](const void* x, const void* y, size_t n) { return std::memcmp(x, y, n) == 0; };
            bool spSame = a.mPixel == b.mPixel && a.mSubpixelIndex == b.mSubpixelIndex &&
                sameBits(&a.mSubpixelX, &b.mSubpixelX, 4) && sameBits(&a.mSubpixelY, &b.mSubpixelY, 4) &&
                a.mPixelSamples == b.mPixelSamples && sameBits(&a.mSampleClampingValue, &b.mSampleClampingValue, 4) &&
                sameBits(&a.mPrimaryRayDiffScale, &b.mPrimaryRayDiffScale, 4) &&
                sameBits(&a.mTextureDiffScale, &b.mTextureDiffScale, 4) && a.mNumDeferredNodes == b.mNumDeferredNodes;
            for (const DeferredNode *x = a.mDeferredNodesHead, *y = b.mDeferredNodesHead; spSame && (x || y);
                 x = x ? x->mNext : nullptr, y = y ? y->mNext : nullptr) {
                if (!x || !y) { spSame = false; break; }
                if (x == y) continue;
                spSame = sameBits(&x->mThroughput, &y->mThroughput, sizeof(x->mThroughput)) &&
                    x->mNonMirrorDepth == y->mNonMirrorDepth && x->mLobeCount == y->mLobeCount &&
                    x->mBsdfIsSpherical == y->mBsdfIsSpherical && sameBits(&x->mRayEpsilon, &y->mRayEpsilon, 4) &&
                    x->mSequenceID == y->mSequenceID && sameBits(&x->mRayDirFootprint, &y->mRayDirFootprint, 4) &&
                    sameBits(&x->mAlbedo, &y->mAlbedo, sizeof(x->mAlbedo)) && x->mParentLobe == y->mParentLobe;
            }
            if (!sameBits(&rs->mPathVertex, &pvS[i], sizeof(PathVertex))) ++xpu2sss::report().pathVertexDiff;
            const int spFlag = spSame ? 0 : 1, zero = 0;
            xpu2sss::compare(xpu2RadO[i], rad[i], xpu2SsO[i], ss[i], sequenceID[i], seqS[i],
                             aovs ? xpu2AovO[i].data() : nullptr, aovs ? aovCap[i].data() : nullptr,
                             aovs ? int(aovNumChannels) : 0, spFlag, zero);
        }
    }
'''


def patch(sources):
    out = perlane.patch(sources)
    t = out['pbr/integrator/PathIntegratorSubsurface.cc']
    original = sources['pbr/integrator/PathIntegratorSubsurface.cc']
    sig = 'PathIntegrator::computeRadiancePathTraceSubsurface(pbr::TLState* pbrTls,'
    # Epilogue of the original body (after the walk loop), without the function's closing brace.
    start = original.index(sig)
    body_start = original.index('\n{\n', start) + 1
    body_end = original.index('\n}\n', body_start) + 2
    body = original[body_start:body_end]
    epilogue = body[body.index('    // Radiance tweak added by Priyamvad'):]
    if not epilogue.rstrip().endswith('}'):
        raise ValueError('epilogue drift')
    epilogue = epilogue.rstrip()[:-1]
    phases = PHASES.replace('@EPILOGUE@', '{\n' + epilogue + '        }\n')
    t = once(t, '    const int xpu2Mode = xpu2sss::mode();\n    if (xpu2Mode == 0) {', phases + '    const int xpu2Mode = xpu2sss::mode();\n    if (xpu2Mode == 0) {')
    t = once(t, 'scene_rdl2::math::Color\nPathIntegrator::computeRadiancePathTraceSubsurface(',
             TYPES + 'scene_rdl2::math::Color\nPathIntegrator::computeRadiancePathTraceSubsurface(')
    t = t.replace('#include "sss_wavefront.h"\n', '#include "sss_wavefront.h"\n#include <memory>\n', 1)
    out['pbr/integrator/PathIntegratorSubsurface.cc'] = t

    u = sources['pbr/integrator/PathIntegratorUtil.cc']
    u = u.replace('#include', '#include "sss_wavefront.h"\n#include <cstring>\n#include <vector>\n#include', 1)
    start = u.index('CPP_computeRadianceSubsurface(const PathIntegrator * pathIntegrator,')
    end = u.index('\n}\n', start)
    f = u[start:end]
    f = once(f, '    for (unsigned i = 0; i < VLEN; ++i) {\n        // Don\'t compute subsurface radiance for invalid lanes.',
             CPP_BEFORE + '    for (unsigned i = 0; i < VLEN; ++i) {\n        // Don\'t compute subsurface radiance for invalid lanes.')
    f = once(f, '        results[i] += radiance.r;\n', CPP_CAPTURE + '        results[i] += radiance.r;\n')
    f = once(f, '    pbrTls->startIspcAccumulator();\n}', CPP_AFTER + '    pbrTls->startIspcAccumulator();\n}') if f.count('    pbrTls->startIspcAccumulator();\n}') == 1 else f
    if CPP_AFTER not in f:
        # The function's last statement: insert before its final startIspcAccumulator.
        k = f.rindex('    pbrTls->startIspcAccumulator();')
        f = f[:k] + CPP_AFTER + f[k:]
    out['pbr/integrator/PathIntegratorUtil.cc'] = u[:start] + f + u[end:]
    return out
