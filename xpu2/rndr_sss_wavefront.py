"""Random-walk subsurface as an in-call wavefront (Embree only), behind a switch.

PathIntegrator::computeRadiancePathTraceSubsurface keeps its name. Its original
body becomes one lambda and a wavefront body (same prologue and epilogue, new
walk loop) another; XPU2_SSS_WAVEFRONT picks one, or "shadow" runs both and
compares them (xpu2/sss_wavefront.h).

The wavefront: every walk of the lane is a small state machine; each wave
prepares the next ray of every active walk, traces them (Embree, in walk order),
then consumes the results. A walk that exits stores pt, pOut, nOut, dNdx and dNdy;
after all walks finish, exits are shaded in the original walk order, so the
radiance and subsurface-AOV sums, the AOV accumulation and sequenceID use are
unchanged. Each walk draws from its own sequences (fixed sssSampleID), so
interleaving cannot change a walk's samples.
"""
from .native_gpu_paths import once

FILES = ('pbr/integrator/PathIntegratorSubsurface.cc',)
HEADERS = ('sss_wavefront.h',)

LOOP = r'''
    // ---- XPU2 wavefront walk loop ----------------------------------------------------------------
    struct Walk {
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
    constexpr size_t maxScatterEvents = 256;
    const scene_rdl2::rdl2::Material* xpu2IsectMaterial = isect.getMaterial();
    std::vector<Walk> walks(nSubsurfaceSample);
    for (int s = 0; s < nSubsurfaceSample; ++s) {
        Walk& w = walks[s];
        w.directionSamples = IntegratorSample2D(SequenceIDIntegrator(
            sp.mPixel, sp.mSubpixelIndex, SequenceType::BssrdfGlobal, SequenceType::VolumePhase, s, sssSampleID));
        w.channelSamples = IntegratorSample1D(SequenceIDIntegrator(
            sp.mPixel, sp.mSubpixelIndex, SequenceType::BssrdfLocalChannel, SequenceType::VolumeScattering, s, sssSampleID));
        w.distanceSamples = IntegratorSample1D(SequenceIDIntegrator(
            sp.mPixel, sp.mSubpixelIndex, SequenceType::BssrdfGlobal, SequenceType::VolumeScattering, s, sssSampleID));
        w.dwivediDirectionSamples = IntegratorSample1D(SequenceIDIntegrator(
            sp.mPixel, sp.mSubpixelIndex, SequenceType::BssrdfGlobal, SequenceType::VolumeScattering, s, sssSampleID));
        w.pt = volumeSubsurface.getScale() * pv.pathThroughput * fresnel / nSubsurfaceSample;
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

    auto setupRay = [&](Walk& w) {
        w.sssRay = Ray(w.org, w.dir, 0.0f, w.tfar, time);
        w.sssRay.ext.geomTls = (void*)geomTls;
        if (xpu2IsectMaterial) {
            const shading::Material *materialExt = &xpu2IsectMaterial->get<const shading::Material>();
            w.sssRay.ext.materialID = materialExt->getMaterialId();
        }
    };
    // Exit bookkeeping shared by the zero- and multi-scatter hits: normals and derivatives.
    auto exitPoint = [&](Walk& w) {
        w.pOut = w.isectOut.getP();
        if (evalNormalBinding && w.isectOut.getMaterial() == xpu2IsectMaterial) {
            w.isectOut.invertIntersectionNormals();
            geom::initIntersectionPhase2(w.isectOut, pbrTls->mTopLevelTls,
                    pv.mirrorDepth, pv.glossyDepth, pv.diffuseDepth,
                    isSubsurfaceAllowed(pv.subsurfaceDepth), pv.minRoughness,
                    w.sssRay.dir);
            RayDifferential rayDiff(ray, w.isectOut.getEpsilonHint(), w.sssRay.tfar);
            w.isectOut.transferAndComputeDerivatives(pbrTls->mTopLevelTls, &rayDiff, sp.mTextureDiffScale);
            w.nOut = evalNormalFunc(sssMaterial, pbrTls->mTopLevelTls->mShadingTls.get(), shading::State(&w.isectOut));
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
    auto finishMulti = [&](Walk& w) {
        if (w.hadScatterEvent) {
            samplesEvaluated += 1;
        }
        w.exited = w.reachSurface;
        w.state = 3;
    };

    std::vector<int> pendingKind(nSubsurfaceSample, 0);   // 1 zero-scatter ray, 2 multi-scatter ray
    std::vector<char> hit(nSubsurfaceSample, 0);
    for (;;) {
        bool any = false;
        for (int s = 0; s < nSubsurfaceSample; ++s) {
            Walk& w = walks[s];
            pendingKind[s] = 0;
            if (w.state == 0) {
                // Zero-scatter event (one at most).
                if (isBlack(w.pt)) { w.state = 3; continue; }
                float uChannel, uDistance;
                w.channelSamples.getSample(&uChannel, pv.nonMirrorDepth);
                w.distanceSamples.getSample(&uDistance, pv.nonMirrorDepth);
                scene_rdl2::math::Color zeroScatterSigmaT = volumeSubsurface.getZeroScatterSigmaT();
                w.recipChannelSum = 1.0f / (w.pt.r + w.pt.g + w.pt.b);
                int channel = (uChannel <     w.pt.r      * w.recipChannelSum) ? 0 :
                              (uChannel < (w.pt.r + w.pt.g) * w.recipChannelSum) ? 1 : 2;
                w.tfar = -scene_rdl2::math::log(1.0f - uDistance) / zeroScatterSigmaT[channel];
                setupRay(w);
                pendingKind[s] = 1;
                any = true;
            } else if (w.state == 2) {
                if (w.nScatter >= maxScatterEvents || isBlack(w.pt)) { finishMulti(w); continue; }
                w.phaseSamplingWeight = (w.nScatter > 1) ? 0.1f : 1.0f;
                float uChannel;
                w.channelSamples.getSample(&uChannel, pv.nonMirrorDepth);
                w.chIndex = sampleChannel(w.pt, uChannel, w.channelSamplingWeights);
                float uDistance;
                w.distanceSamples.getSample(&uDistance, pv.nonMirrorDepth);
                w.NdotDir = dot(w.dir, w.inReferenceFrame.getN());
                w.sigmaTPrime = sigmaT*(1.0f - w.NdotDir / dwivediV0);
                w.tfar = sampleDistanceMultiScatter(sigmaT, w.sigmaTPrime,
                    w.channelSamplingWeights, w.chIndex, uDistance, w.NdotDir, dwivediV0,
                    w.choseDwivediSampling, w.tr, w.pdfDistance, w.pdfDistanceDwivedi);
                setupRay(w);
                pendingKind[s] = 2;
                any = true;
            }
        }
        if (!any) {
            break;
        }
        xpu2sss::report().waves++;
        // Trace the wave (Embree, in walk order).
        for (int s = 0; s < nSubsurfaceSample; ++s) {
            if (pendingKind[s]) {
                Walk& w = walks[s];
                hit[s] = scene->intersectRay(pbrTls->mTopLevelTls, w.sssRay, w.isectOut, lobeType) ? 1 : 0;
            }
        }
        // Consume.
        for (int s = 0; s < nSubsurfaceSample; ++s) {
            Walk& w = walks[s];
            if (pendingKind[s] == 1) {
                if (hit[s] && isValidIntersection(w.isectOut, volumeSubsurface)) {
                    float t1 = w.sssRay.tfar;
                    scene_rdl2::math::Color zeroScatterSigmaT = volumeSubsurface.getZeroScatterSigmaT();
                    scene_rdl2::math::Color tr01 = exp(-zeroScatterSigmaT * t1);
                    float exitProbability = (w.pt.r * tr01.r + w.pt.g * tr01.g + w.pt.b * tr01.b) * w.recipChannelSum;
                    scene_rdl2::math::Color tr = exp(-zeroScatterSigmaT * t1);
                    w.pt *= tr / exitProbability;
                    exitPoint(w);
                    samplesEvaluated += 1;
                    w.exited = true;
                    w.state = 3;
                } else {
                    w.state = 2;
                    w.nScatter = 0;
                }
            } else if (pendingKind[s] == 2) {
                if (hit[s] && isValidIntersection(w.isectOut, volumeSubsurface)) {
                    if (w.hadScatterEvent == false) {
                        finishMulti(w);
                        continue;
                    }
                    w.tr = exp(-sigmaT * w.sssRay.tfar);
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
                    w.pt *= w.tr*sigmaS;
                    w.pt *= scene_rdl2::math::sOneOverFourPi;
                    w.directionSamples.getSample(&w.uDir[0], pv.nonMirrorDepth);
                    float backwardDirectionSample;
                    w.dwivediDirectionSamples.getSample(&backwardDirectionSample, pv.nonMirrorDepth);
                    sampleDirection(w.uDir, backwardDirectionSample, w.chIndex,
                        w.phaseSamplingWeight, wBackwardSampling,
                        w.sssRay, dwivediV0, w.inReferenceFrame, w.choseDwivediSampling, w.NdotDir, w.dir);
                    float pdfSampling = calculateOneSamplePDF(w.NdotDir, w.sssRay.dir, w.dir,
                        dwivediV0, volumeSubsurface.getDwivediNormPDF(),
                        w.channelSamplingWeights, w.phaseSamplingWeight, w.pdfDistance, w.pdfDistanceDwivedi);
                    w.pt /= pdfSampling;
                    ++w.nScatter;
                }
            }
        }
    }
    xpu2sss::report().walks += uint64_t(nSubsurfaceSample);

    // Shade the exits in walk order, exactly as the sequential loop does.
    for (int s = 0; s < nSubsurfaceSample; ++s) {
        Walk& w = walks[s];
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
    // ---- end of XPU2 wavefront walk loop ---------------------------------------------------------

'''

PARAMS = '(Subpixel &sp, unsigned &sequenceID, scene_rdl2::math::Color &ssAov, float* aovs) -> scene_rdl2::math::Color'

DISPATCH = r'''
    const int xpu2Mode = xpu2sss::mode();
    if (xpu2Mode == 0) {
        return xpu2Original(sp, sequenceID, ssAov, aovs);
    }
    if (xpu2Mode == 1) {
        return xpu2Wavefront(sp, sequenceID, ssAov, aovs);
    }
    // Shadow mode: the original runs for real; the wavefront runs from the same
    // starting state into scratch copies and is compared bit for bit.
    const int xpu2AovN = aovs ? pbrTls->mFs->mAovSchema->numChannels() : 0;
    std::vector<float> xpu2AovStart(aovs ? aovs : nullptr, aovs ? aovs + xpu2AovN : nullptr);
    const Subpixel xpu2SpStart = sp;
    const unsigned xpu2SeqStart = sequenceID;
    const scene_rdl2::math::Color xpu2SsStart = ssAov;
    const scene_rdl2::math::Color xpu2R0 = xpu2Original(sp, sequenceID, ssAov, aovs);
    Subpixel xpu2SpShadow = xpu2SpStart;
    unsigned xpu2SeqShadow = xpu2SeqStart;
    scene_rdl2::math::Color xpu2SsShadow = xpu2SsStart;
    std::vector<float> xpu2AovShadow = xpu2AovStart;
    const scene_rdl2::math::Color xpu2R1 = xpu2Wavefront(xpu2SpShadow, xpu2SeqShadow, xpu2SsShadow,
                                                         aovs ? xpu2AovShadow.data() : nullptr);
    // Subpixel holds arena pointers (deferred-node list, Cryptomatte buffers): compare
    // its values and the deferred nodes' numeric content, not addresses.
    auto xpu2SameBits = [](const void* a, const void* b, size_t n) { return std::memcmp(a, b, n) == 0; };
    bool xpu2SpSame = sp.mPixel == xpu2SpShadow.mPixel && sp.mSubpixelIndex == xpu2SpShadow.mSubpixelIndex &&
        xpu2SameBits(&sp.mSubpixelX, &xpu2SpShadow.mSubpixelX, 4) && xpu2SameBits(&sp.mSubpixelY, &xpu2SpShadow.mSubpixelY, 4) &&
        sp.mPixelSamples == xpu2SpShadow.mPixelSamples &&
        xpu2SameBits(&sp.mSampleClampingValue, &xpu2SpShadow.mSampleClampingValue, 4) &&
        xpu2SameBits(&sp.mPrimaryRayDiffScale, &xpu2SpShadow.mPrimaryRayDiffScale, 4) &&
        xpu2SameBits(&sp.mTextureDiffScale, &xpu2SpShadow.mTextureDiffScale, 4) &&
        sp.mNumDeferredNodes == xpu2SpShadow.mNumDeferredNodes &&
        (sp.mCryptomatteBuffer == nullptr) == (xpu2SpShadow.mCryptomatteBuffer == nullptr);
    for (const DeferredNode *a = sp.mDeferredNodesHead, *b = xpu2SpShadow.mDeferredNodesHead;
         xpu2SpSame && (a || b); a = a ? a->mNext : nullptr, b = b ? b->mNext : nullptr) {
        // Nodes present before this call are shared by both lists; compare the rest.
        if (!a || !b) { xpu2SpSame = false; break; }
        if (a == b) continue;
        xpu2SpSame = xpu2SameBits(&a->mThroughput, &b->mThroughput, sizeof(a->mThroughput)) &&
            a->mNonMirrorDepth == b->mNonMirrorDepth && a->mLobeCount == b->mLobeCount &&
            a->mBsdfIsSpherical == b->mBsdfIsSpherical && xpu2SameBits(&a->mRayEpsilon, &b->mRayEpsilon, 4) &&
            a->mSequenceID == b->mSequenceID && xpu2SameBits(&a->mRayDirFootprint, &b->mRayDirFootprint, 4) &&
            xpu2SameBits(&a->mAlbedo, &b->mAlbedo, sizeof(a->mAlbedo)) && a->mParentLobe == b->mParentLobe;
    }
    const int xpu2Flag = xpu2SpSame ? 0 : 1, xpu2Zero = 0;
    xpu2sss::compare(xpu2R0, xpu2R1, ssAov, xpu2SsShadow, sequenceID, xpu2SeqShadow,
                     aovs, xpu2AovShadow.data(), xpu2AovN, xpu2Flag, xpu2Zero);
    return xpu2R0;
'''


def patch(sources):
    t = sources['pbr/integrator/PathIntegratorSubsurface.cc']
    t = t.replace('#include', '#include "sss_wavefront.h"\n#include <vector>\n#include', 1)
    sig = 'PathIntegrator::computeRadiancePathTraceSubsurface(pbr::TLState* pbrTls,'
    start = t.index(sig)
    body_start = t.index('\n{\n', start) + 1
    body_end = t.index('\n}\n', body_start) + 2          # index just after the closing brace
    body = t[body_start:body_end]
    loop_start = body.index('    for (int s = 0; s < nSubsurfaceSample; ++s) {\n')
    epilogue_start = body.index('    // Radiance tweak added by Priyamvad')
    if body.count('    for (int s = 0; s < nSubsurfaceSample; ++s) {\n') != 1:
        raise ValueError('walk loop drift')
    wavefront = body[:loop_start] + LOOP + body[epilogue_start:]
    new_body = ('{\n'
                '    auto xpu2Original = [&]' + PARAMS + '\n' + body + ';\n'
                '    auto xpu2Wavefront = [&]' + PARAMS + '\n' + wavefront + ';\n'
                + DISPATCH + '}\n')
    t = t[:body_start] + new_body + t[body_end:]
    return {**sources, 'pbr/integrator/PathIntegratorSubsurface.cc': t}
