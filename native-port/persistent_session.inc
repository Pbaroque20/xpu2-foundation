// MoonRayForModo persistent preview protocol v1. Included after the CLI class.
// Scene mutations and start/stop always execute on the driver's owning thread.
#include <filesystem>
#include <fstream>
#include <thread>
#include <atomic>
#include <algorithm>
#include <map>
#include <chrono>
#include <memory>
#include <scene_rdl2/common/fb_util/FbTypes.h>
#include <scene_rdl2/common/fb_util/VariablePixelBuffer.h>
#include <moonray/rendering/rndr/RenderOutputDriver.h>
#include <cctype>
#include <cmath>
#include <limits>
#include <vector>
#include "modo_bucket_output.h"
#include "modo_protocol_line.h"
#include <sstream>
#include "modo_shared_image.h"
#include "modo_scene_memory.h"

void moonray::RaasCommandLineApplication::renderPersistent(const std::string& directory)
{
    namespace fs = std::filesystem;
    // The host that started this renderer may end without telling it to quit, as when it crashes. A renderer left
    // behind would hold its memory and the graphics card for good: so it ends itself the moment its host is gone.
    if (const char* host = std::getenv("MOONRAY_MODO_HOST")) {
        if (HANDLE process = OpenProcess(SYNCHRONIZE, FALSE, static_cast<DWORD>(std::strtoul(host, nullptr, 10)))) {
            std::thread([process]() {
                WaitForSingleObject(process, INFINITE);
                TerminateProcess(GetCurrentProcess(), 0);
            }).detach();
        }
    }
    const auto mailbox = fs::u8path(directory) / "command.txt";
    ModoSceneCommand command(std::getenv("MOONRAY_MODO_COMMAND"));
    auto waiting = [&]() { return command.enabled()?command.ready():fs::exists(mailbox); };
    auto event = [](const char* kind, const std::string& id) {
        modoProtocolLine(std::string("@@MODO_SESSION ") + kind + " " + id);
    };
    std::string id = std::getenv("MOONRAY_MODO_GENERATION") ? std::getenv("MOONRAY_MODO_GENERATION") : "0";
    std::unique_ptr<rndr::RenderContext> context;
    auto load = [&]() {
        context.reset();
        context.reset(new rndr::RenderContext(mOptions, &mInitMessages));
        context->initialize(mInitMessages, rndr::RenderContext::LoggingConfiguration::ATHENA_DISABLED);
        context->setRenderMode(rndr::RenderMode::PROGRESSIVE);
        std::cout << "Modo native progressive preview / worker tiles" << std::endl;
        context->setRenderPrepCallBack({}, waiting);
    };
    load();
    ModoSharedImage sharedImage;
    const bool useShared=std::getenv("MOONRAY_MODO_SHARED")!=nullptr;
    for (;;) {
        event("APPLIED", id);
        bool complete = false;
        bool requireReload = false;
        if (!waiting()) {
            {
                // A scene that asks for its picture to be saved as it renders, or to be taken up from a
                // save, is rendered the way that does so. Its picture is shown as it comes on all the same.
                const auto& vars = context->getSceneContext().getSceneVariables();
                const bool saves = vars.get(scene_rdl2::rdl2::SceneVariables::sCheckpointActive) ||
                                   vars.get(scene_rdl2::rdl2::SceneVariables::sResumeRender);
                context->setRenderMode(saves ? rndr::RenderMode::PROGRESS_CHECKPOINT : rndr::RenderMode::PROGRESSIVE);
                if (saves) std::cout << "Modo session: saving the picture as it renders" << std::endl;
            }
            {
                // Whether this render takes the tiles with the most in them first: "1" in important.txt.
                std::ifstream important(fs::u8path(directory)/"important.txt");
                std::string asked;
                rndr::getRenderDriver()->setModoImportant(bool(important>>asked) && asked=="1");
            }
            const auto prep = context->startFrame();
            if (prep == rndr::RenderContext::RP_RESULT::FINISHED) {
                auto driver = rndr::getRenderDriver();
                const char* tileNames[]={"Top to bottom","Bottom to top","Left to right","Right to left","Morton","Random","Spiral square","Spiral rectangle","Morton shift / flip"};
                const unsigned tileOrder=driver->getFrameState().mTileSchedulerType;
                std::cout << "MoonRay tile dispatch order: " << (tileOrder<9?tileNames[tileOrder]:"Unknown")
                          << "; adaptive skipping and parallel completion can appear scattered." << std::endl;
                auto lastSnapshot = std::chrono::steady_clock::now();
                unsigned lastFilmActivity=0;
                bool published=false;
                std::string lastView;
                auto selectedView=[&]() {
                    std::string key="beauty",selected;
                    std::ifstream view(fs::u8path(directory)/"view.txt");
                    if(std::getline(view,selected) && !selected.empty() && selected.size()<=64 &&
                       std::all_of(selected.begin(),selected.end(),[](unsigned char c){return std::isalnum(c)||c=='_';}))key=selected;
                    return key;
                };
                // What the host asks to be handed at one asking, as the render goes on: its number, then the
                // views. Asked for when a render is stopped part way and its picture is to be kept.
                auto heldRequest=[&]() {
                    std::vector<std::string> keys;
                    std::ifstream hold(fs::u8path(directory)/"hold.txt");
                    std::string asked,key;
                    if(!(hold>>asked) || asked!=id)return keys;
                    while(hold>>key && keys.size()<16)
                        if(key.size()<=64 && std::all_of(key.begin(),key.end(),[](unsigned char c){return std::isalnum(c)||c=='_';}))keys.push_back(key);
                    return keys;
                };
                // Asked to finish with what it has in hand: the tiles the workers are on are rendered to the end,
                // the light of their samples comes in, and the frame ends there as a finished one, with every
                // file of it written. Asked for when a render is stopped part way and its picture is to be kept:
                // stopped at once, the tiles then in hand were left dark and coarse.
                auto drainRequest=[&]() {
                    std::ifstream drain(fs::u8path(directory)/"drain.txt");
                    std::string asked;
                    return bool(drain>>asked) && asked==id;
                };
                bool draining=false;
                std::vector<std::string> holdKeys;
                bool reportedMissingSamples=false;
                // The last value shown for each pixel of a view. A pixel a worker was adding to
                // while its picture was taken can be caught part way, lighter or darker than it
                // is; such a pixel keeps the value it last had when it was at rest.
                // Kept for each view that has been looked at, so that coming back to one (as when the beauty
                // and the denoiser's guides are asked for in turn at a stop) finds what it had.
                struct HeldView { std::vector<float> steady, sampledBefore; };
                std::map<std::string,HeldView> heldViews;
                // How much each pixel had been sampled when its picture was last taken. With the graphics
                // card doing part of the work, a sample is counted before all of its light has come back,
                // and for that while the pixel reads too dark: a pixel is shown anew only once it has been
                // left alone from one picture to the next, by when its light is all in.
                // The frame is finished: every pixel is as it will be, and is taken as it is.
                bool finished=false;

                auto snapshot = [&](const std::string& key) -> bool {
                    if(useShared && !sharedImage.ready())return false;
                    const auto preview=fs::u8path(directory)/("preview_"+id+"_"+key+".pfm");
                    if (fs::exists(preview) || !context->isFrameReadyForDisplay()) return false;
                    // Snapshot sample coverage BEFORE color. Coverage alpha is unrelated:
                    // the environment can have zero alpha and valid, sampled RGB.
                    scene_rdl2::fb_util::FloatBuffer sampleWeights;
                    driver->snapshotWeightBuffer(&sampleWeights, true, false);
                    scene_rdl2::fb_util::RenderBuffer pixels;
                    context->snapshotRenderBuffer(&pixels, true, false, false);
                    if (!pixels.getWidth() || !pixels.getHeight()) return false;
                    scene_rdl2::fb_util::VariablePixelBuffer aov;
                    unsigned channels=0;
                    if (key!="beauty") {
                        const auto* outputs=context->getRenderOutputDriver();
                        for (unsigned i=0;i<outputs->getNumberOfRenderOutputs();++i) {
                            if (outputs->getRenderOutput(i)->getName()!="/modo/preview/"+key) continue;
                            if (outputs->requiresDeepBuffer(i)||outputs->requiresCryptomatteBuffer(i)) break;
                            scene_rdl2::fb_util::HeatMapBuffer heat;
                            scene_rdl2::fb_util::FloatBuffer weight;
                            scene_rdl2::fb_util::RenderBuffer odd;
                            if(outputs->requiresHeatMap(i))context->snapshotHeatMapBuffer(&heat,true,false);
                            if(outputs->requiresWeightBuffer(i))context->snapshotWeightBuffer(&weight,true,false);
                            if(outputs->requiresRenderBufferOdd(i))context->snapshotRenderBufferOdd(&odd,true,false);
                            context->snapshotRenderOutput(&aov,i,&pixels,&pixels,
                                outputs->requiresHeatMap(i)?&heat:nullptr,outputs->requiresWeightBuffer(i)?&weight:nullptr,
                                outputs->requiresRenderBufferOdd(i)?&odd:nullptr,true,false);
                            channels=outputs->getNumberOfChannels(i);break;
                        }
                        if (!channels || channels>3 || aov.getWidth()!=pixels.getWidth() || aov.getHeight()!=pixels.getHeight()) return false;
                    }
                    if(sampleWeights.getWidth()!=pixels.getWidth() || sampleWeights.getHeight()!=pixels.getHeight())return false;
                    // Coverage again, after the colours: where it differs from before, a worker
                    // touched the pixel in between.
                    scene_rdl2::fb_util::FloatBuffer weightsAfter;
                    driver->snapshotWeightBuffer(&weightsAfter, true, false);
                    const bool compare=weightsAfter.getWidth()==pixels.getWidth() && weightsAfter.getHeight()==pixels.getHeight();
                    const size_t count=size_t(pixels.getWidth())*pixels.getHeight();
                    if(!heldViews.count(key) && heldViews.size()>=6)heldViews.clear();
                    HeldView& held=heldViews[key];
                    std::vector<float>& steady=held.steady;
                    std::vector<float>& sampledBefore=held.sampledBefore;
                    if(steady.size()!=count*3){
                        steady.assign(count*3,std::numeric_limits<float>::quiet_NaN());
                        sampledBefore.assign(count,std::numeric_limits<float>::quiet_NaN());
                    }
                    size_t missingSamples=0;
                    auto readPixel=[&](size_t i,float* rgb) {
                        const unsigned x=i%pixels.getWidth(),y=i/pixels.getWidth();
                        const auto& pixel=pixels.getRow(y)[x];
                        rgb[0]=pixel[0];rgb[1]=pixel[1];rgb[2]=pixel[2];
                        if(channels){const float* value=reinterpret_cast<const float*>(aov.getData())+i*channels;
                            rgb[0]=value[0];rgb[1]=channels>1?value[1]:value[0];rgb[2]=channels>2?value[2]:(channels==1?value[0]:0);}
                        // Preserve MoonRay's coarse-pass reconstruction; do not replace
                        // it using a weight snapshot captured at a different instant.
                        if(!(sampleWeights.getRow(y)[x]>0.f))++missingSamples;
                        float* kept=&steady[i*3];
                        const float sampled=sampleWeights.getRow(y)[x];
                        // What a pixel lacks while it is being added to is at most the light of the samples
                        // counted since its picture was last taken. Where those are a small share of all it
                        // has, it is as good as right and is shown as it is: held back, a pixel that is
                        // added to a little at every picture, as all are when the picture is being saved as
                        // it renders, would stay at what it was long before. Where they are a large share
                        // (a pass that doubles a tile's samples leaves it half as bright for a moment) it is
                        // held at what it last showed. One given its first samples since the last picture is
                        // held at the stand-in MoonRay drew for it while it had none, where there was one.
                        const float before=sampledBefore[i];
                        const float after=compare?weightsAfter.getRow(y)[x]:sampled;
                        const bool known=before>=0.f;   // not so for the first picture of a view
                        const float added=std::max(after-sampled,0.f)+(known?std::max(sampled-before,0.f):0.f);
                        const bool moving=!finished && known && added>0.03f*std::max(after,sampled);
                        sampledBefore[i]=sampled;
                        if(moving && !std::isnan(kept[0]) && (before>0.f || kept[0]+kept[1]+kept[2]>0.f)){
                            rgb[0]=kept[0];rgb[1]=kept[1];rgb[2]=kept[2];
                        } else {kept[0]=rgb[0];kept[1]=rgb[1];kept[2]=rgb[2];}
                    };
                    auto reportMissing=[&]() {
                        if(missingSamples && !reportedMissingSamples){
                            std::cerr << "MoonRay preview: coarse pixels use native reconstruction; worker outlines indicate sampled tile activity." << std::endl;
                            reportedMissingSamples=true;
                        }
                    };
                    if(useShared && sharedImage.publish(id,key,pixels.getWidth(),pixels.getHeight(),readPixel)){reportMissing();return true;}
                    const auto temporary = fs::u8path(directory) / ("preview_" + id + ".tmp");
                    std::ofstream file(temporary, std::ios::binary);
                    file << "PF\n" << pixels.getWidth() << " " << pixels.getHeight() << "\n-1.0\n";
                    for (size_t i=0; i<size_t(pixels.getWidth())*pixels.getHeight(); ++i) {
                        float rgb[3];readPixel(i,rgb);
                        file.write(reinterpret_cast<const char*>(rgb), sizeof(rgb));
                    }
                    reportMissing();
                    file.close();
                    if (!file) { fs::remove(temporary); return false; }
                    fs::rename(temporary, preview);
                    event(("FRAME_"+key).c_str(), id);
                    return true;
                };
                // The tiles the workers are on are told by a thread of their own, thirty times a second:
                // handing a picture over takes this one long enough that their outlines would move in jerks.
                std::atomic<bool> tellTiles{true};
                const unsigned tilesWidth=driver->getWidth(),tilesHeight=driver->getHeight();
                std::thread tiles([&]() {
                    while (tellTiles.load(std::memory_order_relaxed)) {
                        modoBucketOutput(id,tilesWidth,tilesHeight);
                        std::this_thread::sleep_for(std::chrono::milliseconds(33));
                    }
                });
                struct TilesEnd { std::atomic<bool>& on; std::thread& thread;
                    ~TilesEnd() { on.store(false); if (thread.joinable()) thread.join(); } } tilesEnd{tellTiles,tiles};
                while (!context->isFrameComplete() && !waiting()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    if(waiting())break;
                    if(!draining && drainRequest()) {
                        draining=true;
                        context->requestDrain();
                        std::error_code ignored;fs::remove(fs::u8path(directory)/"drain.txt",ignored);
                        modoProtocolLine("Modo session: finishing with what is in hand");
                    }
                    holdKeys=heldRequest();
                    if(!holdKeys.empty()) {
                        // Handed over while the render still runs, and the render ended after: ended first, the
                        // light of the samples then in hand would never come, and their pixels be left dark. A
                        // view waits for the host to have taken the one before it.
                        // Asked for more than once, the views are taken in rounds, a moment apart: a pixel caught
                        // part way in one round is at rest in another, and the host takes the middle of them.
                        for(size_t n=0;n<holdKeys.size();++n) {
                            const std::string& key=holdKeys[n];
                            if(n && key==holdKeys[0])std::this_thread::sleep_for(std::chrono::milliseconds(80));
                            const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(3);
                            bool handed=false;
                            while(!handed && std::chrono::steady_clock::now()<until) {
                                try { handed=snapshot(key); }
                                catch(const std::exception& e){std::cerr << "Modo held snapshot: " << e.what() << std::endl;break;}
                                if(!handed)std::this_thread::sleep_for(std::chrono::milliseconds(10));
                            }
                        }
                        std::error_code ignored;fs::remove(fs::u8path(directory)/"hold.txt",ignored);
                        break;
                    }
                    const auto now=std::chrono::steady_clock::now();
                    const auto filmActivity=context->getFilmActivity();
                    const auto key=selectedView();
                    if ((!published || now-lastSnapshot >= std::chrono::milliseconds(100)) &&
                        (filmActivity!=lastFilmActivity || key!=lastView || !published) && context->isFrameReadyForDisplay()) {
                        printStatusLine(*context, driver->getLastFrameMcrtStartTime(), false);
                        {
                            // How far the frame is, for the host's time estimate.
                            std::size_t submitted=0,total=0;
                            const float fraction=context->getFrameProgressFraction(&submitted,&total);
                            std::ostringstream progress;
                            progress << "@@MODO_PROGRESS " << id << " " << fraction;
                            modoProtocolLine(progress.str());
                        }
                        try {
                            if(snapshot(key)){published=true;lastFilmActivity=filmActivity;lastView=key;}
                        } catch (const std::exception& e) {
                            std::cerr << "Modo progressive snapshot: " << e.what() << std::endl;
                        }
                        lastSnapshot=std::chrono::steady_clock::now();
                    }
                }
                complete = context->isFrameComplete() && !waiting();
                // Show a finished short pass before the output writer performs disk I/O.
                if(complete) {
                    finished=true;
                    try { snapshot(selectedView()); }
                    catch(const std::exception& e){std::cerr << "Modo final snapshot: " << e.what() << std::endl;}
                }
                context->stopFrame();
            } else {
                // Render preparation can leave partial geometry. Recover from the full snapshot.
                requireReload = true;
            }
        }
        if (complete) {
            renderOutput(*context);
            event("DONE", id);
        } else {
            event("CANCELED", id);
        }
        while (!waiting()) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        std::string mode, delta, full;
        {
            std::ifstream file;std::istringstream memory;
            if(command.enabled())memory.str(command.receive());else file.open(mailbox);
            std::istream& input=command.enabled()?static_cast<std::istream&>(memory):static_cast<std::istream&>(file);
            if (!std::getline(input,id) || !std::getline(input,mode) ||
                !std::getline(input,delta) || !std::getline(input,full))
                throw std::runtime_error("Incomplete Modo session command");
        }
        if(!command.enabled())fs::remove(mailbox);
        if (mode == "quit") return;
        if (mode == "full" || (mode == "delta" && requireReload)) {
            // A scene may come as several files, read in turn: its text, then the binary file holding its
            // geometry's long lists (the plugin's rdlb.py). They are named one after another, split by |, which no
            // Windows path holds.
            std::vector<std::string> files;
            for (size_t at = 0;;) {
                const size_t bar = full.find('|', at);
                files.push_back(full.substr(at, bar == std::string::npos ? std::string::npos : bar - at));
                if (bar == std::string::npos) break;
                at = bar + 1;
            }
            if (files.size() > 1) std::cout << "Modo session: scene files " << files.size() << std::endl;
            mOptions.setSceneFiles(files);
            load();
            event("RELOADED", id);
        } else if (mode == "delta") {
            // Use the same update -> startFrame path as the upstream CLI's -deltas.
            // startFrame regenerates changed geometry and rebuilds acceleration as needed.
            context->updateScene(delta);
        } else throw std::runtime_error("Unknown Modo session command");
    }
}
