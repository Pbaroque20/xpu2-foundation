# XPU2 tools kept outside the build

- kit/: **the kit builder is retired** (build_xpu2_kit.py, with install_xpu2.py, made and installed a separate XPU2 test kit). XPU2 is part of the standard kit since 0.5, and tools/package_development_release.py packages it; both scripts stay as a record. check_xpu2_session_qt.py is the headless Qt check.
- modo-cl/: headless Modo export and comparison (setup_profile, run_cl, export_probe, preview_probe, timing_probe for where the kit's time goes, compare, run_watched), and the Modo crash debugger (debug_modo.py; never gdb, whose JIT breakpoint on nvgpucomp64 ends Modo). They find the session harness in this tree; their results go to the test-results folder each names at its top.
