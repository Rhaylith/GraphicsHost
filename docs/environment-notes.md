# Development machine notes

Things found while building on the main development machine (Windows 11, AMD Ryzen Threadripper PRO 7995WX,
Radeon RX 7900 XTX). They are environment facts, not project bugs.

## Application Control may block freshly built binaries

Windows Code Integrity (an Application Control policy, ID `{0283ac0f-fff1-49ae-ada1-8a933130cad6}`) can refuse to load
unsigned binaries the build just produced. On 2026-10-22 the first build of `sd-cpu-avx2.dll` was refused
(`LoadLibrary` error 4551, Code Integrity events 3033/3077 in `Microsoft-Windows-CodeIntegrity/Operational`) while
`sd-vulkan.dll` from the same build loaded. Rebuilding the DLL produced a file that loaded.

Consequences and rules:

- If a test or the app fails to start with error 4551, check that event log before debugging the code. A rebuild may be enough.
- Never work around the policy (no disabling it, no allow-listing by script). If it keeps blocking, the proper fix is to
  sign the binaries (Phase 6 installer work) or to have the policy owner allow the build folder.
- The LLMHost project hit the same problem ("tests written, not run: blocked by Application Control").

## GPU driver state

As of 2026-10-22 the RX 7900 XTX is in Device Manager error `CM_PROB_FAILED_ADD` (code 31), so Vulkan cannot create an
instance and GPU runs are untested; see [phase0-results.md](phase0-results.md). The session is also an RDP session
(Remote Display Adapter), which can hide GPUs from Vulkan.

## Shell quirks

Some multi-line commands with embedded quotes fail in the bash tool on this machine; write such files with the editor
instead of heredocs.

### Log of blocks seen (2026-10-22)

| Time | File | Result |
|---|---|---|
| 21:28 | `sd-cpu-avx2.dll` (first build) | blocked; a rebuilt copy loaded |
| ~21:35 | `GraphicsHost.exe` (first build, Qt only) | blocked at launch |
| 21:38 | `spdlog.dll` (vcpkg build, loaded by the rebuilt `GraphicsHost.exe`) | blocked (events 3033/3077); the exe itself started |

The pattern looks like a per-file verdict (reputation or managed-installer based) that changes over time, not a rule on
folders or extensions. Vcpkg's DLLs are unsigned and locally built, so any of them can be affected. If blocks persist the
options are signing the build output, or having the policy owner allow the build folder; both are decisions for the
machine's owner, not something the build scripts should try to avoid.
| 21:40 | `spdlog.dll` again (loaded by `GraphicsHost.Tests.exe`) | blocked again; Qt DLLs, the runtime DLLs and the spike all load |
| 21:42 | `GraphicsHost.Tests.exe` (first build) | blocked at launch |

### Verification that is still owed (Phase 1)

Everything below builds, but could not be run on 2026-10-22 because of the blocks above. Run these once the binaries
load (a rebuild, signing, or an allow rule from the machine's owner), and tick Phase 1.9 in the plan when they pass:

```
build\bin\x64\Release\GraphicsHost.Tests.exe "[unit]"          # about 45 test cases: settings, migrations, log, engine selection, CLI
build\bin\x64\Release\GraphicsHost.Tests.exe "[integration]"   # loads a real runtime DLL
build\bin\x64\Release\GraphicsHost.exe --json hardware          # CLI path of the app
build\bin\x64\Release\GraphicsHost.exe runtime info             # app loads the selected runtime
build\bin\x64\Release\GraphicsHost.exe --smoke-test             # opens the Qt window and exits (QT_QPA_PLATFORM=offscreen is fine)
```

### Update 2026-10-23: blocks resolved, GPU works

After the build output was allowed or signed (done by the machine's owner), the owed verification ran:

* `GraphicsHost.Tests.exe` (all tags): 39 test cases, 167 assertions, all pass; the `[integration]` cases load a real runtime DLL.
* `GraphicsHost.exe hardware`, `--json hardware`, `runtime info`: work; `--smoke-test` exits 0 (offscreen and Windows platform).
* The RX 7900 XTX now initialises in Vulkan: `ggml_vulkan: 0 = AMD Radeon RX 7900 XTX (AMD proprietary driver) ... matrix cores: KHR_coopmat`.
  `backend = auto` therefore selects the Vulkan runtime. The Phase 0 GPU timings (SD1.5, SDXL) can now be measured.
* Found and fixed by this run: `hardware` showed 64 logical cores on the 192-thread CPU (`GetSystemInfo` counts one
  processor group only); it now uses `GetActiveProcessorCount(ALL_PROCESSOR_GROUPS)`, with a test.

If a future build is blocked again, the notes above still apply: do not work around the policy, ask the machine's owner.
