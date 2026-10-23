# Development machine notes

Things found while building on the main development machine (Windows 11, AMD Ryzen Threadripper PRO 7995WX,
Radeon RX 7900 XTX). They are environment facts, not project bugs.

## Application Control may block freshly built binaries

Windows Code Integrity (an Application Control policy, ID `{0283ac0f-fff1-49ae-ada1-8a933130cad6}`) can refuse to load
unsigned binaries the build just produced. On 2026-10-22 the first build of `sd-cpu-avx2.dll` was refused
(`LoadLibrary` error 4551, Code Integrity events 3033/3077 in `Microsoft-Windows-CodeIntegrity/Operational`) while
`sd-cpu-noavx.dll` and `sd-vulkan.dll` from the same build loaded. Rebuilding the DLL produced a file that loaded.

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
