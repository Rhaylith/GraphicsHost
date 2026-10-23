# Adding an option

Every new knob follows the same path. Work through the table top to bottom; each row names the place to touch. Rows
marked *(later)* refer to code that arrives in a later phase, so for now only the rows with a file link apply.

Three kinds of option:

* **Setting**: application-wide, stored in `settings.json` (`core::Settings`): backend, port, log level, ...
* **Load option**: how a model is put in memory (`LoadOptions`, phase 2): device, threads, offload, VAE tiling, ...
  Changing it needs a reload. Its C-ABI twin is `gh_load_params` in [runtime/gh_runtime.h](../runtime/gh_runtime.h).
* **Generation option**: per request (`GenerationRequest`, phase 2): prompt, steps, sampler, seed, ... Changing it never
  reloads. Its C-ABI twin is `gh_generate_params`.

## Checklist

| # | Step | Setting | Load | Generation | Where |
|---|------|:---:|:---:|:---:|-------|
| 1 | **Declare** the field with a comment and a default that keeps old behaviour | ✔ | ✔ | ✔ | Settings: [Settings.h](../GraphicsHost.Core/core/Settings.h). Load/generation: `engine/` structs *(later)*. |
| 2 | **Serialise**: add it to `toJson` / `settingsFromJson` (one place, both directions) | ✔ | ✔ | ✔ | [Settings.cpp](../GraphicsHost.Core/core/Settings.cpp). Add the key to `kKnownKeys` for a new top-level setting, otherwise it is preserved as an unknown key. |
| 3 | **Validate** ranges and allowed values, with a field path as name | ✔ | ✔ | ✔ | `validate()` in [Settings.cpp](../GraphicsHost.Core/core/Settings.cpp). Never coerce silently. |
| 4 | **Use it** | ✔ | ✔ | ✔ | Setting: the code that reads `AppHost::settings()`. Load/generation: the engine wrapper over the runtime ABI *(later)*. |
| 5 | **Cross the ABI** (only if the runtime needs it) | | ✔ | ✔ | Append the field at the **end** of the struct in `gh_runtime.h`, handle `struct_size` for old callers, map it in [gh_runtime.cpp](../runtime/gh_runtime.cpp). Bump `GH_ABI_VERSION` only for incompatible changes. |
| 6 | **Defaults, presets, per-model defaults** | | ✔ | ✔ | Automatic once the struct serialises *(later)*. If the meaning of stored data changes, add a migration step (below). |
| 7 | **API fields** | | ✔ | ✔ | OpenAI mapping, native `/api/v1`, browser UI request code *(later, phase 4)*. |
| 8 | **CLI** | ✔ | ✔ | ✔ | `config set <key> <value>` works for every setting through the JSON (see [Cli.cpp](../GraphicsHost.Core/cli/Cli.cpp)): nothing to do. Dedicated flags for `gen`/`load` are manual *(later)*. |
| 9 | **UI**: control in the Qt settings or parameter panel, and in the browser UI | ✔ | ✔ | ✔ | `GraphicsHost/ui/` and `web/` *(later)*. Give the control an interface mode (below). |
| 10 | **Tests** | ✔ | ✔ | ✔ | [SettingsTests.cpp](../GraphicsHost.Tests/SettingsTests.cpp): default, round trip, each validation rule, and the CLI case in [CliTests.cpp](../GraphicsHost.Tests/CliTests.cpp). A model-backed `[integration]` test only if behaviour needs a real model. |
| 11 | **Docs** | ✔ | ✔ | ✔ | README feature list, `docs/` pages, and tick the item in [implementation-plan.md](../implementation-plan.md). |

## Changing how data is stored

`settings.json` (and later presets, per-model defaults, the gallery index) carries a `schemaVersion`. Adding a field with
a sensible default needs **no** migration. Renaming, removing or re-interpreting data does:

1. Raise `kSettingsSchemaVersion` in [Settings.h](../GraphicsHost.Core/core/Settings.h).
2. Append one step to `migrations()` in [Migrations.cpp](../GraphicsHost.Core/core/Migrations.cpp). Step *i* upgrades
   version *i* to *i + 1* and edits the JSON document in place.
3. Add a test in [SettingsTests.cpp](../GraphicsHost.Tests/SettingsTests.cpp) with a document in the old shape.

The original file is kept as `settings.json.bak-v<old>` before it is rewritten. A file from a **newer** version is
refused (it is moved to `settings.json.invalid` and defaults are used), never downgraded.

## Conventions

* Names: C++ `camelCase` members, `camelCase` JSON keys, nested objects for groups (`server.port`).
* Booleans are positive: `flashAttention`, not `noFlashAttention`.
* "Unset" is `0` or an empty string and means *use the engine default*; say so in the field comment.
* Strings with a fixed set of values (`auto | vulkan | cpu`) are validated, never silently coerced.
* A new option must not change behaviour for existing users unless they set it.
* Off-by-default features stay off by default (the safety filter is the model: `safety.enabled = false`).

## Interface modes *(later, phase 6)*

Every control that is not for everyone gets an interface mode: `user`, `power` or `developer`. Controls above the chosen
mode are hidden, never disabled. Advanced samplers, offload switches and the request log are `power` or `developer`.
