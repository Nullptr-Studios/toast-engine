# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Toast Engine is an open source game engine (Nullptr* Studios). It is a polyglot project:

- **`engine/`** — C++23 core engine, built as a shared library (`toast_engine`) via CMake + vcpkg.
- **`editor/`** — C# (.NET 10, Avalonia/Dock) editor application that loads `toast_engine` as a native library via P/Invoke (see `editor/Engine/ToastEngine.cs`).
- **`tools/`** — `reflection_generator` (Rust, parses engine headers and codegens reflection C++), `log_server` + `kenzo` (Rust, out-of-process logging backend/TUI), `player` (C#, standalone runtime host), `dummy_game` (C++ sample game DLL used for FFI testing).
- **`tests/`** — C++ unit tests, one `.cpp` file per test case, registered into a single `toast_tests` binary.

## Build system & commands

Prerequisites: CMake >=3.24, a C++23 compiler (MSVC/VS 2026 or GCC15 tested), Rust (rustc/cargo), .NET 10 SDK, and vcpkg with `VCPKG_ROOT` on PATH.

### Generate + build

```powershell
cmake -B build/Debug -G "Visual Studio 18 2026" -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build ./build/Debug --parallel $env:NUMBER_OF_PROCESSORS
```

```bash
# Linux
cmake -B build/Debug -G Ninja -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build ./build/Debug --parallel "$(nproc)"
```

Or via the Nix flake (`nix develop` exposes `cmake-gen`, `cmake-build`, `editor`, `kenzo` as scripts):

```bash
nix develop
cmake-gen
cmake-build
```

Regardless of which directory you configure CMake in (`build/Debug`, `cmake-build-debug`, etc.), all build products land under `<repo>/out/<Config>/` (e.g. `out/Debug/toast_engine/bin`, `out/Debug/tests`) — this is set by `OUTPUT_ROOT` in the top-level `CMakeLists.txt`.

Run the editor (loads `toast_engine` natively):

```bash
dotnet run --project editor
```

### Tests

```bash
ctest --test-dir build/Debug -C Debug
```

Tests require `TRACY_NO_INVARIANT_CHECK=1` (already set as a compile definition on the `toast_tests` target). Each test file under `tests/<group>/NN-name.cpp` becomes its own ctest case named `<group>/name`, so you can run a single test directly:

```bash
ctest --test-dir build/Debug -C Debug -R "events/03-consume"
# or invoke the test binary directly:
out/Debug/tests/toast_tests --test events/03-consume
out/Debug/tests/toast_tests --list   # list all registered test names
```

New test files are auto-discovered (`file(GLOB_RECURSE ...)` in `tests/CMakeLists.txt`) — no registration needed beyond using `TOAST_TEST_NAMED` (see `tests/test_registry.hpp`).

The Rust `reflection_generator` also has its own test suite, wired into ctest as `refgen_unit_tests` (runs `cargo test`).

### Formatting / linting (C++)

```bash
cmake --build build/Debug --target clang_format   # formats engine/ in place, style from .clang-format
tools/ci/clang_tidy.sh <build-dir>                 # requires compile_commands.json in <build-dir>
```

Both scripts exclude `engine/generated`, `engine/external`, and `engine/ffi`.

## Architecture

### Reflection (`engine/src/toast/reflect/`, `tools/reflection_generator/`)

Toast avoids RTTI/vtables for `Node` dispatch by generating static reflection data instead. The Rust `reflection_generator` parses every header under `engine/src/toast` containing a `[[ToastNode]]` attribute (using Tree-sitter) and emits `engine/generated/*.generated.hpp` plus `reflect.generated.cpp`, filling out `Reflect<T>` specializations and a `NodeInfo` per type (fields, methods, factory/destructor, tick-function bitmask, inheritance chain). This runs as a pre-build step (`reflection_gen` CMake target) — **never hand-edit files in `engine/generated/`**.

Key rule from the design: no macros, and the class header itself is never modified by codegen — everything is attribute-driven (`[[Reflect]]`, `[[Name("...")]]`, `[[ReadOnly]]`, `[[Group("...")]]`, `[[Range(a,b)]]`, `[[Hidden]]`, etc. — full list in `docs/reflection.md`). Field/method access at runtime goes through `NodeInfo::search()/getField()/getMethod()/call<R>()`, not direct member access, when working generically.

### World & scene graph (`engine/src/toast/world/`)

`World` is the sole owner of all `Node` instances; nodes form a tree (one parent, N children). A node's `NodeState` (null/loading/cached/root/global/destroy) and `NodeType` (null/child/root/world_root) track its place in the graph — see `docs/world.md` for the full state table. Scene switching goes through `World::setRoot()` (atomic swap, no empty-world frame), `World::cacheNode()`, `World::loadNode()` (async), `World::destroyNode()` (deferred to next tick).

Tick scheduling: every frame runs four ordered phases (`earlyTick` → `tick` → `postPhysics` → `lateTick`). Within a phase, nodes are grouped into **waves** that run in parallel on the thread pool; wave index is `max(predecessor wave) + 1`, computed via BFS subgraph separation + Tarjan SCC (cyclic dependencies collapse into a `NodeCluster` ticked sequentially). Dependencies are declared explicitly with `World::instance()->registerDependency(a, b)` and trigger a schedule rebuild.

`Workspace` (`engine/src/toast/world/workspace.cpp`) is the editor-viewport counterpart of `World`: owns a node tree the same way but never ticks it and never builds a dependency graph.

### Events (`engine/src/toast/events/`)

Event types derive from `event::Event<T>`; `event::send<T>(...)` is thread-safe and enqueues into a double-buffered circular pool swapped on `event::pollEvents()` (main-thread-only, not thread-safe). Callbacks are managed per-listener via `event::Listener` (auto-unsubscribes on destruction, supports named/unnamed + priority) and stored in a `std::multimap<char priority, callback>` per event type. See `docs/event_system.md` for the internal vtable-of-lambdas trick used to erase callback removal across event types.

### Prefabs (`engine/src/toast/assets/prefab.cpp`)

Serialized node trees, text (`.node`, editor-authored) or binary (`.tnode`, runtime). Instantiated via `Node::spawn(uid_or_uri)`. Prefab instances are opaque to `find()`/`search()` traversal (`m_prefab_interior`), support field overrides (outer file overrides layered onto inner prefab defaults), and guard against self-referential recursion via `m_self_uid` + `InstantiateContext::asset_chain`. Bump `_detail::format_version` in `prefab.hpp` on any binary layout change. Full details in `docs/prefabs.md`.

### Renderer (`engine/src/toast/renderer/`)

Threaded Vulkan backend split across two domains: the **main thread** builds a `VulkanRenderer::RenderFrame` snapshot (frame data + `MeshInstanceProxy` list — mesh handle + model matrix, copied out of the scene graph) and calls `submitFrame()`; the **render thread** consumes the snapshot, records passes (`IRenderPass` implementations, e.g. `MeshPass`), submits, and presents. This keeps simulation code from ever blocking on GPU submission and lets the same pass pipeline drive both on-screen (`SDLOutputTarget` + swapchain) and off-screen (`SharedTextureOutputTarget`, used by the editor viewport) targets. Resource uploads (mesh/texture) are staged and queued via `VulkanRenderer::queueResourceUpload(...)`, flushed asynchronously with fences. See `docs/renderer.md`.

### Logging (`engine/src/toast/logger.cpp`, `tools/log_server`, `tools/kenzo`)

Logging is out-of-process by design (crash-safe, keeps parsing off the game thread): `TOAST_TRACE/INFO/WARN/ERROR/CRITICAL(sink, ...)` macros (`toast/log.hpp`) protobuf-encode (`protos/logging.proto`) and batch-send log records over TCP to `log_server` (Rust), which persists to CSV and fans out to connected TUI/editor clients. `kenzo` is the Rust TUI client (vim-style navigation) for reading logs live or from a CSV. `TOAST_ASSERT(cond, sink, ...)` is the assert macro (compiled out in Release, same as `std::assert`).

### Editor interop (`editor/`)

The C# editor talks to `toast_engine` purely through the C FFI surface in `engine/ffi/*.h` (`engine.h`, `events.h`, `audio.h`, `log.h`, `gltf_importer.h`) via P/Invoke — see `editor/Engine/ToastEngine.cs`. `engine/CMakeLists.txt`'s post-build step copies `include/`, `ffi/`, and required third-party headers into `out/<Config>/toast_engine/`, and the editor loads the built DLL from `<editor-output>/../toast_engine/bin`. Window/workspace events cross the boundary as protobuf messages (`protos/window_events.proto`, `protos/workspace_events.proto`), also consumed on the Rust/log side.

## Code style (C++)

Enforced by `.clang-format` (tabs, width 2, 130 column limit, LF endings) — always run the `clang_format` target rather than hand-formatting. Public headers are asset-generated into the package layout by `sync_api_headers`; when adding a new public header under `engine/src/**.hpp`, no manual step is needed beyond the normal build (it's synced automatically).