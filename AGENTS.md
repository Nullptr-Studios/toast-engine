# Toast Engine Development Guide

## Build System
- Uses CMake with Ninja generator
- Builds for Windows (MSVC) and Linux
- Requires vcpkg with `VCPKG_ROOT` set in PATH

## Key Commands
### Setup
```bash
# Using Nix (recommended)
nix develop
cmake-gen
```

```powershell
# Using PowerShell
cmake -B build/Debug -G "Visual Studio 18 2026" -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
```

### Build
```bash
cmake --build ./build/Debug
dotnet run --project editor
```

### Test
```bash
ctest --test-dir build/Debug -C Debug
```

## Project Structure
- `engine/` - Core engine library
- `editor/` - C# editor application
- `tools/` - Various tools including reflection_generator
- `tests/` - Unit tests

## Engine Architecture
### World and Nodes
- The `World` is the singleton that owns, ticks, and manages every `Node` in the engine
- Nodes form a tree structure with one parent and any number of children  
- Node states: null, loading, cached, root, global, destroy
- Node types: null, child, root, world_root
- Scene management with `setRoot()`, `cacheNode()`, `destroyNode()`

### Reflection System
- Uses code generation with Rust-based `reflection_generator` tool
- Attributes like `[[ToastNode]]`, `[[Reflect]]`, `[[Name("str")]]`, etc.
- Supports field reflection, function reflection, and RTTI replacement
- Generates static `Reflect<T>` structs with get/set accessors and metadata

### Event System
- Thread-safe event sending via `event::send<T>(...)`
- Callbacks with priority support
- Event dispatching through `event::pollEvents()`
- Listener class manages subscriptions with automatic unsubscription on destruction

### Prefabs
- Serialized node trees stored on disk in text (`.node`) or binary (`.tnode`) format
- Support for field overrides and self-referencing prefabs
- Loading via `World::loadNode()` and instantiation through `spawn()`

## Renderer Architecture
- Vulkan-based rendering backend with multi-threaded design
- Separates main thread (frame building) from render thread (GPU work)
- Uses double buffering with 3 frames in flight
- Supports both on-screen SDL swapchain and off-screen editor rendering via IOutputTarget
- Asynchronous resource uploads using staging buffers and fence-based completion tracking
- Frame-based command buffer recording with proper synchronization via semaphores and fences
- Multi-pass rendering through IRenderPass interface
- Resource management using Vulkan Memory Allocator (VMA) for efficient GPU memory handling

## Special Notes
- The editor is built with .NET 10
- The build system uses vcpkg for dependencies
- Tests require TRACY_NO_INVARIANT_CHECK=1 environment variable
- Nix flake provides scripts: cmake-gen, cmake-build, editor, kenzo