#include "vulkan_core.hpp"
#include "vulkan_renderer.hpp"

#include <ffi/renderer.h>

namespace {

/// @returns the renderer, or nullptr before a window exists
///
/// The editor's settings window opens with the shell, so it can ask for counts before there is a renderer
auto rendererOrNull() -> renderer::VulkanRenderer* {
	return renderer::VulkanRenderer::instance;
}

}

extern "C" {

void toast_renderer_bake_reflection_probes() noexcept {
	if (auto* r = rendererOrNull()) {
		r->requestReflectionProbeBake();
	}
}

void toast_renderer_bake_irradiance_volumes() noexcept {
	if (auto* r = rendererOrNull()) {
		r->requestIrradianceBake();
	}
}

auto toast_renderer_stale_reflection_probes() noexcept -> uint32_t {
	auto* r = rendererOrNull();
	return r != nullptr ? r->getStaleReflectionProbeCount() : 0u;
}

auto toast_renderer_stale_irradiance_volumes() noexcept -> uint32_t {
	auto* r = rendererOrNull();
	return r != nullptr ? r->getStaleIrradianceVolumeCount() : 0u;
}

auto toast_renderer_irradiance_probe_count() noexcept -> uint32_t {
	auto* r = rendererOrNull();
	return r != nullptr ? r->getIrradianceProbeCount() : 0u;
}

auto toast_renderer_supports_ray_tracing() noexcept -> int32_t {
	auto* r = rendererOrNull();
	return r != nullptr && r->getCore().isRayTracingSupported() ? 1 : 0;
}
}
