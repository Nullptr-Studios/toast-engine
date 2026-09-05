/// @file shadow_constants.hpp
/// @author dario
/// @date 01/08/2026
///
/// @brief Shared sizing for the shadow system - mirrored by shadow_depth.slang and mesh.slang

#pragma once

#include <cstdint>
#include <toast/export.hpp>

namespace renderer::shadows {

/// Each extra cascade is another full scene redraw
inline constexpr uint32_t k_cascade_count = 4;

/// Per-cascade square resolution; all cascades share one image with k_cascade_count array layers
inline constexpr uint32_t k_cascade_resolution = 1024;

inline constexpr float k_shadow_distance = 150.0f;

/// Split distribution, uniform (0) to logarithmic (1). Pure logarithmic puts the first split absurdly close
inline constexpr float k_cascade_split_lambda = 0.75f;

inline constexpr uint32_t k_max_spot_shadows = 4;

/// Each burns six layers, one per cube face
inline constexpr uint32_t k_max_point_shadows = 2;

inline constexpr uint32_t k_cube_faces = 6;

inline constexpr uint32_t k_punctual_resolution = 512;

/// A shrinking light renders into a sub-rect of its full-size layer rather than a smaller image, so nothing
/// reallocates as lights move. The remainder stays cleared to "unshadowed" - see GpuLight::shadow_atlas
inline constexpr uint32_t k_min_punctual_resolution = 64;

inline constexpr float k_punctual_full_resolution_distance = 10.0f;
inline constexpr float k_punctual_min_resolution_distance = 60.0f;

/**
 * @brief The three knobs the settings system may move; they default to the k_* values above
 *
 * Only these three. The counts size fixed arrays in shadow_depth.slang and the FrameUBO, so moving one is a
 * shader change. Resolutions are read when ShadowPass allocates, so they take effect on the next run;
 * distance is read per frame and applies at once
 *
 * @note Set from the main thread before the passes are built. Atomic because the render thread reads the
 * resolutions while startup settings may still be applying
 */
TOAST_API auto cascadeResolution() -> uint32_t;
TOAST_API auto punctualResolution() -> uint32_t;
TOAST_API auto shadowDistance() -> float;

TOAST_API void setCascadeResolution(uint32_t resolution);
TOAST_API void setPunctualResolution(uint32_t resolution);
TOAST_API void setShadowDistance(float distance);

/**
 * @brief Shadow resolution for a punctual light @p camera_distance metres from the camera
 * @param resolution_scale Light::shadowResolutionScale(), an artist multiplier on top of the distance curve
 */
[[nodiscard]]
inline auto punctualShadowResolution(float camera_distance, float resolution_scale) -> uint32_t {
	const uint32_t full_resolution = punctualResolution();
	const float span = k_punctual_min_resolution_distance - k_punctual_full_resolution_distance;
	float t = span > 0.0f ? (camera_distance - k_punctual_full_resolution_distance) / span : 1.0f;
	t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);

	const auto max_resolution = static_cast<float>(full_resolution);
	const auto min_resolution = static_cast<float>(k_min_punctual_resolution);
	const float scale = resolution_scale < 0.0f ? 0.0f : (resolution_scale > 1.0f ? 1.0f : resolution_scale);
	const float target = (max_resolution + ((min_resolution - max_resolution) * t)) * scale;

	// Power of two: a fractional sub-rect puts the light's texel grid out of step with the atlas grid, which
	// shimmers as the camera moves
	uint32_t resolution = k_min_punctual_resolution;
	while ((resolution * 2) <= full_resolution && static_cast<float>(resolution * 2) <= target) {
		resolution *= 2;
	}
	return resolution;
}

/// Punctual image layout: spots first, then each point light's six faces
inline constexpr uint32_t k_spot_layer_base = 0;
inline constexpr uint32_t k_point_layer_base = k_max_spot_shadows;
inline constexpr uint32_t k_punctual_layer_count = k_max_spot_shadows + (k_max_point_shadows * k_cube_faces);

/// Also the size of shadow_depth.slang's matrix array
inline constexpr uint32_t k_max_shadow_views = k_cascade_count + k_punctual_layer_count;

inline constexpr float k_punctual_near = 0.05f;

/// Extra texels rendered beyond each cube face's 90-degree frustum. Six independent array layers have no
/// seamless filtering, so a tap near a face edge reads the cleared border and draws a bright seam. Must
/// exceed mesh.slang's filter radius
inline constexpr float k_cube_face_guard_texels = 4.0f;

}
