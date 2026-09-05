/**
 * @file post_process_settings.hpp
 * @author dario
 * @date 19/08/2026
 */

#pragma once
#include <cstdint>
#include <toast/export.hpp>

namespace renderer {

/**
 * @brief The look of the post chain - what an artist grades a level to
 *
 * Scene content, not a machine setting: authored on a PostProcessVolume and blended each frame from the
 * volumes containing the camera. Travels in the frame snapshot, because the passes run on the render thread.
 * Whether a layer runs at all is IPostProcessPass::setEnabled()
 *
 * @note Every member must be handled in blendPostProcess(). One that is not silently keeps the
 * lowest-priority volume's value
 */
struct PostProcessSettings {
	struct Bloom {
		/// Luminance where bloom starts
		float threshold = 1.0f;
		/// Soft shoulder below the threshold; 0 makes highlights pop on and off as they cross it
		float knee = 0.5f;
		/// Upsample filter width in source texels - how far each step spreads the glow
		float filter_radius = 1.0f;
		/// How much bloom is mixed back over the scene
		float strength = 0.05f;
	} bloom;

	struct Tonemap {
		/// Linear multiplier applied before the curve
		float exposure = 1.0f;
		/// 0 = Reinhard, 1 = ACES
		uint32_t mode = 0;
		/// Encode gamma; 2.2 for a UNORM swapchain
		float gamma = 2.2f;
		/// Contrast around middle grey; 1 is neutral
		float contrast = 1.0f;
		/// 0 fully desaturated, 1 neutral, above 1 oversaturated
		float saturation = 1.0f;
		/// Corner darkening, 0 disables it
		float vignette = 0.0f;
		/// Film grain amount, 0 disables it
		float grain = 0.0f;
	} tonemap;

	struct Fxaa {
		/// Absolute local contrast below which a pixel is left alone
		float contrast_threshold = 0.0312f;
		/// Contrast relative to the brightest neighbour, so dark areas are not over-filtered
		float relative_threshold = 0.125f;
		/// Strength of the final blend, 0..1
		float subpixel_blending = 0.75f;
	} fxaa;

	struct Ssr {
		/// How much of the traced reflection is mixed in, on top of the Fresnel and roughness weighting
		float intensity = 1.0f;

		/// Roughness past which a surface stops tracing and keeps the probe answer. One mirror ray only models
		/// a smooth surface
		float max_roughness = 0.6f;

		/// How far behind a surface a ray may pass and still count as a hit, in world units. The depth buffer
		/// has no thickness, so without a limit a ray far behind a thin railing still hits it
		float thickness = 0.5f;

		/// World-space distance between march samples. Larger costs less and misses thin geometry
		float stride = 0.15f;

		/// Samples per ray before giving up and falling through to the probe tier
		uint32_t max_steps = 48;
	} ssr;

	struct Ssao {
		/// Hemisphere radius in world units. Scene-scale dependent - the first thing to move if the effect is
		/// invisible or covers everything
		float radius = 0.5f;

		/// How strongly the occlusion darkens the indirect diffuse; 0 disables it
		float strength = 1.0f;

		/// Distance past which an occluder counts as a different surface. Without it a wall across the room
		/// occludes the railing in front of it, and silhouettes grow dark halos
		float range_cutoff = 1.0f;

		/// Samples per pixel. Rotated per pixel, so raising this trades cost for less noise
		uint32_t sample_count = 16;
	} ssao;
};

/**
 * @brief Mixes @p target @p weight of the way towards @p source
 *
 * Linear per field, so a volume's blend distance reads as a fade. The tonemap curve switches at halfway
 * instead - there is nothing in between Reinhard and ACES
 */
TOAST_API void blendPostProcess(PostProcessSettings& target, const PostProcessSettings& source, float weight);

}
