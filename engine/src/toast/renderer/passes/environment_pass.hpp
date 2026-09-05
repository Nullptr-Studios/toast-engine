/**
 * @file environment_pass.hpp
 * @author dario
 * @date 03/08/2026
 */

#pragma once

#include "../cubemap_target.hpp"
#include "../render_pass_base.hpp"
#include "../shader_layout.hpp"
#include "../vulkan_common.hpp"
#include "../vulkan_pipeline.hpp"

#include <glm/glm.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <toast/assets/hdr_image.hpp>
#include <vector>

namespace renderer {
class VulkanCore;

/**
 * @brief Builds the environment cubemaps that light the scene indirectly
 *
 * Sky (256px), its cosine convolution for diffuse (32px) and its GGX convolution per roughness for specular
 * (128px chain). Split-sum: the integral depends only on the environment, so it is precomputed once
 *
 * Nothing here runs per frame except the skybox draw
 */
class EnvironmentPass : public IRenderPass {
public:
	EnvironmentPass(const VulkanCore& core, vk::Format hdr_format, vk::Format depth_format);

	[[nodiscard]]
	auto name() const -> std::string_view override {
		return "Environment";
	}

	/// @brief The precompute, recorded once on the first frame - it needs a command buffer, and construction
	/// has none
	void recordPre(vk::CommandBuffer cmd, uint32_t frame_index, uint32_t image_index) override;

	/// @brief Draws the sky behind the scene, in the world stage
	void record(vk::CommandBuffer cmd, uint32_t frame_index, uint32_t image_index) override;

	[[nodiscard]]
	auto stage() const -> RenderStage override {
		return RenderStage::world;
	}

	/// @returns cosine-convolved irradiance, sampled as gIrradianceMap; null until the precompute has run
	[[nodiscard]]
	auto getIrradianceView() const -> vk::ImageView;

	/// @returns roughness-prefiltered radiance, sampled as gPrefilteredEnv
	[[nodiscard]]
	auto getPrefilteredView() const -> vk::ImageView;

	/// @returns the sampler both are read through - linear, with mip filtering for the prefiltered chain
	[[nodiscard]]
	auto getSampler() const -> vk::Sampler {
		return *m_sampler;
	}

	/// @returns how many roughness levels the prefiltered chain holds, which the shader maps roughness onto
	[[nodiscard]]
	auto getPrefilteredMipCount() const noexcept -> uint32_t {
		return m_prefiltered_mips;
	}

	/// @returns true once the cubemaps hold real data and may be sampled
	///
	/// Not "is the precompute up to date" - a rebuild overwrites in place, and reporting not-ready mid-way
	/// would drop every surface to uniform ambient for a frame
	[[nodiscard]]
	auto isReady() const noexcept -> bool {
		return m_has_data;
	}

	/// @returns brightness the generated sky is folded with, see m_intensity
	[[nodiscard]]
	auto getSkyIntensity() const noexcept -> float {
		return m_intensity;
	}

	/// @brief Sets the sky's brightness and re-runs the precompute next frame
	///
	/// All three rebuild together so they cannot drift apart. ~16k samples per texel for the irradiance
	/// convolution alone, so drive it from a committed edit, not a slider being dragged
	void setSkyIntensity(float intensity) noexcept {
		if (intensity == m_intensity) {
			return;
		}
		m_intensity = intensity;
		m_precompute_pending = true;
	}

	/**
	 * @brief Replaces the generated sky with an imported equirectangular HDR
	 *
	 * Resolved into the same cubemap the procedural sky fills, so both convolutions, the roughness chain, the
	 * skybox and every probe capture follow without knowing the difference
	 *
	 * @param uri Asset URI of a Radiance `.hdr`; pass an empty string to return to the generated sky
	 * @returns false if the file could not be read or decoded, leaving the current sky untouched
	 */
	auto setEnvironmentMap(std::string_view uri) -> bool;

	/// @returns URI of the imported environment, or empty when the sky is generated
	[[nodiscard]]
	auto getEnvironmentMapUri() const -> const std::string& {
		return m_environment_uri;
	}

private:
	/// @brief Mirrors environment.slang's EnvironmentParams
	struct Params {
		glm::vec4 face_right {0.0f};
		glm::vec4 face_up {0.0f};
		glm::vec4 face_forward {0.0f};
		float roughness = 0.0f;
		float intensity = 1.0f;
		glm::vec2 _pad0 {0.0f};
	};

	void createPipelines(const VulkanCore& core);
	void createDescriptors(const VulkanCore& core);

	/// @brief Renders one face of one mip with @p pipeline, @p params supplying that face's basis
	void renderFace(
	    vk::CommandBuffer cmd, const VulkanPipeline& pipeline, vk::DescriptorSet set, const CubemapTarget& target, uint32_t mip,
	    uint32_t face, const Params& params
	);

	const VulkanCore* m_core = nullptr;
	vk::Format m_format = vk::Format::eUndefined;

	ShaderLayout m_shader_layout;
	VulkanPipeline m_sky_pipeline;
	/// Projects an imported equirectangular image onto the cube faces, in place of m_sky_pipeline
	VulkanPipeline m_equirect_pipeline;
	VulkanPipeline m_irradiance_pipeline;
	VulkanPipeline m_prefilter_pipeline;
	/// Draws into the scene target rather than a cubemap face, so it needs the depth format the others don't
	VulkanPipeline m_skybox_pipeline;
	vk::Format m_depth_format = vk::Format::eUndefined;

	vk::raii::Sampler m_sampler = nullptr;

	CubemapTarget m_sky;
	CubemapTarget m_irradiance;
	CubemapTarget m_prefiltered;

	/// Reads the sky, for the two convolution steps
	vk::raii::DescriptorSet m_sky_source_set = nullptr;

	/// @brief The imported equirectangular image, when there is one
	///
	/// A 1x1 placeholder stands in otherwise. The set is written once at startup, and an unwritten binding is
	/// undefined behaviour on any draw touching the layout - including ones that never sample it
	std::optional<vma::raii::Image> m_equirect_image;
	std::optional<vk::raii::ImageView> m_equirect_view;
	vk::raii::Sampler m_equirect_sampler = nullptr;

	/// @brief Reads the equirectangular image while the sky cubemap is the render target
	///
	/// Separate from m_sky_source_set, whose binding 0 is the sky cube this step writes - a descriptor may not
	/// point at an image being written in the same draw. Binding 0 here is the 1x1 black cube instead
	vk::raii::DescriptorSet m_equirect_set = nullptr;
	std::string m_environment_uri;
	/// False while m_equirect_image holds only the placeholder
	bool m_has_equirect = false;

	void createEquirectPlaceholder(const VulkanCore& core);
	/// @brief Points m_equirect_set at whatever m_equirect_view currently holds
	void writeEquirectSet(const VulkanCore& core);
	/// @brief Replaces m_equirect_image with @p image and points the descriptor at it
	auto uploadEquirect(const assets::HdrImage& image) -> bool;

	uint32_t m_prefiltered_mips = 1;
	/// Set whenever the cubemaps need (re)generating; cleared once recordPre() has done it
	bool m_precompute_pending = true;
	/// Latches on the first successful precompute and never clears - see isReady()
	bool m_has_data = false;

	/// Folded into the maps rather than applied at sample time, so the chain stays consistent with the skybox
	///
	/// environment.slang's gradient is authored in 0.1..0.7 - fine as a colour, far too dark to light a scene.
	/// This is the one number standing in for an imported HDR's absolute values
	float m_intensity = 3.0f;
};

}
