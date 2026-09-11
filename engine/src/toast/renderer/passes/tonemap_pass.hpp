/**
 * @file tonemap_pass.hpp
 * @author dario
 * @date 03/08/2026
 */

#pragma once

#include "../post_process_pass_base.hpp"
#include "../post_process_target.hpp"
#include "../shader_layout.hpp"
#include "../vulkan_common.hpp"
#include "../vulkan_pipeline.hpp"

#include <glm/glm.hpp>
#include <optional>
#include <vector>

namespace renderer {
class VulkanCore;

/// @brief Tone curve applied when mapping the HDR scene to the display
enum class TonemapMode : uint32_t {
	/// x / (1 + x). What the material shaders applied inline before this pass existed
	reinhard = 0,
	/// Narkowicz's ACES fit - holds highlight saturation far better, slightly contrastier midtones
	aces = 1,
};

/**
 * @brief Maps the linear HDR scene into the display-space image, as the last step of the post chain
 *
 * The one place that decides how unbounded radiance becomes pixels. It used to be the last four lines of
 * every material shader, which meant each carried its own copy of the policy, transparent surfaces blended
 * against already-tonemapped colours, and there was nowhere for bloom or exposure to live. Draws a
 * full-screen triangle with no vertex buffer - see tonemap.slang
 */
class TonemapPass : public IPostProcessPass {
public:
	/// @param ldr_format Format of the target this writes - the display-space format the rest of the chain
	///        and the present step work in
	TonemapPass(const VulkanCore& core, vk::Format ldr_format, vk::Extent2D extent);

	[[nodiscard]]
	auto name() const -> std::string_view override {
		return "Tonemap";
	}

	auto record(vk::CommandBuffer cmd, uint32_t frame_index, vk::ImageView source_view) -> vk::ImageView override;

	void onResize(vk::Extent2D extent) override;

private:
	/// @brief Mirrors tonemap.slang's TonemapParams
	///
	/// std140 rounds a uniform block up to a multiple of 16, so the shader's block is 48 bytes even though
	/// its fields end at 36. The trailing vec4 keeps this struct the same size - a shorter one would leave
	/// the descriptor range smaller than the shader reads
	struct Params {
		float exposure = 1.0f;
		uint32_t mode = static_cast<uint32_t>(TonemapMode::reinhard);
		float gamma = 2.2f;
		float contrast = 1.0f;
		float saturation = 1.0f;
		float vignette = 0.0f;
		float grain = 0.0f;
		float time = 0.0f;
		glm::vec4 _pad0 {0.0f};
	};

	void createResources(const VulkanCore& core);
	void createTarget(const VulkanCore& core, vk::Extent2D extent);

	const VulkanCore* m_core = nullptr;
	vk::Format m_ldr_format = vk::Format::eUndefined;

	/// Where the tonemapped image lands. It is not the swapchain image because passes after this one - FXAA,
	/// and anything else that has to reason about displayed contrast - need to read it back
	PostProcessTarget m_target;

	ShaderLayout m_shader_layout;
	VulkanPipeline m_pipeline;

	vk::raii::Sampler m_sampler = nullptr;
	std::vector<vk::raii::DescriptorSet> m_descriptor_sets;
	std::vector<FrameResources> m_params_buffers;

	/// Last view written into each frame's descriptor set, so an unchanged source doesn't rewrite it
	std::vector<vk::ImageView> m_bound_views;
};

}
