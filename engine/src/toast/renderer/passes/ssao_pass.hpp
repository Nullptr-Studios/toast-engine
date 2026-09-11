/**
 * @file ssao_pass.hpp
 * @author dario
 * @date 07/08/2026
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

/**
 * @brief Screen-space ambient occlusion, weighted onto the indirect diffuse term only
 *
 * Derived from the depth and normal buffers the world stage already writes
 *
 * What it multiplies is the part worth knowing. Occlusion stands in for nearby geometry blocking the
 * environment, so it belongs to indirect light alone - darkening the whole summed colour makes the scene
 * murky instead of adding contact shadows. mesh.slang therefore writes indirect diffuse to its own
 * attachment, and this pass does `color - indirect + indirect * ao`
 *
 * @note Registered before SSR and bloom, so a darkened crevice does not feed bloom its unoccluded brightness
 */
class SsaoPass : public IPostProcessPass {
public:
	SsaoPass(const VulkanCore& core, vk::Format scene_format, vk::Extent2D extent);

	[[nodiscard]]
	auto name() const -> std::string_view override {
		return "SSAO";
	}

	auto record(vk::CommandBuffer cmd, uint32_t frame_index, vk::ImageView source_view) -> vk::ImageView override;

	void onResize(vk::Extent2D extent) override;

private:
	/// @brief Mirrors ssao.slang's SsaoParams push constant block
	struct Params {
		glm::mat4 view_projection {1.0f};
		glm::mat4 inverse_view_projection {1.0f};
		glm::vec4 screen_size {0.0f};
		/// x = world-space sample radius, y = strength, z = depth-range cutoff, w = sample count
		glm::vec4 tuning {0.0f};
		/// x = render mode, yzw = spare
		glm::vec4 misc {0.0f};
	};

	void createResources(const VulkanCore& core);
	void createTarget(const VulkanCore& core, vk::Extent2D extent);

	const VulkanCore* m_core = nullptr;
	vk::Format m_scene_format = vk::Format::eUndefined;

	ShaderLayout m_shader_layout;
	VulkanPipeline m_pipeline;

	vk::raii::Sampler m_sampler = nullptr;
	/// Depth and the geometry buffer are point-sampled - see SsrPass for why filtering them invents surfaces
	vk::raii::Sampler m_point_sampler = nullptr;

	std::vector<vk::raii::DescriptorSet> m_descriptor_sets;
	std::vector<vk::ImageView> m_bound_views;

	PostProcessTarget m_target;
};

}
