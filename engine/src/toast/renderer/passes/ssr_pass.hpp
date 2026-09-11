/**
 * @file ssr_pass.hpp
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
 * @brief Screen-space reflections, composited into the HDR scene before the tonemap
 *
 * The top tier of the reflection chain. A probe only reflects geometry sitting on its proxy box, so anything
 * standing inside the room slides under camera motion; this pass has the depth buffer and reflects what is
 * really there
 *
 * Only what is on screen, though. Rays that leave the viewport or hit a back face fall through to the probe
 * and environment tiers, which are already in the scene colour - so this pass adds rather than replaces. The
 * screen-edge fade is the technique's whole quality problem
 *
 * @note Registered before TonemapPass. A reflection carries radiance; compositing it after the tonemap would
 *       blow out every highlight it touched
 */
class SsrPass : public IPostProcessPass {
public:
	SsrPass(const VulkanCore& core, vk::Format scene_format, vk::Extent2D extent);

	[[nodiscard]]
	auto name() const -> std::string_view override {
		return "SSR";
	}

	auto record(vk::CommandBuffer cmd, uint32_t frame_index, vk::ImageView source_view) -> vk::ImageView override;

	void onResize(vk::Extent2D extent) override;

private:
	/// @brief Mirrors ssr.slang's SsrParams push constant block
	///
	/// 176 bytes, which is more than the 128 Vulkan guarantees. Both matrices are load-bearing - one rebuilds
	/// a world position from depth, the other projects the marched ray back to screen - and there is no third
	/// form that does both. The constructor checks maxPushConstantsSize and disables the pass rather than
	/// letting pipeline creation fail on a device that cannot take it; a UBO is the fix if one ever turns up
	struct Params {
		glm::mat4 view_projection {1.0f};
		glm::mat4 inverse_view_projection {1.0f};
		glm::vec4 camera_position {0.0f};
		/// x = intensity, y = max roughness that still reflects, z = thickness, w = step count
		glm::vec4 tuning {0.0f};
		/// x = stride in world units, y = render mode, zw = spare
		glm::vec4 marching {0.0f};
	};

	void createResources(const VulkanCore& core);
	void createTarget(const VulkanCore& core, vk::Extent2D extent);

	const VulkanCore* m_core = nullptr;
	vk::Format m_scene_format = vk::Format::eUndefined;

	ShaderLayout m_shader_layout;
	VulkanPipeline m_pipeline;

	/// Colour is sampled with filtering; depth and the geometry buffer are not - see createResources()
	vk::raii::Sampler m_sampler = nullptr;
	vk::raii::Sampler m_point_sampler = nullptr;

	std::vector<vk::raii::DescriptorSet> m_descriptor_sets;
	std::vector<vk::ImageView> m_bound_views;

	PostProcessTarget m_target;
};

}
