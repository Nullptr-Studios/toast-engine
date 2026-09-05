/**
 * @file depth_prepass.hpp
 * @author dario
 * @date 13/08/2026
 */

#pragma once

#include "../shader_layout.hpp"
#include "../vulkan_common.hpp"
#include "../vulkan_pipeline.hpp"

#include <cstdint>
#include <string_view>
#include <vector>

namespace renderer {
class VulkanCore;

/**
 * @brief Lays down opaque depth first, so the main pass rejects occluded fragments at the depth test
 *
 * Forward+ shades every fragment that reaches the fragment stage, really expensive six texture
 * fetches, a 12-tap shadow lookup, clustered lights, IBL and an 8-probe SH interpolation. Overdraw costs far
 * more than the geometry that caused it
 *
 * @note Only pays off because non-cutout materials compile without a discard. A shader that can discard
 *       forces late-Z, and then the main pass shades occluded fragments anyway and this is pure added cost.
 * @note Cutout materials are skipped - their silhouette comes from an alpha test, and a depth-only pipeline
 *       has no fragment stage to run it, so prepassing them would punch out depth the cutout discards
 */
class DepthPrepass {
public:
	DepthPrepass(const VulkanCore& core, vk::Format depth_format, vk::Extent2D extent);

	[[nodiscard]]
	auto isReady() const -> bool {
		return m_pipeline.isReady();
	}

	/// @brief Records depth-only draws for every visible opaque non-cutout instance
	///
	/// Expects an already-open rendering scope with a depth attachment and no colour attachments
	void record(vk::CommandBuffer cmd, uint32_t frame_index);

	/// @returns instances drawn last frame, for the debug panel
	[[nodiscard]]
	auto getDrawnCount() const noexcept -> uint32_t {
		return m_drawn;
	}

private:
	void createResources(const VulkanCore& core);

	const VulkanCore* m_core = nullptr;

	ShaderLayout m_shader_layout;
	VulkanPipeline m_pipeline;

	/// Set 0 holds only what a position needs: the camera, the instance buffer and the joint matrices
	std::vector<vk::raii::DescriptorSet> m_descriptor_sets;

	uint32_t m_drawn = 0;
};

}
