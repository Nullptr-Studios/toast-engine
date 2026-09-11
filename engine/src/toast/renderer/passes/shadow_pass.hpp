/**
 * @file shadow_pass.hpp
 * @author dario
 * @date 01/08/2026
 */

#pragma once

#include "../render_pass_base.hpp"
#include "../shader_layout.hpp"
#include "../shadow_constants.hpp"
#include "../vulkan_common.hpp"
#include "../vulkan_pipeline.hpp"

#include <array>
#include <glm/glm.hpp>
#include <optional>
#include <span>
#include <vector>

namespace renderer {
class VulkanCore;

/// @brief Creates the depth-comparison sampler every shadow map is read through
///
/// A free function because MaterialPass must bind *some* comparison sampler even with no ShadowPass - a
/// plain sampler on a shadow-typed descriptor is invalid usage
///
/// @param format Depth format read; linear filtering is dropped if the driver cannot filter it
[[nodiscard]]
auto createShadowSampler(const VulkanCore& core, vk::Format format) -> vk::raii::Sampler;

/// @brief Renders scene depth from every shadow-casting light into two layered shadow maps
///
/// A point light's cube is six ordinary array layers rather than a real cube map, so nothing depends on
/// `imageCubeArray`; the shader selects the face
///
/// Everything records in recordPre(), so both maps are readable before any MaterialPass samples them in the
/// same command buffer. Which lights cast and every view matrix come from the RenderFrame snapshot
class ShadowPass : public IRenderPass {
public:
	explicit ShadowPass(const VulkanCore& core);

	[[nodiscard]]
	auto name() const -> std::string_view override {
		return "ShadowPass";
	}

	void recordPre(vk::CommandBuffer cmd, uint32_t frame_index, uint32_t image_index) override;

	/// @brief No-op - see recordPre()
	void record(vk::CommandBuffer cmd, uint32_t frame_index, uint32_t image_index) override { }

	[[nodiscard]]
	auto getCascadeMapView(uint32_t frame_index) const -> vk::ImageView;

	[[nodiscard]]
	auto getPunctualMapView(uint32_t frame_index) const -> vk::ImageView;

	/// @returns the sampler both maps are bound with, clamped to an opaque-white border so a lookup outside
	///          a fitted frustum reads as unshadowed
	[[nodiscard]]
	auto getShadowSampler() const -> vk::Sampler {
		return *m_sampler;
	}

	/// @returns draw calls last frame - the only evidence instancing and multiview do anything, since
	///          neither changes the image
	[[nodiscard]]
	auto getDrawCount() const noexcept -> uint32_t {
		return m_draw_count;
	}

	[[nodiscard]]
	auto getPassCount() const noexcept -> uint32_t {
		return m_pass_count;
	}

private:
	/// @brief Mirrors shadow_depth.slang's ShadowUBO
	struct ShadowUBO {
		std::array<glm::mat4, shadows::k_max_shadow_views> view_projection {};
	};

	/// @brief Mirrors shadow_depth.slang's PushConstants
	///
	/// No transform - it comes from RenderFrame::shadow_instance_data, so same-mesh proxies batch
	struct ShadowPushConstants {
		uint32_t instance_base = 0;
		uint32_t view_index = 0;
	};

	/// @brief Consecutive layers rendered as one pass
	///
	/// More than one layer is multiview, recorded once with `SV_ViewID` picking the matrix. Fixed rather than
	/// per frame, because a pipeline's `viewMask` must equal the scope's
	struct LayerGroup {
		uint32_t base_layer = 0;
		uint32_t layer_count = 1;

		/// One bit per layer. Never 0 - that means "not multiview" and leaves `SV_ViewID` undefined for a
		/// shader reading it unconditionally, so a single-layer group uses the one-view mask 0x1
		uint32_t view_mask = 1;

		std::optional<vk::raii::ImageView> view;

		/// One clear after losing its views, or a light that stopped casting keeps shadowing from last frame.
		/// True on creation, since a fresh image holds undefined depth
		bool dirty = true;
	};

	struct ShadowMap {
		std::optional<vma::raii::Image> image;
		std::optional<vk::raii::ImageView> array_view;           ///< sampled by the material shaders
		std::vector<LayerGroup> groups;                          ///< render targets, in layer order
		vk::ImageLayout layout = vk::ImageLayout::eUndefined;    ///< render-thread-only, see recordPre()
		uint32_t resolution = 0;
		uint32_t layer_count = 0;
	};

	/// @brief Pipelines for one multiview mask - Vulkan compares a pipeline's `viewMask` against the scope's
	///        exactly, so cascades, cubes and spots each need their own
	struct PipelineSet {
		uint32_t view_mask = 1;
		VulkanPipeline pipeline;
	};

	struct FrameTarget {
		ShadowMap cascades;
		ShadowMap punctual;
		FrameResources ubo;
	};

	void createResources(const VulkanCore& core);

	/// @param group_layers Layer count of each render-target group, in order; entries above 1 become multiview
	///                     passes. Must sum to the map's total layer count
	void createShadowMap(
	    const VulkanCore& core, ShadowMap& map, uint32_t resolution, std::span<const uint32_t> group_layers,
	    std::string_view debug_name
	);

	/// @returns the pipelines built for @p view_mask, or nullptr when none were
	[[nodiscard]]
	auto pipelineSetFor(uint32_t view_mask) const -> const PipelineSet*;

	/// @brief Renders every layer of @p map, clearing any without a view so a stale depth cannot keep
	///        shadowing
	///
	/// @param directional Selects which of RenderFrame's shadow views target this map
	void recordMap(vk::CommandBuffer cmd, ShadowMap& map, uint32_t frame_index, bool directional);

	[[nodiscard]]
	static auto selectShadowFormat(const VulkanCore& core) -> vk::Format;

	const VulkanCore* m_core = nullptr;
	vk::Format m_format = vk::Format::eUndefined;

	ShaderLayout m_shader_layout;

	/// One per multiview mask: single view, cascades, a cube's six faces. A fixed array because
	/// VulkanPipeline owns a vk::raii::Pipeline and is not movable
	std::array<PipelineSet, 3> m_pipeline_sets;

	uint32_t m_draw_count = 0;
	uint32_t m_pass_count = 0;

	vk::raii::Sampler m_sampler = nullptr;
	std::vector<vk::raii::DescriptorSet> m_descriptor_sets;
	std::vector<FrameTarget> m_targets;
};

}
