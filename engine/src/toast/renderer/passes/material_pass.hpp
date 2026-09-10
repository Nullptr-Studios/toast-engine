/**
 * @file material_pass.hpp
 * @author Xein
 * @date 17 Jul 2026
 */

#pragma once

#include "../material_runtime.hpp"
#include "../render_pass_base.hpp"
#include "../scene_descriptor_set.hpp"
#include "../shader_layout.hpp"
#include "../vulkan_pipeline.hpp"
#include "../vulkan_renderer.hpp"

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <toast/assets/material.hpp>
#include <unordered_map>
#include <vector>

namespace renderer {
class VulkanCore;

/**
 * @class MaterialPass
 * @brief Draws every mesh instance whose root material owns this pass
 */
class MaterialPass : public IRenderPass {
public:
	MaterialPass(
	    const VulkanCore& core, assets::Material* root_material, vk::Format color_format, vk::Format depth_format,
	    vk::Extent2D extent
	);

	[[nodiscard]]
	auto name() const -> std::string_view override {
		return m_name;
	}

	void record(vk::CommandBuffer cmd, uint32_t frame_index, uint32_t image_index) override;

	/// @brief Records one instance
	void recordInstance(vk::CommandBuffer cmd, uint32_t frame_index, const VulkanRenderer::MeshInstanceProxy& proxy);

	[[nodiscard]]
	auto rootMaterial() const -> assets::Material* {
		return m_root_material;
	}

	/// Schedules a full pipeline and descriptor rebuild at the start of the next record
	void markShadersDirty() { m_rebuild_pending.store(true, std::memory_order_release); }

	/// Schedules a re-bake of parameter values
	void markValuesDirty() { m_values_dirty.store(true, std::memory_order_release); }

	[[nodiscard]]
	auto isValid() const -> bool {
		return m_pipeline.isReady();
	}

	/// @returns true when this pass built the alpha-cutout variant
	[[nodiscard]]
	auto usesCutout() const noexcept -> bool {
		return m_uses_cutout;
	}

	/// @returns true when this pass blends rather than writing opaque coverage
	[[nodiscard]]
	auto isBlended() const -> bool {
		return m_root_material != nullptr && m_root_material->settings().blend_mode != assets::BlendMode::opaque;
	}

private:
	/// @returns the root material resolved alphaCutoff
	[[nodiscard]]
	auto resolvedAlphaCutoff() -> float;

	/// @returns @p runtime resolved alphaCutoff, 0 when it declares none
	[[nodiscard]]
	static auto resolvedAlphaCutoffOf(MaterialRuntime& runtime) -> float;

	bool m_uses_cutout = false;

	/// @brief Binds this instance material resources and issues its draw
	/// @param posed_vertex_offset Slice of @p posed_vertices
	/// @param bound_material Caller's currently-bound cache
	/// @param instance_count Proxies in this run
	void drawInstance(
	    vk::CommandBuffer cmd, uint32_t frame_index, const VulkanRenderer::MeshInstanceProxy& proxy, vk::Buffer posed_vertices,
	    uint32_t posed_vertex_offset, assets::Material** bound_material, uint32_t instance_count = 1
	);

	/// Per material GPU resources within this pass
	struct InstanceResources {
		std::unique_ptr<MaterialRuntime> runtime;

		struct UboBuffer {
			uint32_t set = 0;
			uint32_t binding = 0;
			std::vector<std::optional<vma::raii::Buffer>> buffers;
		};

		std::vector<UboBuffer> ubo_buffers;
		std::vector<std::vector<vk::raii::DescriptorSet>> sets;
		std::vector<std::vector<vk::ImageView>> bound_views;
	};

	void rebuildPipeline();
	auto ensureInstanceResources(assets::Material* material) -> InstanceResources*;
	void updateInstanceDescriptors(InstanceResources& res, uint32_t frame_index);

	const VulkanCore* m_core = nullptr;
	assets::Material* m_root_material = nullptr;

	/// @brief Keeps the material alive for as long as this pass exists
	assets::Handle<assets::Material> m_root_material_ref;
	std::string m_name;

	vk::Format m_color_format = vk::Format::eUndefined;
	vk::Format m_depth_format = vk::Format::eUndefined;
	vk::Extent2D m_extent;

	MaterialRuntime m_root_runtime;
	ShaderLayout m_layout;
	/// One pipeline for both static and skinned - a posed instance arrives already deformed
	VulkanPipeline m_pipeline;

	/// Set 0, the engine-owned half. Shared with any pass importing the same lighting module
	SceneDescriptorSets m_scene_sets;
	std::unordered_map<assets::Material*, InstanceResources> m_instances;

	/// Reused scratch - as locals these were a heap allocation per draw and per pass per frame
	std::vector<std::byte> m_push_scratch;
	std::vector<uint32_t> m_draw_order;

	std::atomic_bool m_rebuild_pending {false};
	std::atomic_bool m_values_dirty {false};
};

}
