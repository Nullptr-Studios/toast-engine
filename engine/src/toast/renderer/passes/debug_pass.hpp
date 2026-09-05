/// @file debug_pass.hpp
/// @author dario
/// @date 10/06/2026

#pragma once
#include "../render_pass_base.hpp"
#include "../shader_layout.hpp"
#include "../vulkan_common.hpp"
#include "../vulkan_pipeline.hpp"

#include <array>
#include <glm/glm.hpp>
#include <toast/world/gizmo_layout.hpp>
#include <unordered_map>
#include <vector>

namespace renderer {
class VulkanCore;
class ClusterLightingPass;

/// @brief Editor/debug visualization pass: ground grid, immediate-mode debug lines, and axis gizmos
///
/// Nothing drawn here is owned here - it is queued through vulkan_renderer.hpp's free functions and arrives
/// in the same RenderFrame snapshot as a MeshInstanceProxy
class DebugPass : public IRenderPass {
public:
	/// @param cluster_lighting_pass Optional. Only the cluster heatmap view reads it, so a renderer with no
	///        clustered lighting - a fully ray-traced path, say - can still use the whole debug overlay
	///        by passing nullptr
	DebugPass(
	    const renderer::VulkanCore& core, vk::Format color_format, vk::Format depth_format, vk::Extent2D extent,
	    const ClusterLightingPass* cluster_lighting_pass = nullptr
	);

	~DebugPass() override;

	/// Editor overlay, not scene content: ImGui and the debug primitives are authored as literal colours, and
	/// running them through the tone curve would shift every one of them. Still depth-tested against the
	/// scene - the output scope binds the depth buffer the world scope wrote
	[[nodiscard]]
	auto stage() const -> RenderStage override {
		return RenderStage::overlay;
	}

	void record(vk::CommandBuffer cmd, uint32_t frame_index, uint32_t image_index) override;

	void update(uint32_t frame_index, float dt) override;

	[[nodiscard]]
	auto name() const -> std::string_view override {
		return "Debug";
	}

private:
	struct DrawPushConstants {
		glm::mat4 model;
		glm::vec4 tint {1.0f};
	};

	/// @brief Vertex range for one translate-gizmo
	struct GizmoHandleRange {
		uint32_t first_vertex = 0;
		uint32_t vertex_count = 0;
		glm::vec4 base_color {1.0f};
	};

	/// @brief A vk::raii-owned buffer that can grow
	struct DynamicVertexBuffer {
		vma::raii::Buffer buffer = nullptr;
		void* mapped = nullptr;
		vk::DeviceSize capacity_bytes = 0;
	};

	void createResources(const renderer::VulkanCore& core);

	void initImGui(const renderer::VulkanCore& core, vk::Format color_format, vk::Format depth_format);
	void createGizmoGeometry(const renderer::VulkanCore& core);
	void createTranslateGizmoGeometry(const renderer::VulkanCore& core);
	void createRotateGizmoGeometry(const renderer::VulkanCore& core);
	void createScaleGizmoGeometry(const renderer::VulkanCore& core);

	/// @brief Builds the billboard pipeline, its own ShaderLayout and its per-frame frame-UBO sets
	void createBillboardResources(
	    const renderer::VulkanCore& core, vk::Format color_format, vk::Format depth_format, vk::Extent2D extent
	);

	/// @brief Returns the set-1 descriptor set bound to @p view, allocating it on first use
	///
	/// Whatever called debugDrawBillboard() picks the textures, so these are cached by view rather than
	/// created up front. Debug icons are a fixed handful, so it never grows unbounded
	auto billboardTextureSet(const renderer::VulkanCore& core, vk::ImageView view) -> vk::DescriptorSet;

	/// @brief Grows @p buffer so it can hold at least @p required_vertex_count DebugVertex entries
	void ensureLineCapacity(const renderer::VulkanCore& core, DynamicVertexBuffer& buffer, size_t required_vertex_count);

	VulkanPipeline m_line_pipeline;
	VulkanPipeline m_gizmo_pipeline;

	ShaderLayout m_shader_layout;
	std::vector<vk::raii::DescriptorSet> m_frame_descriptor_sets;

	// Debug lines
	std::vector<DynamicVertexBuffer> m_line_vertex_buffers;
	std::vector<uint32_t> m_line_vertex_counts;

	// Gizmo axis triad, generic
	vma::raii::Buffer m_gizmo_vertex_buffer = nullptr;
	uint32_t m_gizmo_vertex_count = 0;

	// Translate gizmo, one static buffer, 7 independently tintable handle sub-ranges
	vma::raii::Buffer m_translate_gizmo_vertex_buffer = nullptr;
	std::array<GizmoHandleRange, 7> m_translate_gizmo_handles;

	// Rotate gizmo
	vma::raii::Buffer m_rotate_gizmo_vertex_buffer = nullptr;
	std::array<GizmoHandleRange, 7> m_rotate_gizmo_handles;

	// Scale gizmo, 3 cube-tipped axis handles
	vma::raii::Buffer m_scale_gizmo_vertex_buffer = nullptr;
	std::array<GizmoHandleRange, 7> m_scale_gizmo_handles;

	// Camera-facing textured icons, Separate shader/layout from the untextured debug
	// shapes above, since those have no set 1 and no sampler
	struct BillboardPushConstants {
		glm::vec4 center_size;    ///< xyz world-space centre, w world-space edge length
		glm::vec4 tint {1.0f};
	};

	VulkanPipeline m_billboard_pipeline;
	ShaderLayout m_billboard_layout;
	std::vector<vk::raii::DescriptorSet> m_billboard_frame_sets;
	vk::raii::Sampler m_billboard_sampler = nullptr;
	std::unordered_map<VkImageView, vk::raii::DescriptorSet> m_billboard_texture_sets;

	bool m_imgui_ready = false;

	const ClusterLightingPass* m_cluster_lighting_pass = nullptr;

	float m_sky_intensity_ui = -1.0f;

	std::array<char, 256> m_environment_uri_ui {};
};

}
