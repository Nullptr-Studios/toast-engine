/// @file debug_pass.hpp
/// @author dario
/// @date 10/06/2026.

#pragma once
#include "../render_pass_base.hpp"
#include "../shader_layout.hpp"
#include "../vulkan_common.hpp"
#include "../vulkan_pipeline.hpp"

#include <array>
#include <glm/glm.hpp>
#include <toast/world/gizmo_layout.hpp>
#include <vector>

namespace renderer {
class VulkanCore;
class ClusterLightingPass;

/**
 * @brief Editor/debug visualization pass: ground grid, immediate-mode debug lines, and axis gizmos
 *
 * Debug lines and gizmo transforms aren't owned by this class - they're queued via the free functions in
 * vulkan_renderer.hpp (debugDrawLine(), debugDrawBox(), debugDrawSphere(), debugDrawAxes()) between
 * beginFrameBuild() and submitFrame(), exactly like MeshInstanceProxy. They flow through the same
 * VulkanRenderer::RenderFrame snapshot
 */
class DebugPass : public IRenderPass {
public:
	DebugPass(
	    const renderer::VulkanCore& core, vk::Format color_format, vk::Format depth_format, vk::Extent2D extent,
	    const ClusterLightingPass& cluster_lighting_pass
	);

	~DebugPass() override;

	void record(vk::CommandBuffer cmd, uint32_t frame_index, uint32_t image_index) override;

	void update(uint32_t frame_index, float dt) override;

	/// @brief Toggles the ground grid
	void setGridEnabled(bool enabled) { m_grid_enabled = enabled; }

	[[nodiscard]]
	auto gridEnabled() const -> bool {
		return m_grid_enabled;
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
	/// @brief Sets up a Dear ImGui context + the Vulkan backend, sharing this pass's dynamic-rendering scope.
	/// Input is fed from RenderFrame::imgui_input (resolved main-thread-side, same pattern as the gizmo state)
	/// since ImGui itself isn't thread-safe and NewFrame()/Render() below both run on the render thread
	void initImGui(const renderer::VulkanCore& core, vk::Format color_format, vk::Format depth_format);
	void createGridGeometry(const renderer::VulkanCore& core);
	void createGizmoGeometry(const renderer::VulkanCore& core);
	void createTranslateGizmoGeometry(const renderer::VulkanCore& core);
	void createRotateGizmoGeometry(const renderer::VulkanCore& core);
	void createScaleGizmoGeometry(const renderer::VulkanCore& core);

	/// @brief Grows @p buffer so it can hold at least @p required_vertex_count DebugVertex entries
	void ensureLineCapacity(const renderer::VulkanCore& core, DynamicVertexBuffer& buffer, size_t required_vertex_count);

	VulkanPipeline m_plane_pipeline;
	VulkanPipeline m_line_pipeline;
	VulkanPipeline m_gizmo_pipeline;

	ShaderLayout m_shader_layout;
	std::vector<vk::raii::DescriptorSet> m_frame_descriptor_sets;

	// Ground grid
	bool m_grid_enabled = true;
	vma::raii::Buffer m_grid_vertex_buffer = nullptr;

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

	// Dear ImGui, for ad-hoc debug UI. Editor-viewport-only (this pass isn't constructed for the SDL/player
	// window), so it doesn't need a platform backend - input is fed manually from RenderFrame::imgui_input
	bool m_imgui_ready = false;

	// Non-owning; used to read back cluster light counts for the cluster-heatmap overlay's text labels.
	// Always valid - engine.cpp constructs ClusterLightingPass before this pass and keeps it alive for the
	// renderer's lifetime
	const ClusterLightingPass* m_cluster_lighting_pass = nullptr;
};

}
