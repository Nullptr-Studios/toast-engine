/// @file debug_pass.cpp
/// @author dario
/// @date 10/06/2026.

#include "debug_pass.hpp"

#include "../clustered_lighting_constants.hpp"
#include "../shader_compiler.hpp"
#include "../vulkan_core.hpp"
#include "../vulkan_debug.hpp"
#include "../vulkan_renderer.hpp"
#include "cluster_lighting_pass.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <format>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <imgui_impl_vulkan.h>
#include <string>
#include <toast/log.hpp>

namespace renderer {

namespace {

using DebugVertex = VulkanRenderer::DebugVertex;

/// @brief Appends an axis-aligned box
void appendBox(std::vector<DebugVertex>& out, glm::vec3 min, glm::vec3 max, glm::vec4 color) {
	const std::array<glm::vec3, 8> v {
	  glm::vec3 {min.x, min.y, min.z},
	  glm::vec3 {max.x, min.y, min.z},
	  glm::vec3 {max.x, max.y, min.z},
	  glm::vec3 {min.x, max.y, min.z},
	  glm::vec3 {min.x, min.y, max.z},
	  glm::vec3 {max.x, min.y, max.z},
	  glm::vec3 {max.x, max.y, max.z},
	  glm::vec3 {min.x, max.y, max.z},
	};

	// Two triangles per face
	static constexpr std::array<std::array<int, 4>, 6> faces {
	  {{0, 1, 2, 3}, {5, 4, 7, 6}, {4, 0, 3, 7}, {1, 5, 6, 2}, {4, 5, 1, 0}, {3, 2, 6, 7}}
	};

	for (const auto& f : faces) {
		out.push_back({v[f[0]], color});
		out.push_back({v[f[1]], color});
		out.push_back({v[f[2]], color});
		out.push_back({v[f[0]], color});
		out.push_back({v[f[2]], color});
		out.push_back({v[f[3]], color});
	}
}

/// @brief Appends a thin box running from 0 to @p length along @p axis (0=X, 1=Y, 2=Z)
void appendShaftAlongAxis(std::vector<DebugVertex>& out, int axis, float length, float half_size, glm::vec4 color) {
	glm::vec3 min {-half_size, -half_size, -half_size};
	glm::vec3 max {half_size, half_size, half_size};
	min[axis] = 0.0f;
	max[axis] = length;
	appendBox(out, min, max, color);
}

/// @brief Appends a square pyramid running from @p base_pos to @p apex_pos along @p axis
void appendPyramidAlongAxis(
    std::vector<DebugVertex>& out, int axis, float base_pos, float apex_pos, float half_size, glm::vec4 color
) {
	const int u = (axis + 1) % 3;
	const int w = (axis + 2) % 3;

	auto make = [&](float main, float along_u, float along_w) {
		glm::vec3 p {0.0f, 0.0f, 0.0f};
		p[axis] = main;
		p[u] = along_u;
		p[w] = along_w;
		return p;
	};

	const glm::vec3 apex = make(apex_pos, 0.0f, 0.0f);
	const std::array<glm::vec3, 4> base {
	  make(base_pos, half_size, half_size),
	  make(base_pos, half_size, -half_size),
	  make(base_pos, -half_size, -half_size),
	  make(base_pos, -half_size, half_size),
	};

	for (int i = 0; i < 4; ++i) {
		const int j = (i + 1) % 4;
		out.push_back({apex, color});
		out.push_back({base[i], color});
		out.push_back({base[j], color});
	}

	// Base cap, so the arrowhead isn't see-through where it meets the shaft
	out.push_back({base[0], color});
	out.push_back({base[2], color});
	out.push_back({base[1], color});
	out.push_back({base[0], color});
	out.push_back({base[3], color});
	out.push_back({base[2], color});
}

/// @brief Appends a flat square quad in the plane whose normal is @p axis (0=X,1=Y,2=Z)
void appendQuad(std::vector<DebugVertex>& out, int axis, float offset, float size, glm::vec4 color) {
	const int u = (axis + 1) % 3;
	const int w = (axis + 2) % 3;

	auto make = [&](float along_u, float along_w) {
		glm::vec3 p {0.0f, 0.0f, 0.0f};
		p[u] = along_u;
		p[w] = along_w;
		return p;
	};

	const glm::vec3 a = make(offset, offset);
	const glm::vec3 b = make(offset + size, offset);
	const glm::vec3 c = make(offset + size, offset + size);
	const glm::vec3 d = make(offset, offset + size);

	out.push_back({a, color});
	out.push_back({b, color});
	out.push_back({c, color});
	out.push_back({a, color});
	out.push_back({c, color});
	out.push_back({d, color});
}

/// @brief Appends a flat ring lying in the plane whose normal is @p axis (0=X,1=Y,2=Z)
void appendRing(std::vector<DebugVertex>& out, int axis, float radius, float thickness, int segments, glm::vec4 color) {
	const int u = (axis + 1) % 3;
	const int w = (axis + 2) % 3;
	const float inner = radius - (thickness * 0.5f);
	const float outer = radius + (thickness * 0.5f);

	auto make = [&](float r, float angle) {
		glm::vec3 p {0.0f, 0.0f, 0.0f};
		p[u] = r * std::cos(angle);
		p[w] = r * std::sin(angle);
		return p;
	};

	for (int i = 0; i < segments; ++i) {
		const float a0 = (static_cast<float>(i) / static_cast<float>(segments)) * glm::two_pi<float>();
		const float a1 = (static_cast<float>(i + 1) / static_cast<float>(segments)) * glm::two_pi<float>();

		const glm::vec3 i0 = make(inner, a0);
		const glm::vec3 o0 = make(outer, a0);
		const glm::vec3 i1 = make(inner, a1);
		const glm::vec3 o1 = make(outer, a1);

		out.push_back({i0, color});
		out.push_back({o0, color});
		out.push_back({o1, color});
		out.push_back({i0, color});
		out.push_back({o1, color});
		out.push_back({i1, color});
	}
}

}    // namespace

DebugPass::DebugPass(
    const renderer::VulkanCore& core, vk::Format color_format, vk::Format depth_format, vk::Extent2D extent,
    const ClusterLightingPass& cluster_lighting_pass
)
    : m_cluster_lighting_pass(&cluster_lighting_pass) {
	m_shader_layout.rebuild(core, "debug");

	const vk::VertexInputBindingDescription position_only_binding(0, sizeof(glm::vec3), vk::VertexInputRate::eVertex);
	const std::vector<vk::VertexInputAttributeDescription> position_only_attributes {
	  vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, 0)
	};

	const vk::VertexInputBindingDescription debug_vertex_binding(0, sizeof(DebugVertex), vk::VertexInputRate::eVertex);
	const std::vector<vk::VertexInputAttributeDescription> debug_vertex_attributes {
	  vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(DebugVertex, position)),
	  vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(DebugVertex, color)),
	};

	// Ground grid
	{
		auto shader = renderer::ShaderCompiler::compileShaderModuleFromSource("./grid.slang");

		VulkanPipeline::Config config;
		config.pipeline_type = VulkanPipeline::PipelineType::graphics;
		config.debug_name = "DebugPass Grid";
		config.color_format = color_format;
		config.depth_format = depth_format;
		config.extent = extent;
		config.shader_spirv = std::move(shader.spirv);
		config.pipeline_layout = *m_shader_layout.getPipelineLayout();
		config.vertex_binding = position_only_binding;
		config.vertex_attributes = position_only_attributes;
		config.topology = vk::PrimitiveTopology::eTriangleList;
		config.cull_mode = vk::CullModeFlagBits::eNone;
		config.depth_test = true;
		config.depth_write = false;
		config.blend_enable = true;
		m_plane_pipeline.rebuild(core, config);
	}

	// Debug lines
	{
		auto shader = renderer::ShaderCompiler::compileShaderModuleFromSource("./debug_shape.slang");

		VulkanPipeline::Config config;
		config.pipeline_type = VulkanPipeline::PipelineType::graphics;
		config.debug_name = "DebugPass Lines";
		config.color_format = color_format;
		config.depth_format = depth_format;
		config.extent = extent;
		config.shader_spirv = std::move(shader.spirv);
		config.pipeline_layout = *m_shader_layout.getPipelineLayout();
		config.vertex_binding = debug_vertex_binding;
		config.vertex_attributes = debug_vertex_attributes;
		config.topology = vk::PrimitiveTopology::eLineList;
		config.cull_mode = vk::CullModeFlagBits::eNone;
		config.depth_test = true;
		config.depth_write = false;
		config.blend_enable = false;
		m_line_pipeline.rebuild(core, config);
	}

	// Gizmo axis triad
	{
		auto shader = renderer::ShaderCompiler::compileShaderModuleFromSource("./debug_shape.slang");

		VulkanPipeline::Config config;
		config.pipeline_type = VulkanPipeline::PipelineType::graphics;
		config.debug_name = "DebugPass Gizmo";
		config.color_format = color_format;
		config.depth_format = depth_format;
		config.extent = extent;
		config.shader_spirv = std::move(shader.spirv);
		config.pipeline_layout = *m_shader_layout.getPipelineLayout();
		config.vertex_binding = debug_vertex_binding;
		config.vertex_attributes = debug_vertex_attributes;
		config.topology = vk::PrimitiveTopology::eTriangleList;
		config.cull_mode = vk::CullModeFlagBits::eNone;
		config.depth_test = false;
		config.depth_write = false;
		config.blend_enable = false;
		m_gizmo_pipeline.rebuild(core, config);
	}

	createResources(core);
	initImGui(core, color_format, depth_format);
}

DebugPass::~DebugPass() {
	if (m_imgui_ready) {
		ImGui_ImplVulkan_Shutdown();
		ImGui::DestroyContext();
	}
}

namespace {

/// @brief Maps the SDL-keycode encoding used by event::WindowKey (a printable codepoint, or
/// (1<<30)|scancode for non-printable keys) to the small subset of ImGuiKey this engine's C# side actually
/// forwards (see ViewportControl.axaml.cs's MapKey())
auto sdlKeyToImGuiKey(int32_t key) -> ImGuiKey {
	constexpr int32_t k_scancode_mask = 1 << 30;

	if (key >= 'a' && key <= 'z') {
		return static_cast<ImGuiKey>(ImGuiKey_A + (key - 'a'));
	}
	if (key >= '0' && key <= '9') {
		return static_cast<ImGuiKey>(ImGuiKey_0 + (key - '0'));
	}

	switch (key) {
		case 32: return ImGuiKey_Space;
		case 13: return ImGuiKey_Enter;
		case 27: return ImGuiKey_Escape;
		case 8: return ImGuiKey_Backspace;
		case 9: return ImGuiKey_Tab;
		case 127: return ImGuiKey_Delete;
		default: break;
	}

	if ((key & k_scancode_mask) != 0) {
		switch (key & ~k_scancode_mask) {
			case 79: return ImGuiKey_RightArrow;
			case 80: return ImGuiKey_LeftArrow;
			case 81: return ImGuiKey_DownArrow;
			case 82: return ImGuiKey_UpArrow;
			case 225: return ImGuiKey_LeftShift;
			case 229: return ImGuiKey_RightShift;
			case 224: return ImGuiKey_LeftCtrl;
			case 228: return ImGuiKey_RightCtrl;
			case 226: return ImGuiKey_LeftAlt;
			case 230: return ImGuiKey_RightAlt;
			case 58: return ImGuiKey_F1;
			case 59: return ImGuiKey_F2;
			case 60: return ImGuiKey_F3;
			case 61: return ImGuiKey_F4;
			case 62: return ImGuiKey_F5;
			case 63: return ImGuiKey_F6;
			case 64: return ImGuiKey_F7;
			case 65: return ImGuiKey_F8;
			case 66: return ImGuiKey_F9;
			case 67: return ImGuiKey_F10;
			case 68: return ImGuiKey_F11;
			case 69: return ImGuiKey_F12;
			default: break;
		}
	}

	return ImGuiKey_None;
}

/// @brief Cluster light-count -> color ramp for the ImGui cluster-grid overlay - mirrors
/// clusterHeatmapColor() in mesh.slang exactly, so the overlay and the per-pixel shader heatmap agree
auto clusterHeatmapColorImGui(uint32_t light_count) -> ImVec4 {
	constexpr float k_heatmap_max_lights = 8.0f;
	const float t = std::clamp(static_cast<float>(light_count) / k_heatmap_max_lights, 0.0f, 1.0f);

	const ImVec4 c0(0.0f, 0.0f, 1.0f, 1.0f);
	const ImVec4 c1(0.0f, 1.0f, 0.0f, 1.0f);
	const ImVec4 c2(1.0f, 1.0f, 0.0f, 1.0f);
	const ImVec4 c3(1.0f, 0.0f, 0.0f, 1.0f);

	auto lerp = [](const ImVec4& a, const ImVec4& b, float u) {
		return ImVec4(a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u, a.z + (b.z - a.z) * u, 1.0f);
	};

	if (t < 0.333f) {
		return lerp(c0, c1, t / 0.333f);
	}
	if (t < 0.667f) {
		return lerp(c1, c2, (t - 0.333f) / 0.334f);
	}
	return lerp(c2, c3, (t - 0.667f) / 0.333f);
}

}    // namespace

void DebugPass::update(uint32_t frame_index, float dt) {
	const auto* frame = VulkanRenderer::instance->renderingFrame();
	if (frame == nullptr || frame_index >= m_line_vertex_buffers.size()) {
		return;
	}

	if (m_imgui_ready) {
		ImGuiIO& io = ImGui::GetIO();
		const auto& input = frame->imgui_input;

		io.DisplaySize = ImVec2(frame->viewport_extent.x, frame->viewport_extent.y);
		io.DeltaTime = std::max(dt, 0.0001f);

		if (input.mouse_pos.x >= 0.0f && input.mouse_pos.y >= 0.0f) {
			io.AddMousePosEvent(input.mouse_pos.x, input.mouse_pos.y);
		}
		for (size_t i = 0; i < input.mouse_down.size(); ++i) {
			io.AddMouseButtonEvent(static_cast<int>(i), input.mouse_down[i]);
		}
		if (input.mouse_wheel_x != 0.0f || input.mouse_wheel_y != 0.0f) {
			io.AddMouseWheelEvent(input.mouse_wheel_x, input.mouse_wheel_y);
		}
		for (const auto& key_event : input.key_events) {
			const ImGuiKey imgui_key = sdlKeyToImGuiKey(key_event.key);
			if (imgui_key != ImGuiKey_None) {
				io.AddKeyEvent(imgui_key, key_event.down);
			}
		}
		for (const uint32_t codepoint : input.char_events) {
			io.AddInputCharacter(codepoint);
		}

		ImGui_ImplVulkan_NewFrame();
		ImGui::NewFrame();

		if (ImGui::Begin("Toast Debug")) {
			ImGui::Text("Frame time: %.3f ms (%.1f FPS)", dt * 1000.0f, dt > 0.0f ? 1.0f / dt : 0.0f);
			ImGui::Text("Mesh instances: %zu", frame->mesh_instances.size());
			ImGui::Text("Lights: %zu", frame->lights.size());
			ImGui::Separator();
			ImGui::Text("Render mode: %s", frame->render_mode == 1 ? "Cluster Heatmap (toolbar Mode button)" : "Lit");
			if (frame->render_mode == 1) {
				ImGui::TextColored(ImVec4(0.3f, 0.6f, 1.0f, 1.0f), "Blue = 0 lights/cluster");
				ImGui::TextColored(ImVec4(1.0f, 0.2f, 0.2f, 1.0f), "Red = 8+ lights/cluster");
			}
		}
		ImGui::End();

		// Cluster-heatmap overlay: divides the viewport into the 16x9 XY tile grid (matching
		// clustered_lighting_constants.hpp), fills each cell by its worst-case light count across all Z
		// slices, and labels it with the exact number - the shader heatmap already color-codes per-pixel
		// using the correct depth-derived Z slice, this adds the screen-space grid + exact counts on top
		if (frame->render_mode == 1 && m_cluster_lighting_pass != nullptr) {
			const auto counts = m_cluster_lighting_pass->getClusterLightGridCounts(frame_index);
			if (!counts.empty()) {
				using namespace clustered_lighting;

				ImDrawList* draw_list = ImGui::GetForegroundDrawList();
				const float cell_w = frame->viewport_extent.x / static_cast<float>(k_cluster_dim_x);
				const float cell_h = frame->viewport_extent.y / static_cast<float>(k_cluster_dim_y);

				for (uint32_t ty = 0; ty < k_cluster_dim_y; ++ty) {
					for (uint32_t tx = 0; tx < k_cluster_dim_x; ++tx) {
						uint32_t max_count = 0;
						for (uint32_t tz = 0; tz < k_cluster_dim_z; ++tz) {
							const uint32_t idx = tx + ty * k_cluster_dim_x + tz * k_cluster_dim_x * k_cluster_dim_y;
							if (counts[idx] > max_count) {
								max_count = counts[idx];
							}
						}

						const ImVec2 cell_min(static_cast<float>(tx) * cell_w, static_cast<float>(ty) * cell_h);
						const ImVec2 cell_max(cell_min.x + cell_w, cell_min.y + cell_h);

						const ImVec4 heat = clusterHeatmapColorImGui(max_count);
						draw_list->AddRectFilled(cell_min, cell_max, ImGui::ColorConvertFloat4ToU32(ImVec4(heat.x, heat.y, heat.z, 0.30f)));
						draw_list->AddRect(cell_min, cell_max, IM_COL32(255, 255, 255, 50));

						const std::string label = std::format("{}", max_count);
						const ImVec2 text_size = ImGui::CalcTextSize(label.c_str());
						const ImVec2 text_pos(cell_min.x + (cell_w - text_size.x) * 0.5f, cell_min.y + (cell_h - text_size.y) * 0.5f);
						draw_list->AddText(ImVec2(text_pos.x + 1, text_pos.y + 1), IM_COL32(0, 0, 0, 200), label.c_str());
						draw_list->AddText(text_pos, IM_COL32(255, 255, 255, 255), label.c_str());
					}
				}
			}
		}

		ImGui::Render();
	}

	const auto& core = VulkanRenderer::instance->getCore();
	auto& buffer = m_line_vertex_buffers[frame_index];
	const auto& vertices = frame->debug_line_vertices;

	m_line_vertex_counts[frame_index] = static_cast<uint32_t>(vertices.size());
	if (vertices.empty()) {
		return;
	}

	ensureLineCapacity(core, buffer, vertices.size());
	std::memcpy(buffer.mapped, vertices.data(), vertices.size() * sizeof(DebugVertex));
	buffer.buffer.getAllocation().flush(0, vertices.size() * sizeof(DebugVertex));
}

void DebugPass::record(vk::CommandBuffer cmd, uint32_t frame_index, uint32_t image_index) {
	(void)image_index;

	if (frame_index >= m_frame_descriptor_sets.size()) {
		TOAST_ERROR("DebugPass", "Frame index {} out of bounds for descriptor sets", frame_index);
		return;
	}

	const auto* frame = VulkanRenderer::instance->renderingFrame();
	if (frame == nullptr) {
		return;
	}

	cmd.bindDescriptorSets(
	    vk::PipelineBindPoint::eGraphics,
	    *m_shader_layout.getPipelineLayout(),
	    0,
	    std::array<vk::DescriptorSet, 1> {*m_frame_descriptor_sets[frame_index]},
	    {}
	);

	// Ground grid:
	if (m_grid_enabled && m_plane_pipeline.isReady()) {
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, m_plane_pipeline.getPipeline());

		constexpr float k_grid_half_extent = 1000.0f;    // matches grid.slang's fade-to-zero distance
		const glm::vec3 cam_pos = frame->frame_data.camera_position;

		DrawPushConstants pc {};
		pc.model = glm::translate(glm::mat4(1.0f), glm::vec3(cam_pos.x, cam_pos.y, 0.0f)) *
		           glm::scale(glm::mat4(1.0f), glm::vec3(k_grid_half_extent));
		cmd.pushConstants(*m_shader_layout.getPipelineLayout(), vk::ShaderStageFlagBits::eVertex, 0, sizeof(DrawPushConstants), &pc);

		cmd.bindVertexBuffers(0, std::array<vk::Buffer, 1> {*m_grid_vertex_buffer}, std::array<vk::DeviceSize, 1> {0});
		cmd.draw(6, 1, 0, 0);
	}

	// Debug lines
	const uint32_t line_vertex_count = frame_index < m_line_vertex_counts.size() ? m_line_vertex_counts[frame_index] : 0;
	if (line_vertex_count > 0 && m_line_pipeline.isReady()) {
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, m_line_pipeline.getPipeline());

		// glm::mat4 has no default member initializer and GLM_FORCE_CTOR_INIT isn't defined anywhere in this
		// project, so a default-constructed DrawPushConstants{} leaves model uninitialized, not identity -
		// must set it explicitly. Line vertices are already in world space, so identity is what we want
		DrawPushConstants pc {};
		pc.model = glm::mat4(1.0f);
		cmd.pushConstants(*m_shader_layout.getPipelineLayout(), vk::ShaderStageFlagBits::eVertex, 0, sizeof(DrawPushConstants), &pc);

		cmd.bindVertexBuffers(
		    0, std::array<vk::Buffer, 1> {*m_line_vertex_buffers[frame_index].buffer}, std::array<vk::DeviceSize, 1> {0}
		);
		cmd.draw(line_vertex_count, 1, 0, 0);
	}

	// Gizmo axis triads
	if (!frame->debug_gizmo_instances.empty() && m_gizmo_pipeline.isReady() && m_gizmo_vertex_count > 0) {
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, m_gizmo_pipeline.getPipeline());
		cmd.bindVertexBuffers(0, std::array<vk::Buffer, 1> {*m_gizmo_vertex_buffer}, std::array<vk::DeviceSize, 1> {0});

		for (const auto& transform : frame->debug_gizmo_instances) {
			DrawPushConstants pc {};
			pc.model = transform;
			cmd.pushConstants(
			    *m_shader_layout.getPipelineLayout(), vk::ShaderStageFlagBits::eVertex, 0, sizeof(DrawPushConstants), &pc
			);
			cmd.draw(m_gizmo_vertex_count, 1, 0, 0);
		}
	}

	// Selection's active gizmo (Translate/Rotate/Scale)
	if (frame->transform_gizmo.visible && m_gizmo_pipeline.isReady()) {
		vk::Buffer buffer;
		const std::array<GizmoHandleRange, 7>* handles = nullptr;
		switch (frame->transform_gizmo.tool) {
			case toast::GizmoTool::translate:
				buffer = *m_translate_gizmo_vertex_buffer;
				handles = &m_translate_gizmo_handles;
				break;
			case toast::GizmoTool::rotate:
				buffer = *m_rotate_gizmo_vertex_buffer;
				handles = &m_rotate_gizmo_handles;
				break;
			case toast::GizmoTool::scale:
				buffer = *m_scale_gizmo_vertex_buffer;
				handles = &m_scale_gizmo_handles;
				break;
			default: break;
		}

		if (handles != nullptr) {
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, m_gizmo_pipeline.getPipeline());
			cmd.bindVertexBuffers(0, std::array<vk::Buffer, 1> {buffer}, std::array<vk::DeviceSize, 1> {0});

			constexpr glm::vec4 k_highlight {1.0f, 0.85f, 0.1f, 1.0f};    // hover, yellow

			for (size_t i = 0; i < handles->size(); ++i) {
				const auto& range = (*handles)[i];
				if (range.vertex_count == 0) {
					continue;
				}
				const auto handle = static_cast<toast::GizmoHandle>(i);
				const bool is_active = handle == frame->transform_gizmo.active;
				const bool highlighted = handle == frame->transform_gizmo.hover || is_active;

				DrawPushConstants pc {};
				pc.model = frame->transform_gizmo.model;

				// Scale feedback: stretch just the dragged handle's own local geometry by the live factor,
				// so it visibly grows/shrinks with the mouse instead of sitting there static during the drag
				if (frame->transform_gizmo.tool == toast::GizmoTool::scale && is_active) {
					glm::vec3 stretch {1.0f};
					if (handle == toast::GizmoHandle::center) {
						stretch = glm::vec3(frame->transform_gizmo.drag_scale_factor);
					} else {
						const auto axis_index = static_cast<int>(handle) - static_cast<int>(toast::GizmoHandle::axis_x);
						stretch[axis_index] = frame->transform_gizmo.drag_scale_factor;
					}
					pc.model = pc.model * glm::scale(glm::mat4(1.0f), stretch);
				}

				pc.tint = highlighted ? k_highlight : range.base_color;
				cmd.pushConstants(
				    *m_shader_layout.getPipelineLayout(), vk::ShaderStageFlagBits::eVertex, 0, sizeof(DrawPushConstants), &pc
				);
				cmd.draw(range.vertex_count, 1, range.first_vertex, 0);
			}
		}
	}

	// ImGui draws last, always on top - shares this pass's already-open dynamic rendering scope, no
	// separate begin/end needed
	if (m_imgui_ready) {
		ImDrawData* draw_data = ImGui::GetDrawData();
		if (draw_data != nullptr) {
			ImGui_ImplVulkan_RenderDrawData(draw_data, cmd);
		}
	}
}

void DebugPass::createResources(const renderer::VulkanCore& core) {
	const auto& device = core.getDevice();
	const auto& layouts = m_shader_layout.getDescriptorSetLayouts();
	if (layouts.empty()) {
		TOAST_CRITICAL("DebugPass", "ShaderLayout has no descriptor set layouts");
		return;
	}

	const vk::DescriptorSetLayout frame_set_layout = *layouts[0];
	const vk::DescriptorPool pool = VulkanRenderer::instance->getDescriptorPoolHandle();

	m_frame_descriptor_sets.clear();
	m_frame_descriptor_sets.reserve(VulkanRenderer::k_frames_in_flight);

	for (uint32_t i = 0; i < VulkanRenderer::k_frames_in_flight; ++i) {
		const vk::DescriptorSetAllocateInfo alloc_info(pool, 1, &frame_set_layout);
		auto allocated = device.allocateDescriptorSets(alloc_info);
		m_frame_descriptor_sets.push_back(std::move(allocated[0]));
		setDebugName(core, *m_frame_descriptor_sets[i], std::format("DebugPass FrameSet[{}]", i));

		const auto* frame_res = VulkanRenderer::instance->getFrameUBORes(i);
		if (!frame_res->gpu_buffer.has_value()) {
			TOAST_CRITICAL("DebugPass", "Frame UBO buffer missing for frame {}", i);
			continue;
		}

		const vk::DescriptorBufferInfo buffer_info(**frame_res->gpu_buffer, 0, sizeof(VulkanRenderer::FrameUBO));
		const vk::WriteDescriptorSet write(
		    *m_frame_descriptor_sets[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &buffer_info
		);
		device.updateDescriptorSets(write, {});
	}

	m_line_vertex_buffers.resize(VulkanRenderer::k_frames_in_flight);
	m_line_vertex_counts.assign(VulkanRenderer::k_frames_in_flight, 0);

	createGridGeometry(core);
	createGizmoGeometry(core);

	// Gizmos
	createTranslateGizmoGeometry(core);
	createRotateGizmoGeometry(core);
	createScaleGizmoGeometry(core);
}

void DebugPass::initImGui(const renderer::VulkanCore& core, vk::Format color_format, vk::Format depth_format) {
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.BackendPlatformName = "toast_engine (manual input feed)";
	io.BackendRendererName = "imgui_impl_vulkan";

	// No OS cursor/clipboard integration yet (input comes from Avalonia via events, not a platform backend) -
	// keep ImGui from trying to touch either
	io.ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;
	io.SetClipboardTextFn = nullptr;
	io.GetClipboardTextFn = nullptr;

	static const vk::Format color_formats[1] {color_format};
	vk::PipelineRenderingCreateInfo rendering_ci {};
	rendering_ci.colorAttachmentCount = 1;
	rendering_ci.pColorAttachmentFormats = color_formats;
	// Must match the actual depth attachment bound when this pass's dynamic-rendering scope is begun (this
	// pass shares that scope with MeshPass's draws), even though ImGui itself never reads/writes depth -
	// vkCmdBeginRendering with a depth attachment present requires every pipeline drawn within it to declare
	// a matching depthAttachmentFormat, or validation flags every single draw call
	rendering_ci.depthAttachmentFormat = depth_format;

	ImGui_ImplVulkan_InitInfo init_info {};
	init_info.ApiVersion = VK_API_VERSION_1_4;
	init_info.Instance = *core.getInstance();
	init_info.PhysicalDevice = *core.getPhysicalDevice();
	init_info.Device = *core.getDevice();
	init_info.QueueFamily = core.getGraphicsQueueFamilyIndex();
	init_info.Queue = core.getGraphicsQueue();
	init_info.DescriptorPoolSize = 8;    // backend creates and owns its own pool at this size
	init_info.MinImageCount = 2;
	init_info.ImageCount = VulkanRenderer::k_frames_in_flight;
	init_info.UseDynamicRendering = true;
	init_info.PipelineInfoMain.PipelineRenderingCreateInfo = static_cast<VkPipelineRenderingCreateInfo>(rendering_ci);

	m_imgui_ready = ImGui_ImplVulkan_Init(&init_info);
	if (!m_imgui_ready) {
		TOAST_ERROR("DebugPass", "Failed to initialize ImGui Vulkan backend");
		ImGui::DestroyContext();
	}
}

void DebugPass::createGridGeometry(const renderer::VulkanCore& core) {
	const std::array<glm::vec3, 6> vertices {
	  glm::vec3 {-1.0f, -1.0f, 0.0f},
	  glm::vec3 { 1.0f, -1.0f, 0.0f},
	  glm::vec3 { 1.0f,  1.0f, 0.0f},
	  glm::vec3 {-1.0f, -1.0f, 0.0f},
	  glm::vec3 { 1.0f,  1.0f, 0.0f},
	  glm::vec3 {-1.0f,  1.0f, 0.0f},
	};

	vk::BufferCreateInfo buffer_ci {};
	buffer_ci.size = sizeof(vertices);
	buffer_ci.usage = vk::BufferUsageFlagBits::eVertexBuffer;

	vma::AllocationCreateInfo alloc_ci {};
	alloc_ci.usage = vma::MemoryUsage::eAuto;
	alloc_ci.flags = vma::AllocationCreateFlagBits::eMapped | vma::AllocationCreateFlagBits::eHostAccessSequentialWrite;

	m_grid_vertex_buffer = core.getAllocator().createBuffer(buffer_ci, alloc_ci);
	setDebugName(core, *m_grid_vertex_buffer, "DebugPass GridVertexBuffer");

	void* mapped = m_grid_vertex_buffer.getAllocation().getInfo().pMappedData;
	std::memcpy(mapped, vertices.data(), sizeof(vertices));
	m_grid_vertex_buffer.getAllocation().flush(0, sizeof(vertices));
}

void DebugPass::createGizmoGeometry(const renderer::VulkanCore& core) {
	constexpr float k_shaft_length = 0.8f;
	constexpr float k_shaft_half_size = 0.02f;
	constexpr float k_head_length = 0.25f;
	constexpr float k_head_half_size = 0.06f;

	constexpr glm::vec4 k_red {1.0f, 0.1f, 0.1f, 1.0f};
	constexpr glm::vec4 k_green {0.1f, 1.0f, 0.1f, 1.0f};
	constexpr glm::vec4 k_blue {0.1f, 0.1f, 1.0f, 1.0f};

	std::vector<DebugVertex> vertices;

	for (const auto& [axis, color] : {
	       std::pair {0,   k_red},
          std::pair {1, k_green},
          std::pair {2,  k_blue}
  }) {
		appendShaftAlongAxis(vertices, axis, k_shaft_length, k_shaft_half_size, color);
		appendPyramidAlongAxis(vertices, axis, k_shaft_length, k_shaft_length + k_head_length, k_head_half_size, color);
	}

	m_gizmo_vertex_count = static_cast<uint32_t>(vertices.size());

	vk::BufferCreateInfo buffer_ci {};
	buffer_ci.size = vertices.size() * sizeof(DebugVertex);
	buffer_ci.usage = vk::BufferUsageFlagBits::eVertexBuffer;

	vma::AllocationCreateInfo alloc_ci {};
	alloc_ci.usage = vma::MemoryUsage::eAuto;
	alloc_ci.flags = vma::AllocationCreateFlagBits::eMapped | vma::AllocationCreateFlagBits::eHostAccessSequentialWrite;

	m_gizmo_vertex_buffer = core.getAllocator().createBuffer(buffer_ci, alloc_ci);
	setDebugName(core, *m_gizmo_vertex_buffer, "DebugPass GizmoVertexBuffer");

	void* mapped = m_gizmo_vertex_buffer.getAllocation().getInfo().pMappedData;
	std::memcpy(mapped, vertices.data(), vertices.size() * sizeof(DebugVertex));
	m_gizmo_vertex_buffer.getAllocation().flush(0, vertices.size() * sizeof(DebugVertex));
}

void DebugPass::createTranslateGizmoGeometry(const renderer::VulkanCore& core) {
	using namespace toast::gizmo_layout;

	constexpr glm::vec4 k_white {1.0f, 1.0f, 1.0f, 1.0f};

	std::vector<DebugVertex> vertices;

	auto record_handle = [&](toast::GizmoHandle handle, size_t start, glm::vec4 base_color) {
		m_translate_gizmo_handles[static_cast<size_t>(handle)] = {
		  static_cast<uint32_t>(start), static_cast<uint32_t>(vertices.size() - start), base_color
		};
	};

	// Axis arrows colored red, green, blue
	constexpr std::array<toast::GizmoHandle, 3> axis_handles {
	  toast::GizmoHandle::axis_x, toast::GizmoHandle::axis_y, toast::GizmoHandle::axis_z
	};

	constexpr std::array<glm::vec4, 3> axis_colors {
	  glm::vec4 { 1.0f, 0.15f, 0.15f, 1.0f},
     glm::vec4 {0.15f,  1.0f, 0.15f, 1.0f},
     glm::vec4 {0.15f, 0.15f,  1.0f, 1.0f}
	};

	for (int axis = 0; axis < 3; ++axis) {
		const size_t start = vertices.size();
		appendShaftAlongAxis(vertices, axis, k_shaft_length, k_shaft_half_size, k_white);
		appendPyramidAlongAxis(vertices, axis, k_shaft_length, k_shaft_length + k_head_length, k_head_half_size, k_white);
		record_handle(axis_handles[axis], start, axis_colors[axis]);
	}

	// Plane handles: XY/YZ/XZ, offset from the origin along their own two axes
	constexpr std::array<toast::GizmoHandle, 3> plane_handles {
	  toast::GizmoHandle::plane_xy, toast::GizmoHandle::plane_yz, toast::GizmoHandle::plane_xz
	};
	constexpr std::array<int, 3> plane_normal_axis {2, 0, 1};    // xy->Z, yz->X, xz->Y
	constexpr std::array<glm::vec4, 3> plane_colors {
	  glm::vec4 { 1.0f,  1.0f, 0.15f, 1.0f},
     glm::vec4 {0.15f,  1.0f,  1.0f, 1.0f},
     glm::vec4 { 1.0f, 0.15f,  1.0f, 1.0f}
	};
	for (int i = 0; i < 3; ++i) {
		const size_t start = vertices.size();
		appendQuad(vertices, plane_normal_axis[i], k_plane_offset, k_plane_size, k_white);
		record_handle(plane_handles[i], start, plane_colors[i]);
	}

	// Center handle freee move
	{
		const size_t start = vertices.size();
		appendBox(vertices, glm::vec3(-k_center_half_size), glm::vec3(k_center_half_size), k_white);
		record_handle(toast::GizmoHandle::center, start, glm::vec4(0.9f, 0.9f, 0.9f, 1.0f));
	}

	vk::BufferCreateInfo buffer_ci {};
	buffer_ci.size = vertices.size() * sizeof(DebugVertex);
	buffer_ci.usage = vk::BufferUsageFlagBits::eVertexBuffer;

	vma::AllocationCreateInfo alloc_ci {};
	alloc_ci.usage = vma::MemoryUsage::eAuto;
	alloc_ci.flags = vma::AllocationCreateFlagBits::eMapped | vma::AllocationCreateFlagBits::eHostAccessSequentialWrite;

	m_translate_gizmo_vertex_buffer = core.getAllocator().createBuffer(buffer_ci, alloc_ci);
	setDebugName(core, *m_translate_gizmo_vertex_buffer, "DebugPass TranslateGizmoVertexBuffer");

	void* mapped = m_translate_gizmo_vertex_buffer.getAllocation().getInfo().pMappedData;
	std::memcpy(mapped, vertices.data(), vertices.size() * sizeof(DebugVertex));
	m_translate_gizmo_vertex_buffer.getAllocation().flush(0, vertices.size() * sizeof(DebugVertex));
}

void DebugPass::createRotateGizmoGeometry(const renderer::VulkanCore& core) {
	using namespace toast::gizmo_layout;
	constexpr glm::vec4 k_white {1.0f, 1.0f, 1.0f, 1.0f};

	std::vector<DebugVertex> vertices;

	constexpr std::array<toast::GizmoHandle, 3> axis_handles {
	  toast::GizmoHandle::axis_x, toast::GizmoHandle::axis_y, toast::GizmoHandle::axis_z
	};

	constexpr std::array<glm::vec4, 3> axis_colors {
	  glm::vec4 { 1.0f, 0.15f, 0.15f, 1.0f},
     glm::vec4 {0.15f,  1.0f, 0.15f, 1.0f},
     glm::vec4 {0.15f, 0.15f,  1.0f, 1.0f}
	};

	for (int axis = 0; axis < 3; ++axis) {
		const size_t start = vertices.size();
		appendRing(vertices, axis, k_ring_radius, k_ring_thickness, k_ring_segments, k_white);
		m_rotate_gizmo_handles[static_cast<size_t>(axis_handles[axis])] = {
		  static_cast<uint32_t>(start), static_cast<uint32_t>(vertices.size() - start), axis_colors[axis]
		};
	}

	vk::BufferCreateInfo buffer_ci {};
	buffer_ci.size = vertices.size() * sizeof(DebugVertex);
	buffer_ci.usage = vk::BufferUsageFlagBits::eVertexBuffer;

	vma::AllocationCreateInfo alloc_ci {};
	alloc_ci.usage = vma::MemoryUsage::eAuto;
	alloc_ci.flags = vma::AllocationCreateFlagBits::eMapped | vma::AllocationCreateFlagBits::eHostAccessSequentialWrite;

	m_rotate_gizmo_vertex_buffer = core.getAllocator().createBuffer(buffer_ci, alloc_ci);
	setDebugName(core, *m_rotate_gizmo_vertex_buffer, "DebugPass RotateGizmoVertexBuffer");

	void* mapped = m_rotate_gizmo_vertex_buffer.getAllocation().getInfo().pMappedData;
	std::memcpy(mapped, vertices.data(), vertices.size() * sizeof(DebugVertex));
	m_rotate_gizmo_vertex_buffer.getAllocation().flush(0, vertices.size() * sizeof(DebugVertex));
}

void DebugPass::createScaleGizmoGeometry(const renderer::VulkanCore& core) {
	using namespace toast::gizmo_layout;
	constexpr glm::vec4 k_white {1.0f, 1.0f, 1.0f, 1.0f};

	std::vector<DebugVertex> vertices;

	constexpr std::array<toast::GizmoHandle, 3> axis_handles {
	  toast::GizmoHandle::axis_x, toast::GizmoHandle::axis_y, toast::GizmoHandle::axis_z
	};

	constexpr std::array<glm::vec4, 3> axis_colors {
	  glm::vec4 { 1.0f, 0.15f, 0.15f, 1.0f},
     glm::vec4 {0.15f,  1.0f, 0.15f, 1.0f},
     glm::vec4 {0.15f, 0.15f,  1.0f, 1.0f}
	};

	for (int axis = 0; axis < 3; ++axis) {
		const size_t start = vertices.size();
		appendShaftAlongAxis(vertices, axis, k_shaft_length, k_shaft_half_size, k_white);

		// cube head, sitting right past the shaft tip - the scale-tool equivalent of translate's arrowhead
		glm::vec3 min(-k_scale_head_half_size);
		glm::vec3 max(k_scale_head_half_size);
		min[axis] = k_shaft_length;
		max[axis] = k_shaft_length + (2.0f * k_scale_head_half_size);
		appendBox(vertices, min, max, k_white);

		m_scale_gizmo_handles[static_cast<size_t>(axis_handles[axis])] = {
		  static_cast<uint32_t>(start), static_cast<uint32_t>(vertices.size() - start), axis_colors[axis]
		};
	}

	{
		const size_t start = vertices.size();
		appendBox(vertices, glm::vec3(-k_center_half_size), glm::vec3(k_center_half_size), k_white);
		m_scale_gizmo_handles[static_cast<size_t>(toast::GizmoHandle::center)] = {
		  static_cast<uint32_t>(start), static_cast<uint32_t>(vertices.size() - start), glm::vec4(0.9f, 0.9f, 0.9f, 1.0f)
		};
	}

	vk::BufferCreateInfo buffer_ci {};
	buffer_ci.size = vertices.size() * sizeof(DebugVertex);
	buffer_ci.usage = vk::BufferUsageFlagBits::eVertexBuffer;

	vma::AllocationCreateInfo alloc_ci {};
	alloc_ci.usage = vma::MemoryUsage::eAuto;
	alloc_ci.flags = vma::AllocationCreateFlagBits::eMapped | vma::AllocationCreateFlagBits::eHostAccessSequentialWrite;

	m_scale_gizmo_vertex_buffer = core.getAllocator().createBuffer(buffer_ci, alloc_ci);
	setDebugName(core, *m_scale_gizmo_vertex_buffer, "DebugPass ScaleGizmoVertexBuffer");

	void* mapped = m_scale_gizmo_vertex_buffer.getAllocation().getInfo().pMappedData;
	std::memcpy(mapped, vertices.data(), vertices.size() * sizeof(DebugVertex));
	m_scale_gizmo_vertex_buffer.getAllocation().flush(0, vertices.size() * sizeof(DebugVertex));
}

void DebugPass::ensureLineCapacity(const renderer::VulkanCore& core, DynamicVertexBuffer& buffer, size_t required_vertex_count) {
	const vk::DeviceSize required_bytes = required_vertex_count * sizeof(DebugVertex);
	if (required_bytes <= buffer.capacity_bytes) {
		return;
	}

	// Grow
	const vk::DeviceSize new_capacity = std::max<vk::DeviceSize>(required_bytes * 2, sizeof(DebugVertex) * 1024);

	vk::BufferCreateInfo buffer_ci {};
	buffer_ci.size = new_capacity;
	buffer_ci.usage = vk::BufferUsageFlagBits::eVertexBuffer;

	vma::AllocationCreateInfo alloc_ci {};
	alloc_ci.usage = vma::MemoryUsage::eAuto;
	alloc_ci.flags = vma::AllocationCreateFlagBits::eMapped | vma::AllocationCreateFlagBits::eHostAccessSequentialWrite;

	buffer.buffer = core.getAllocator().createBuffer(buffer_ci, alloc_ci);
	buffer.capacity_bytes = new_capacity;
	buffer.mapped = buffer.buffer.getAllocation().getInfo().pMappedData;
	setDebugName(core, *buffer.buffer, "DebugPass LineVertexBuffer");
}

}    // namespace renderer
