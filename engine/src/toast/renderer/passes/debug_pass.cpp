/// @file debug_pass.cpp
/// @author dario
/// @date 10/06/2026

#include "debug_pass.hpp"

#include "../clustered_lighting_constants.hpp"
#include "../ray_tracing_scene.hpp"
#include "../shader_cache.hpp"
#include "../skinned_blas_pool.hpp"
#include "../vulkan_core.hpp"
#include "../vulkan_debug.hpp"
#include "../vulkan_mesh.hpp"
#include "../vulkan_renderer.hpp"
#include "../vulkan_texture.hpp"
#include "cluster_lighting_pass.hpp"
#include "environment_pass.hpp"
#include "shadow_pass.hpp"
#include "skinning_pass.hpp"

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
#include <toast/assets/assets.hpp>
#include <toast/log.hpp>

namespace renderer {

namespace {

constexpr std::array<toast::GizmoHandle, 3> k_axis_handles {
  toast::GizmoHandle::axis_x, toast::GizmoHandle::axis_y, toast::GizmoHandle::axis_z
};

constexpr std::array<glm::vec4, 3> k_axis_colors {
  glm::vec4 { 1.0f, 0.15f, 0.15f, 1.0f}, // X
  glm::vec4 {0.15f,  1.0f, 0.15f, 1.0f}, // Y
  glm::vec4 {0.15f, 0.15f,  1.0f, 1.0f}  // Z
};

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

namespace {

auto acquireShader(std::string_view uri) -> std::shared_ptr<const ShaderCache::Entry> {
	const auto uid = assets::resolveURI(uri);
	if (!uid.has_value()) {
		TOAST_ERROR("Render", "DebugPass shader not found in the asset manifest: {}", uri);
		return nullptr;
	}
	return ShaderCache::get().acquire(*uid);
}

}

DebugPass::DebugPass(
    const renderer::VulkanCore& core, vk::Format color_format, vk::Format depth_format, vk::Extent2D extent,
    const ClusterLightingPass* cluster_lighting_pass
)
    : m_cluster_lighting_pass(cluster_lighting_pass) {
	const auto shape_shader = acquireShader("core://shaders/debug_shape.slang");
	if (!shape_shader) {
		TOAST_ERROR("Render", "DebugPass has no usable shaders, the pass will draw nothing");
		return;
	}
	m_shader_layout.rebuild(core, shape_shader->reflection, "DebugPass");

	const vk::VertexInputBindingDescription debug_vertex_binding(0, sizeof(DebugVertex), vk::VertexInputRate::eVertex);
	const std::vector<vk::VertexInputAttributeDescription> debug_vertex_attributes {
	  vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(DebugVertex, position)),
	  vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(DebugVertex, color)),
	};

	// Debug lines
	if (shape_shader) {
		VulkanPipeline::Config config;
		config.pipeline_type = VulkanPipeline::PipelineType::graphics;
		config.debug_name = "DebugPass Lines";
		config.color_format = color_format;
		config.depth_format = depth_format;
		config.extent = extent;
		config.shader_spirv = shape_shader->spirv;
		config.pipeline_layout = *m_shader_layout.getPipelineLayout();
		config.vertex_bindings = {debug_vertex_binding};
		config.vertex_attributes = debug_vertex_attributes;
		config.topology = vk::PrimitiveTopology::eLineList;
		config.cull_mode = vk::CullModeFlagBits::eNone;
		config.depth_test = true;
		config.depth_write = false;
		config.blend_enable = false;
		m_line_pipeline.rebuild(core, config);
	}

	// Gizmo axis triad
	if (shape_shader) {
		VulkanPipeline::Config config;
		config.pipeline_type = VulkanPipeline::PipelineType::graphics;
		config.debug_name = "DebugPass Gizmo";
		config.color_format = color_format;
		config.depth_format = depth_format;
		config.extent = extent;
		config.shader_spirv = shape_shader->spirv;
		config.pipeline_layout = *m_shader_layout.getPipelineLayout();
		config.vertex_bindings = {debug_vertex_binding};
		config.vertex_attributes = debug_vertex_attributes;
		config.topology = vk::PrimitiveTopology::eTriangleList;
		config.cull_mode = vk::CullModeFlagBits::eNone;
		config.depth_test = false;
		config.depth_write = false;
		config.blend_enable = false;
		m_gizmo_pipeline.rebuild(core, config);
	}

	// Editor furniture meshes
	if (shape_shader) {
		VulkanPipeline::Config config;
		config.pipeline_type = VulkanPipeline::PipelineType::graphics;
		config.debug_name = "DebugPass Mesh";
		config.color_format = color_format;
		config.depth_format = depth_format;
		config.extent = extent;
		config.shader_spirv = shape_shader->spirv;
		config.pipeline_layout = *m_shader_layout.getPipelineLayout();
		config.vertex_entry = "vertexMesh";
		config.fragment_entry = "fragmentMesh";
		config.vertex_bindings = {vertexBindingDescription()};
		// Position, normal and colour out of the full renderer::Vertex stream
		const auto mesh_attributes = vertexAttributeDescriptions();
		config.vertex_attributes = {mesh_attributes[0], mesh_attributes[1], mesh_attributes[4]};
		config.topology = vk::PrimitiveTopology::eTriangleList;
		// Backface culled and depth tested, unlike the flat gizmos
		config.cull_mode = vk::CullModeFlagBits::eBack;
		config.depth_test = true;
		config.depth_write = true;
		config.blend_enable = false;
		m_mesh_pipeline.rebuild(core, config);
	}

	createResources(core);
	createBillboardResources(core, color_format, depth_format, extent);
	initImGui(core, color_format, depth_format);
}

DebugPass::~DebugPass() {
	if (m_imgui_ready) {
		ImGui_ImplVulkan_Shutdown();
		ImGui::DestroyContext();
	}
}

namespace {

/// @brief Maps the SDL-keycode encoding used by event::WindowKey
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

/// @brief Cluster light-count, color ramp for the ImGui cluster-grid overlay
auto clusterHeatmapColorImGui(uint32_t light_count) -> ImVec4 {
	constexpr float k_heatmap_max_lights = 8.0f;
	const float t = std::clamp(static_cast<float>(light_count) / k_heatmap_max_lights, 0.0f, 1.0f);

	const ImVec4 c0(0.0f, 0.0f, 1.0f, 1.0f);
	const ImVec4 c1(0.0f, 1.0f, 0.0f, 1.0f);
	const ImVec4 c2(1.0f, 1.0f, 0.0f, 1.0f);
	const ImVec4 c3(1.0f, 0.0f, 0.0f, 1.0f);

	auto lerp = [](const ImVec4& a, const ImVec4& b, float u) {
		return ImVec4(a.x + ((b.x - a.x) * u), a.y + ((b.y - a.y) * u), a.z + ((b.z - a.z) * u), 1.0f);
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
			ImGui::Text(
			    "Mesh instances: %u/%zu drawn", VulkanRenderer::instance->getVisibleInstanceCount(), frame->mesh_instances.size()
			);
			ImGui::Text("Lights: %zu", frame->lights.size());

			{
				const uint32_t dropped = VulkanRenderer::instance->getDroppedFrameCount();
				const uint32_t out_of_order = VulkanRenderer::instance->getOutOfOrderFrameCount();
				if (out_of_order > 0) {
					ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Frames out of order: %u", out_of_order);
				}

				ImGui::TextDisabled("Frames dropped: %u", dropped);

				const uint32_t skipped = VulkanRenderer::instance->getSkippedBuildCount();
				if (skipped > 0) {
					ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f), "Ticks with no frame built: %u", skipped);
				} else {
					ImGui::TextDisabled("Ticks with no frame built: 0");
				}

				// The spread, not the average
				float avg_ms = 0.0f;
				float min_ms = 0.0f;
				float max_ms = 0.0f;
				VulkanRenderer::instance->getFrameTimeStats(avg_ms, min_ms, max_ms);
				const float spread = max_ms - min_ms;
				if (spread > 8.0f) {
					ImGui::TextColored(
					    ImVec4(1.0f, 0.7f, 0.2f, 1.0f), "Frame time: %.1f ms (%.1f-%.1f, spread %.1f)", avg_ms, min_ms, max_ms, spread
					);
				} else {
					ImGui::TextDisabled("Frame time: %.1f ms (%.1f-%.1f)", avg_ms, min_ms, max_ms);
				}

				// CAP
				float cap = static_cast<float>(VulkanRenderer::instance->frameRateLimit());
				if (ImGui::SliderFloat("FPS cap", &cap, 0.0f, 144.0f, cap <= 0.0f ? "uncapped" : "%.0f")) {
					VulkanRenderer::instance->setFrameRateLimit(static_cast<double>(cap));
				}
			}

			ImGui::TextDisabled("Depth prepass: %u instances", VulkanRenderer::instance->getPrepassDrawnCount());

			if (const auto* shadows = VulkanRenderer::instance->getShadowPass(); shadows != nullptr) {
				ImGui::TextDisabled("Shadow pass: %u draws, %u scopes", shadows->getDrawCount(), shadows->getPassCount());
			}

			if (const auto* skinning = VulkanRenderer::instance->getSkinningPass(); skinning != nullptr) {
				const auto* pool = VulkanRenderer::instance->getSkinnedBlasPool();
				ImGui::TextDisabled(
				    "Skinning: %u posed, %u skinned BLAS refit",
				    skinning->getPosedInstanceCount(),
				    pool != nullptr ? pool->getRecordedCount() : 0u
				);
			}

			// Says whether there is anything to trace at all
			if (auto* rt = VulkanRenderer::instance->getRayTracingScene(); rt != nullptr) {
				const uint32_t traced = rt->getInstanceCount();
				if (traced == 0) {
					ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f), "TLAS: empty - nothing to trace against");
				} else {
					ImGui::TextDisabled("TLAS: %u instances", traced);
				}

				bool trace_all_lights = VulkanRenderer::instance->tracedShadowsEnabled();
				if (ImGui::Checkbox("Traced shadows (all lights)", &trace_all_lights)) {
					VulkanRenderer::instance->setTracedShadowsEnabled(trace_all_lights);
				}
			} else {
				ImGui::TextDisabled("TLAS: ray query unsupported");
			}

			{
				bool cull_debug = VulkanRenderer::instance->getCullDebugDraw();
				if (ImGui::Checkbox("Show culling volumes", &cull_debug)) {
					VulkanRenderer::instance->setCullDebugDraw(cull_debug);
				}

				bool cull_freeze = VulkanRenderer::instance->getCullFreeze();
				if (ImGui::Checkbox("Freeze cull frustum", &cull_freeze)) {
					VulkanRenderer::instance->setCullFreeze(cull_freeze);
				}
			}

			ImGui::Separator();

			// The generated sky is the scene's only light until something else is added
			if (auto* environment = VulkanRenderer::instance->getEnvironmentPassMutable(); environment != nullptr) {
				if (m_sky_intensity_ui < 0.0f) {
					m_sky_intensity_ui = environment->getSkyIntensity();
				}
				// A URI rather than a file picker
				ImGui::InputText("Environment HDR", m_environment_uri_ui.data(), m_environment_uri_ui.size());
				ImGui::SameLine();
				if (ImGui::Button("Load")) {
					environment->setEnvironmentMap(std::string_view(m_environment_uri_ui.data()));
				}
				if (!environment->getEnvironmentMapUri().empty()) {
					ImGui::SameLine();
					if (ImGui::Button("Clear")) {
						environment->setEnvironmentMap("");
					}
				}

				ImGui::SliderFloat("Sky intensity", &m_sky_intensity_ui, 0.0f, 20.0f, "%.2f");

				if (ImGui::IsItemDeactivatedAfterEdit()) {
					environment->setSkyIntensity(m_sky_intensity_ui);
				}
			}

			// What the shader actually received, not what the node holds. A probe that contributes nothing is
			// indistinguishable from no probe from the viewport
			{
				const uint32_t probe_count = frame->frame_data.reflection_probe_count_pad.x;
				ImGui::Text("Reflection probes: %u", probe_count);
				for (uint32_t i = 0; i < probe_count && i < 4; ++i) {
					const auto& probe = frame->frame_data.reflection_probes[i];
					const glm::vec3 position(probe.position_radius);
					const glm::vec3 extents(probe.box_extents_intensity);
					if (extents.x > 0.0f || extents.y > 0.0f || extents.z > 0.0f) {
						ImGui::BulletText(
						    "cube %d box +/-(%.0f, %.0f, %.0f) at (%.0f, %.0f, %.0f)",
						    static_cast<int>(probe.params.x),
						    extents.x,
						    extents.y,
						    extents.z,
						    position.x,
						    position.y,
						    position.z
						);
					} else {
						ImGui::BulletText(
						    "cube %d sphere r=%.0f at (%.0f, %.0f, %.0f)",
						    static_cast<int>(probe.params.x),
						    probe.position_radius.w,
						    position.x,
						    position.y,
						    position.z
						);
					}
				}
				ImGui::Text(
				    "Camera: (%.0f, %.0f, %.0f)",
				    frame->frame_data.camera_position.x,
				    frame->frame_data.camera_position.y,
				    frame->frame_data.camera_position.z
				);
			}

			if (ImGui::Button("Bake reflection probes")) {
				VulkanRenderer::instance->requestReflectionProbeBake();
			}
			ImGui::SameLine();
			if (const uint32_t stale = VulkanRenderer::instance->getStaleReflectionProbeCount(); stale > 0) {
				ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f), "%u stale", stale);
			} else {
				ImGui::TextDisabled("(6 frames per probe)");
			}

			// Read-only
			if (ImGui::CollapsingHeader("Screen-space reflections")) {
				const auto& ssr = frame->post_process.ssr;
				ImGui::TextDisabled("intensity %.2f, max roughness %.2f", ssr.intensity, ssr.max_roughness);
				ImGui::TextDisabled("stride %.3f m x %u steps", ssr.stride, ssr.max_steps);
				ImGui::TextDisabled("thickness %.2f m", ssr.thickness);
				ImGui::TextDisabled("Ray reach: %.1f m", ssr.stride * static_cast<float>(ssr.max_steps));
			}

			if (const uint32_t irradiance_probes = VulkanRenderer::instance->getIrradianceProbeCount(); irradiance_probes > 0) {
				if (ImGui::Button("Bake irradiance volumes")) {
					VulkanRenderer::instance->requestIrradianceBake();
				}
				ImGui::SameLine();
				if (const uint32_t stale = VulkanRenderer::instance->getStaleIrradianceVolumeCount(); stale > 0) {
					ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f), "%u stale (%u probes)", stale, irradiance_probes);
				} else {
					ImGui::TextDisabled("(%u probes, %u frames)", irradiance_probes, irradiance_probes * 6);
				}
			}

			if (ImGui::CollapsingHeader("Ambient occlusion")) {
				const auto& ssao = frame->post_process.ssao;
				ImGui::TextDisabled("radius %.2f m, strength %.2f", ssao.radius, ssao.strength);
				ImGui::TextDisabled("range cutoff %.2f m, %u samples", ssao.range_cutoff, ssao.sample_count);
			}

			ImGui::Separator();
			ImGui::Text("Render mode: %s", frame->render_mode == 1 ? "Cluster Heatmap (toolbar Mode button)" : "Lit");
			if (frame->render_mode == 1) {
				ImGui::TextColored(ImVec4(0.3f, 0.6f, 1.0f, 1.0f), "Blue = 0 lights/cluster");
				ImGui::TextColored(ImVec4(1.0f, 0.2f, 0.2f, 1.0f), "Red = 8+ lights/cluster");
			}
		}
		ImGui::End();

		// Cluster-heatmap overlay
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
							const uint32_t idx = tx + (ty * k_cluster_dim_x) + (tz * k_cluster_dim_x * k_cluster_dim_y);
							max_count = std::max(max_count, counts[idx]);
						}

						const ImVec2 cell_min(static_cast<float>(tx) * cell_w, static_cast<float>(ty) * cell_h);
						const ImVec2 cell_max(cell_min.x + cell_w, cell_min.y + cell_h);

						const ImVec4 heat = clusterHeatmapColorImGui(max_count);
						draw_list->AddRectFilled(cell_min, cell_max, ImGui::ColorConvertFloat4ToU32(ImVec4(heat.x, heat.y, heat.z, 0.30f)));
						draw_list->AddRect(cell_min, cell_max, IM_COL32(255, 255, 255, 50));

						const std::string label = std::format("{}", max_count);
						const ImVec2 text_size = ImGui::CalcTextSize(label.c_str());
						const ImVec2 text_pos(cell_min.x + ((cell_w - text_size.x) * 0.5f), cell_min.y + ((cell_h - text_size.y) * 0.5f));
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
		TOAST_ERROR("Render", "Frame index {} out of bounds for descriptor sets", frame_index);
		return;
	}

	const auto* frame = VulkanRenderer::instance->renderingFrame();
	if (frame == nullptr) {
		return;
	}

	// Nothing here belongs in a reflection probe
	if (frame->probe_capture_index >= 0) {
		return;
	}

	cmd.bindDescriptorSets(
	    vk::PipelineBindPoint::eGraphics,
	    *m_shader_layout.getPipelineLayout(),
	    0,
	    std::array<vk::DescriptorSet, 1> {*m_frame_descriptor_sets[frame_index]},
	    {}
	);

	// Debug lines
	const uint32_t line_vertex_count = frame_index < m_line_vertex_counts.size() ? m_line_vertex_counts[frame_index] : 0;
	if (line_vertex_count > 0 && m_line_pipeline.isReady()) {
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, m_line_pipeline.getPipeline());

		DrawPushConstants pc {};
		pc.model = glm::mat4(1.0f);
		cmd.pushConstants(
		    *m_shader_layout.getPipelineLayout(),
		    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
		    0,
		    sizeof(DrawPushConstants),
		    &pc
		);

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
			    *m_shader_layout.getPipelineLayout(),
			    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
			    0,
			    sizeof(DrawPushConstants),
			    &pc
			);
			cmd.draw(m_gizmo_vertex_count, 1, 0, 0);
		}
	}

	// Editor meshes
	if (!frame->debug_meshes.empty() && m_mesh_pipeline.isReady()) {
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, m_mesh_pipeline.getPipeline());

		for (const auto& entry : frame->debug_meshes) {
			if (!entry.mesh.hasValue()) {
				continue;
			}

			const auto& gpu_mesh = entry.mesh->gpuMesh();
			if (!gpu_mesh.isReady() || gpu_mesh.getIndexCount() == 0) {
				continue;
			}

			DrawPushConstants pc {};
			pc.model = entry.model;
			pc.tint = entry.tint;
			cmd.pushConstants(
			    *m_shader_layout.getPipelineLayout(),
			    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
			    0,
			    sizeof(DrawPushConstants),
			    &pc
			);

			gpu_mesh.bind(cmd);
			cmd.drawIndexed(gpu_mesh.getIndexCount(), 1, 0, 0, 0);
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

				// Scale feedback
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
				// Stage flags must cover every stage the layout
				cmd.pushConstants(
				    *m_shader_layout.getPipelineLayout(),
				    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
				    0,
				    sizeof(DrawPushConstants),
				    &pc
				);
				cmd.draw(range.vertex_count, 1, range.first_vertex, 0);
			}
		}
	}

	// Camera-facing textured icons
	if (!frame->debug_billboards.empty() && m_billboard_pipeline.isReady() && frame_index < m_billboard_frame_sets.size()) {
		const auto& core = VulkanRenderer::instance->getCore();
		const vk::PipelineLayout layout = *m_billboard_layout.getPipelineLayout();

		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, m_billboard_pipeline.getPipeline());
		cmd.bindDescriptorSets(
		    vk::PipelineBindPoint::eGraphics, layout, 0, std::array<vk::DescriptorSet, 1> {*m_billboard_frame_sets[frame_index]}, {}
		);

		vk::DescriptorSet bound_texture_set {};
		for (const auto& billboard : frame->debug_billboards) {
			if (!billboard.texture.hasValue()) {
				continue;
			}
			const auto& gpu_texture = billboard.texture->gpuTexture();
			if (!gpu_texture.isReady() || !gpu_texture.getView()) {
				continue;    // still uploading
			}

			const vk::DescriptorSet texture_set = billboardTextureSet(core, gpu_texture.getView());
			if (!texture_set) {
				continue;
			}
			if (texture_set != bound_texture_set) {
				cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 1, std::array {texture_set}, {});
				bound_texture_set = texture_set;
			}

			BillboardPushConstants pc {};
			pc.center_size = glm::vec4(billboard.position, billboard.size);
			pc.tint = billboard.tint;
			cmd.pushConstants(
			    layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(BillboardPushConstants), &pc
			);

			// Six vertices
			cmd.draw(6, 1, 0, 0);
		}
	}

	// ImGui draws last, always on top
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
		TOAST_CRITICAL("Render", "ShaderLayout has no descriptor set layouts");
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
			TOAST_CRITICAL("Render", "Frame UBO buffer missing for frame {}", i);
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

	createGizmoGeometry(core);

	// Gizmos
	createTranslateGizmoGeometry(core);
	createRotateGizmoGeometry(core);
	createScaleGizmoGeometry(core);
}

void DebugPass::createBillboardResources(
    const renderer::VulkanCore& core, vk::Format color_format, vk::Format depth_format, vk::Extent2D extent
) {
	const auto shader = acquireShader("core://shaders/debug_billboard.slang");
	if (!shader) {
		TOAST_ERROR("Render", "DebugPass billboard shader unavailable, debug billboards will not draw");
		return;
	}

	m_billboard_layout.rebuild(core, shader->reflection, "DebugPass Billboard");

	VulkanPipeline::Config config;
	config.pipeline_type = VulkanPipeline::PipelineType::graphics;
	config.debug_name = "DebugPass Billboard";
	config.color_format = color_format;
	config.depth_format = depth_format;
	config.extent = extent;
	config.shader_spirv = shader->spirv;
	config.pipeline_layout = *m_billboard_layout.getPipelineLayout();
	// The shader generates its own quad from SV_VertexIDt
	config.topology = vk::PrimitiveTopology::eTriangleList;
	config.cull_mode = vk::CullModeFlagBits::eNone;
	// Depth-tested so icons sit correctly in the scene and get hidden behind geometry
	config.depth_test = true;
	config.depth_write = false;
	config.blend_preset = VulkanPipeline::BlendPreset::alpha;
	m_billboard_pipeline.rebuild(core, config);

	const auto& device = core.getDevice();

	const auto sampler_ci = linearClampMippedSamplerInfo(VK_LOD_CLAMP_NONE);
	m_billboard_sampler = vk::raii::Sampler(device, sampler_ci);
	setDebugName(core, *m_billboard_sampler, "DebugPass BillboardSampler");

	const auto& layouts = m_billboard_layout.getDescriptorSetLayouts();
	if (layouts.empty()) {
		TOAST_ERROR("Render", "DebugPass billboard layout has no descriptor sets");
		return;
	}

	const vk::DescriptorPool pool = VulkanRenderer::instance->getDescriptorPoolHandle();
	const vk::DescriptorSetLayout frame_set_layout = *layouts[0];

	m_billboard_frame_sets.clear();
	m_billboard_frame_sets.reserve(VulkanRenderer::k_frames_in_flight);
	for (uint32_t i = 0; i < VulkanRenderer::k_frames_in_flight; ++i) {
		const vk::DescriptorSetAllocateInfo alloc_info(pool, 1, &frame_set_layout);
		auto allocated = device.allocateDescriptorSets(alloc_info);
		m_billboard_frame_sets.push_back(std::move(allocated[0]));
		setDebugName(core, *m_billboard_frame_sets[i], std::format("DebugPass BillboardFrameSet[{}]", i));

		const auto* frame_res = VulkanRenderer::instance->getFrameUBORes(i);
		if (!frame_res->gpu_buffer.has_value()) {
			continue;
		}
		const vk::DescriptorBufferInfo buffer_info(**frame_res->gpu_buffer, 0, sizeof(VulkanRenderer::FrameUBO));
		const vk::WriteDescriptorSet write(
		    *m_billboard_frame_sets[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &buffer_info
		);
		device.updateDescriptorSets(write, {});
	}
}

auto DebugPass::billboardTextureSet(const renderer::VulkanCore& core, vk::ImageView view) -> vk::DescriptorSet {
	if (const auto it = m_billboard_texture_sets.find(view); it != m_billboard_texture_sets.end()) {
		return *it->second;
	}

	const auto& layouts = m_billboard_layout.getDescriptorSetLayouts();
	if (layouts.size() < 2) {
		return nullptr;
	}

	const auto& device = core.getDevice();
	const vk::DescriptorSetLayout texture_set_layout = *layouts[1];
	const vk::DescriptorSetAllocateInfo alloc_info(VulkanRenderer::instance->getDescriptorPoolHandle(), 1, &texture_set_layout);
	auto allocated = device.allocateDescriptorSets(alloc_info);

	vk::DescriptorImageInfo image_info {};
	image_info.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
	image_info.imageView = view;
	image_info.sampler = *m_billboard_sampler;

	const vk::WriteDescriptorSet write(*allocated[0], 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &image_info);
	device.updateDescriptorSets(write, {});

	auto [it, _] = m_billboard_texture_sets.emplace(view, std::move(allocated[0]));
	return *it->second;
}

void DebugPass::initImGui(const renderer::VulkanCore& core, vk::Format color_format, vk::Format depth_format) {
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.BackendPlatformName = "toast_engine (manual input feed)";
	io.BackendRendererName = "imgui_impl_vulkan";

	// FIXME: No OS cursor/clipboard integration yet
	io.ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;
	io.SetClipboardTextFn = nullptr;
	io.GetClipboardTextFn = nullptr;

	const std::array<vk::Format, 1> color_formats {color_format};
	vk::PipelineRenderingCreateInfo rendering_ci {};
	rendering_ci.colorAttachmentCount = 1;
	rendering_ci.pColorAttachmentFormats = color_formats.data();
	// Must match the depth attachment of the scope this pass shares with the material draws
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

	for (int axis = 0; axis < 3; ++axis) {
		const size_t start = vertices.size();
		appendShaftAlongAxis(vertices, axis, k_shaft_length, k_shaft_half_size, k_white);
		appendPyramidAlongAxis(vertices, axis, k_shaft_length, k_shaft_length + k_head_length, k_head_half_size, k_white);
		record_handle(k_axis_handles[axis], start, k_axis_colors[axis]);
	}

	// Plane handles XY/YZ/XZ, offset from the origin along their own two axes
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

	for (int axis = 0; axis < 3; ++axis) {
		const size_t start = vertices.size();
		appendRing(vertices, axis, k_ring_radius, k_ring_thickness, k_ring_segments, k_white);
		m_rotate_gizmo_handles[static_cast<size_t>(k_axis_handles[axis])] = {
		  static_cast<uint32_t>(start), static_cast<uint32_t>(vertices.size() - start), k_axis_colors[axis]
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

	for (int axis = 0; axis < 3; ++axis) {
		const size_t start = vertices.size();
		appendShaftAlongAxis(vertices, axis, k_shaft_length, k_shaft_half_size, k_white);

		// cube head
		glm::vec3 min(-k_scale_head_half_size);
		glm::vec3 max(k_scale_head_half_size);
		min[axis] = k_shaft_length;
		max[axis] = k_shaft_length + (2.0f * k_scale_head_half_size);
		appendBox(vertices, min, max, k_white);

		m_scale_gizmo_handles[static_cast<size_t>(k_axis_handles[axis])] = {
		  static_cast<uint32_t>(start), static_cast<uint32_t>(vertices.size() - start), k_axis_colors[axis]
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

}
