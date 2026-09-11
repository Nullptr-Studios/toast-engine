/**
 * @file traced_shadow_pass.cpp
 * @author dario
 * @date 13/08/2026
 */

#include "traced_shadow_pass.hpp"

#include "../descriptor_writer.hpp"
#include "../ray_tracing_scene.hpp"
#include "../shader_cache.hpp"
#include "../vulkan_core.hpp"
#include "../vulkan_debug.hpp"
#include "../vulkan_renderer.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <toast/assets/assets.hpp>
#include <toast/log.hpp>

namespace renderer {

namespace {

/// @brief How far a shadow ray travels before it is treated as unoccluded
///
/// A hard cap rather than infinity: a ray that leaves the scene entirely still costs traversal, and beyond
/// this distance a directional light's occluders are no longer what the eye is looking at
constexpr float k_max_trace_distance = 500.0f;

/// @brief Offset along the surface normal before tracing, in world units
///
/// The only bias a traced shadow needs. It exists to escape the surface's own triangle, not to compensate for
/// depth quantisation - which is why it is one small fixed distance rather than the slope-scaled,
/// cascade-scaled pair the shadow-map path has to carry
constexpr float k_normal_bias = 0.02f;

/// @brief Must match `RenderMode` in editor/Workspace/WorkspaceViewModel.cs
///
/// The enum is ordinal and the shader compares against raw numbers, so inserting a mode anywhere above these
/// silently shifts them - which shows up as the wrong view being drawn, or as nothing happening at all,
/// rather than as any kind of error. Kept here and passed to the shader instead of being written twice
constexpr uint32_t k_mode_visibility = 17;
constexpr uint32_t k_mode_applied = 18;

}

TracedShadowPass::TracedShadowPass(const VulkanCore& core, vk::Format scene_format, vk::Extent2D extent)
    : m_core(&core),
      m_scene_format(scene_format) {
	const auto uid = assets::resolveURI("core://shaders/traced_shadows.slang");
	const auto shader = uid.has_value() ? ShaderCache::get().acquire(*uid) : nullptr;
	if (!shader) {
		TOAST_ERROR("Render", "TracedShadowPass shader unavailable; traced shadows are disabled");
		setEnabled(false);
		return;
	}

	m_shader_layout.rebuild(core, shader->reflection, "TracedShadowPass");

	VulkanPipeline::Config config;
	config.pipeline_type = VulkanPipeline::PipelineType::graphics;
	config.debug_name = "TracedShadowPass";
	config.color_format = scene_format;
	config.extent = extent;
	config.shader_spirv = shader->spirv;
	config.pipeline_layout = *m_shader_layout.getPipelineLayout();
	config.vertex_bindings = {};
	config.vertex_attributes = {};
	config.cull_mode = vk::CullModeFlagBits::eNone;
	config.depth_test = false;
	config.depth_write = false;
	m_pipeline.rebuild(core, config);

	createResources(core);
	createTarget(core, extent);
	TOAST_INFO("Render", "TracedShadowPass ready");
}

void TracedShadowPass::createResources(const VulkanCore& core) {
	const auto& device = core.getDevice();
	const auto& layouts = m_shader_layout.getDescriptorSetLayouts();
	if (layouts.empty()) {
		return;
	}

	const auto sampler_ci = linearClampSamplerInfo();
	m_sampler = vk::raii::Sampler(device, sampler_ci);

	// Depth and the geometry buffer point-sampled, same as SsrPass: a filtered depth across a silhouette is a
	// value describing no geometry at all, and tracing from it starts the ray in mid-air
	const auto point_ci = nearestClampSamplerInfo();
	m_point_sampler = vk::raii::Sampler(device, point_ci);

	const vk::DescriptorSetLayout set_layout = *layouts[0];
	const vk::DescriptorPool pool = VulkanRenderer::instance->getDescriptorPoolHandle();

	m_descriptor_sets.clear();
	m_bound_views.assign(VulkanRenderer::k_frames_in_flight, vk::ImageView {});
	m_bound_acceleration_structures.assign(VulkanRenderer::k_frames_in_flight, vk::AccelerationStructureKHR {});
	for (uint32_t i = 0; i < VulkanRenderer::k_frames_in_flight; ++i) {
		const vk::DescriptorSetAllocateInfo alloc_info(pool, 1, &set_layout);
		auto allocated = device.allocateDescriptorSets(alloc_info);
		m_descriptor_sets.push_back(std::move(allocated[0]));
		setDebugName(core, *m_descriptor_sets[i], std::format("TracedShadowPass DescriptorSet[{}]", i));
	}
}

void TracedShadowPass::createTarget(const VulkanCore& core, vk::Extent2D extent) {
	m_target.create(core, extent, m_scene_format, "TracedShadowPass");
	std::ranges::fill(m_bound_views, vk::ImageView {});
}

void TracedShadowPass::onResize(vk::Extent2D extent) {
	if (m_core != nullptr) {
		createTarget(*m_core, extent);
	}
}

auto TracedShadowPass::record(vk::CommandBuffer cmd, uint32_t frame_index, vk::ImageView source_view) -> vk::ImageView {
	if (!m_pipeline.isReady() || frame_index >= m_descriptor_sets.size() || !source_view || !m_target.isReady()) {
		return source_view;
	}

	const auto* frame = VulkanRenderer::instance->renderingFrame();
	if (frame == nullptr) {
		return source_view;
	}

	// Only the two debug modes trace. Everything else skips the pass entirely rather than running a
	// full-screen read that returns its input unchanged
	const uint32_t render_mode = frame->render_mode;
	if (render_mode != k_mode_visibility && render_mode != k_mode_applied) {
		return source_view;
	}

	auto* scene = VulkanRenderer::instance->getRayTracingScene();
	const vk::AccelerationStructureKHR tlas = scene != nullptr ? scene->getAccelerationStructure(frame_index) : nullptr;
	const vk::ImageView depth_view = VulkanRenderer::instance->getDepthView();
	const vk::ImageView normal_view = VulkanRenderer::instance->getSceneNormalView();

	// No TLAS this frame means nothing traceable was submitted - passing the image through leaves the scene
	// as it was rather than painting it black and looking like a broken trace
	if (!tlas || !depth_view || !normal_view) {
		return source_view;
	}

	if (m_bound_views[frame_index] != source_view || m_bound_acceleration_structures[frame_index] != tlas) {
		constexpr auto k_sampler = vk::DescriptorType::eCombinedImageSampler;
		const vk::DescriptorSet set = *m_descriptor_sets[frame_index];

		// The acceleration structure goes in through a pNext struct rather than pImageInfo or pBufferInfo -
		// the descriptor has no other way to name it, which is why DescriptorWriter has a method for it
		DescriptorWriter writer;
		writer.image(set, 0, k_sampler, *m_sampler, source_view, vk::ImageLayout::eShaderReadOnlyOptimal)
		    .image(set, 1, k_sampler, *m_point_sampler, depth_view, vk::ImageLayout::eDepthReadOnlyOptimal)
		    .image(set, 2, k_sampler, *m_point_sampler, normal_view, vk::ImageLayout::eShaderReadOnlyOptimal)
		    .accelerationStructure(set, 3, tlas);
		writer.flush(m_core->getDevice());
		m_bound_views[frame_index] = source_view;
		m_bound_acceleration_structures[frame_index] = tlas;
	}

	m_target.beginScope(cmd);

	// The shadow-casting directional light, picked the same way ShadowPass picks it so the traced result can
	// be compared against the rastered one rather than against a different light
	glm::vec4 light_direction(0.0f, -1.0f, 0.0f, 0.0f);
	if (frame->frame_data.directional_light_count_pad.x > 0) {
		light_direction = frame->frame_data.directional_lights[0].direction;
	}

	Params params {};
	params.inverse_view_projection = glm::inverse(frame->frame_data.view_projection);
	params.light_direction = light_direction;
	// Normalized to 0 = visibility, 1 = applied, so the shader never has to know the editor's enum ordinals
	const float mode_flag = render_mode == k_mode_applied ? 1.0f : 0.0f;
	params.tuning = glm::vec4(mode_flag, k_normal_bias, k_max_trace_distance, 0.0f);

	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline.getPipeline());
	cmd.bindDescriptorSets(
	    vk::PipelineBindPoint::eGraphics,
	    *m_shader_layout.getPipelineLayout(),
	    0,
	    std::array<vk::DescriptorSet, 1> {*m_descriptor_sets[frame_index]},
	    {}
	);
	cmd.pushConstants(
	    *m_shader_layout.getPipelineLayout(),
	    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
	    0,
	    sizeof(Params),
	    &params
	);
	cmd.draw(3, 1, 0, 0);
	m_target.endScope(cmd);

	return m_target.view();
}

}
