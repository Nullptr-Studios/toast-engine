/**
 * @file ssr_pass.cpp
 * @author dario
 * @date 07/08/2026
 */

#include "ssr_pass.hpp"

#include "../shader_cache.hpp"
#include "../vulkan_core.hpp"
#include "../vulkan_debug.hpp"
#include "../vulkan_renderer.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <glm/gtc/matrix_inverse.hpp>
#include <toast/assets/assets.hpp>
#include <toast/log.hpp>

namespace renderer {

SsrPass::SsrPass(const VulkanCore& core, vk::Format scene_format, vk::Extent2D extent)
    : m_core(&core),
      m_scene_format(scene_format) {
	// Checked rather than assumed: 128 bytes is all Vulkan guarantees, and this block is 176. Desktop parts
	// report 256 and are fine, but a device that cannot take it would otherwise fail inside pipeline creation
	// with an error about a limit rather than about reflections
	const uint32_t push_constant_limit = core.getPhysicalDevice().getProperties().limits.maxPushConstantsSize;
	if (push_constant_limit < sizeof(Params)) {
		TOAST_ERROR(
		    "Render",
		    "SsrPass needs {} bytes of push constants but this device allows {}; screen-space reflections are disabled",
		    sizeof(Params),
		    push_constant_limit
		);
		setEnabled(false);
		return;
	}

	const auto uid = assets::resolveURI("core://shaders/ssr.slang");
	const auto shader = uid.has_value() ? ShaderCache::get().acquire(*uid) : nullptr;
	if (!shader) {
		TOAST_ERROR("Render", "SsrPass shader core://shaders/ssr.slang unavailable; screen-space reflections are disabled");
		setEnabled(false);
		return;
	}

	m_shader_layout.rebuild(core, shader->reflection, "SsrPass");

	VulkanPipeline::Config config;
	config.pipeline_type = VulkanPipeline::PipelineType::graphics;
	config.debug_name = "SsrPass";
	// The scene's HDR format, not the display format: this composites radiance and runs before the tonemap
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
	TOAST_INFO("Render", "SsrPass ready ({})", vk::to_string(scene_format));
}

void SsrPass::createResources(const VulkanCore& core) {
	const auto& device = core.getDevice();
	const auto& layouts = m_shader_layout.getDescriptorSetLayouts();
	if (layouts.empty()) {
		TOAST_ERROR("Render", "SsrPass shader layout has no descriptor sets");
		return;
	}

	const auto sampler_ci = linearClampSamplerInfo();
	m_sampler = vk::raii::Sampler(device, sampler_ci);
	setDebugName(core, *m_sampler, "SsrPass Sampler");

	// Depth and the geometry buffer are sampled with nearest filtering deliberately. Interpolating between
	// two depths produces a value describing neither surface - across a silhouette it is a halfway depth
	// where there is no geometry at all - and a filtered normal between two facings points somewhere the
	// surface never faced. Both would put reflection hits on geometry that does not exist
	const auto point_ci = nearestClampSamplerInfo();
	m_point_sampler = vk::raii::Sampler(device, point_ci);
	setDebugName(core, *m_point_sampler, "SsrPass PointSampler");

	const vk::DescriptorSetLayout set_layout = *layouts[0];
	const vk::DescriptorPool pool = VulkanRenderer::instance->getDescriptorPoolHandle();

	m_descriptor_sets.clear();
	m_bound_views.assign(VulkanRenderer::k_frames_in_flight, vk::ImageView {});
	for (uint32_t i = 0; i < VulkanRenderer::k_frames_in_flight; ++i) {
		const vk::DescriptorSetAllocateInfo alloc_info(pool, 1, &set_layout);
		auto allocated = device.allocateDescriptorSets(alloc_info);
		m_descriptor_sets.push_back(std::move(allocated[0]));
		setDebugName(core, *m_descriptor_sets[i], std::format("SsrPass DescriptorSet[{}]", i));
	}
}

void SsrPass::createTarget(const VulkanCore& core, vk::Extent2D extent) {
	m_target.create(core, extent, m_scene_format, "SsrPass");
	std::ranges::fill(m_bound_views, vk::ImageView {});
}

void SsrPass::onResize(vk::Extent2D extent) {
	if (m_core != nullptr) {
		createTarget(*m_core, extent);
	}
}

auto SsrPass::record(vk::CommandBuffer cmd, uint32_t frame_index, vk::ImageView source_view) -> vk::ImageView {
	if (!m_pipeline.isReady() || frame_index >= m_descriptor_sets.size() || !source_view || !m_target.isReady()) {
		return source_view;
	}

	const vk::ImageView depth_view = VulkanRenderer::instance->getDepthView();
	const vk::ImageView normal_view = VulkanRenderer::instance->getSceneNormalView();
	if (!depth_view || !normal_view) {
		return source_view;
	}

	// Rebound when the chain's source changes. The depth and geometry views are stable for the life of the
	// target, but they are written in the same call so a resize cannot leave one set stale against the other
	if (m_bound_views[frame_index] != source_view) {
		const std::array image_infos {
		  vk::DescriptorImageInfo(*m_sampler, source_view, vk::ImageLayout::eShaderReadOnlyOptimal),
		  // Depth sits in eDepthReadOnlyOptimal, not eShaderReadOnlyOptimal - recordFrame() leaves it there so
		  // the overlay stage can still depth-test against it after this pass has sampled it
		  vk::DescriptorImageInfo(*m_point_sampler, depth_view, vk::ImageLayout::eDepthReadOnlyOptimal),
		  vk::DescriptorImageInfo(*m_point_sampler, normal_view, vk::ImageLayout::eShaderReadOnlyOptimal)
		};

		const std::array writes {
		  vk::WriteDescriptorSet(
		      *m_descriptor_sets[frame_index], 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, image_infos.data()
		  ),
		  vk::WriteDescriptorSet(
		      *m_descriptor_sets[frame_index], 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &image_infos[1]
		  ),
		  vk::WriteDescriptorSet(*m_descriptor_sets[frame_index], 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &image_infos[2])
		};
		m_core->getDevice().updateDescriptorSets(writes, {});
		m_bound_views[frame_index] = source_view;
	}

	m_target.beginScope(cmd);

	// From the frame snapshot, not from renderer members - the render thread must see the same camera the
	// geometry buffer was written with, and members move on the main thread
	const auto* frame = VulkanRenderer::instance->renderingFrame();
	const auto settings = frame != nullptr ? frame->post_process.ssr : VulkanRenderer::PostProcessSettings::Ssr {};

	const glm::mat4 view_projection = frame != nullptr ? frame->frame_data.view_projection : glm::mat4(1.0f);

	Params params {};
	params.view_projection = view_projection;
	params.inverse_view_projection = glm::inverse(view_projection);
	params.camera_position = glm::vec4(frame != nullptr ? frame->frame_data.camera_position : glm::vec3(0.0f), 1.0f);
	params.tuning =
	    glm::vec4(settings.intensity, settings.max_roughness, settings.thickness, static_cast<float>(settings.max_steps));
	params.marching = glm::vec4(settings.stride, static_cast<float>(frame != nullptr ? frame->render_mode : 0u), 0.0f, 0.0f);

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
