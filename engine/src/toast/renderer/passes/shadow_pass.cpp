/**
 * @file shadow_pass.cpp
 * @author dario
 * @date 01/08/2026
 */

#include "shadow_pass.hpp"

#include "../descriptor_writer.hpp"
#include "../shader_cache.hpp"
#include "../vulkan_core.hpp"
#include "../vulkan_debug.hpp"
#include "../vulkan_mesh.hpp"
#include "../vulkan_renderer.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <format>
#include <toast/assets/assets.hpp>
#include <toast/log.hpp>
#include <tracy/Tracy.hpp>

namespace renderer {

namespace {

auto shadowLayerRange(uint32_t base_layer, uint32_t layer_count) -> vk::ImageSubresourceRange {
	return {vk::ImageAspectFlagBits::eDepth, 0, 1, base_layer, layer_count};
}

}

auto createShadowSampler(const VulkanCore& core, vk::Format format) -> vk::raii::Sampler {
	const auto props = core.getPhysicalDevice().getFormatProperties(format);
	const bool linear_filterable =
	    (props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear) != vk::FormatFeatureFlags {};

	// Nearest is correct, just harder-edged, so degrade rather than refuse
	const vk::Filter filter = linear_filterable ? vk::Filter::eLinear : vk::Filter::eNearest;

	vk::SamplerCreateInfo sampler_ci {};
	sampler_ci.magFilter = filter;
	sampler_ci.minFilter = filter;
	sampler_ci.addressModeU = vk::SamplerAddressMode::eClampToBorder;
	sampler_ci.addressModeV = vk::SamplerAddressMode::eClampToBorder;
	sampler_ci.addressModeW = vk::SamplerAddressMode::eClampToBorder;
	// Outside the fitted map reads depth 1.0, which passes the comparison and so reads as lit
	sampler_ci.borderColor = vk::BorderColor::eFloatOpaqueWhite;
	sampler_ci.mipmapMode = vk::SamplerMipmapMode::eNearest;
	sampler_ci.compareEnable = VK_TRUE;
	// The shader passes the surface's own depth; a stored depth at or beyond it means nothing occludes
	sampler_ci.compareOp = vk::CompareOp::eLessOrEqual;

	return {core.getDevice(), sampler_ci};
}

auto ShadowPass::selectShadowFormat(const VulkanCore& core) -> vk::Format {
	// No stencil: a combined format forces an aspect mask the sampler cannot read from
	const std::array candidates {vk::Format::eD32Sfloat, vk::Format::eD16Unorm};

	constexpr auto required = vk::FormatFeatureFlagBits::eDepthStencilAttachment | vk::FormatFeatureFlagBits::eSampledImage;

	for (const auto candidate : candidates) {
		const auto props = core.getPhysicalDevice().getFormatProperties(candidate);
		if ((props.optimalTilingFeatures & required) == required) {
			return candidate;
		}
	}

	return vk::Format::eUndefined;
}

ShadowPass::ShadowPass(const VulkanCore& core) : m_core(&core) {
	ZoneScoped;
	m_format = selectShadowFormat(core);
	if (m_format == vk::Format::eUndefined) {
		TOAST_ERROR("Render", "No sampleable depth format available, shadows are disabled");
		return;
	}

	const auto uid = assets::resolveURI("core://shaders/shadow_depth.slang");
	const auto shader = uid.has_value() ? ShaderCache::get().acquire(*uid) : nullptr;
	if (!shader) {
		TOAST_ERROR("Render", "ShadowPass shader core://shaders/shadow_depth.slang unavailable, shadows will not render");
		return;
	}

	m_shader_layout.rebuild(core, shader->reflection, "ShadowPass");

	VulkanPipeline::Config config;
	config.pipeline_type = VulkanPipeline::PipelineType::graphics;
	config.debug_name = "ShadowPass";
	config.depth_only = true;
	config.depth_format = m_format;
	// Viewport and scissor are dynamic, so one pipeline covers both the cascade and punctual resolutions
	config.extent = vk::Extent2D {shadows::cascadeResolution(), shadows::cascadeResolution()};
	config.shader_spirv = shader->spirv;
	config.pipeline_layout = *m_shader_layout.getPipelineLayout();
	config.vertex_bindings = {vertexBindingDescription()};
	// Position only, though the full renderer::Vertex stream is what gets bound - see shadow_depth.slang
	config.vertex_attributes = {vertexAttributeDescriptions()[0]};
	config.depth_test = true;
	config.depth_write = true;
	// Acne belongs here, not in a normal offset: slope-scaled bias shifts along the depth gradient only, and
	// a sideways shift is what detaches a shadow from its caster
	config.depth_bias_constant = 1.5f;
	config.depth_bias_slope = 3.0f;
	// Both sides: culling either leaves a plane or a foliage card casting nothing at all, which is worse than
	// the self-shadowing the lookup's normal offset already handles
	config.cull_mode = vk::CullModeFlagBits::eNone;

	// One per mask a LayerGroup uses, built up front because a pipeline's viewMask must equal the scope's.
	// Single view is 0x1, not 0 - 0 means "not multiview" and leaves SV_ViewID undefined
	constexpr std::array view_masks {
	  1u,
	  (1u << shadows::k_cascade_count) - 1u,
	  (1u << shadows::k_cube_faces) - 1u,
	};

	static_assert(view_masks.size() == std::tuple_size_v<decltype(m_pipeline_sets)>, "one pipeline set per mask");

	for (size_t mask_index = 0; mask_index < view_masks.size(); ++mask_index) {
		const uint32_t view_mask = view_masks[mask_index];

		// Filled in place - VulkanPipeline is neither copyable nor movable
		PipelineSet& set = m_pipeline_sets[mask_index];
		set.view_mask = view_mask;

		VulkanPipeline::Config mask_config = config;
		mask_config.view_mask = view_mask;
		mask_config.debug_name = std::format("ShadowPass [viewMask {:#x}]", view_mask);
		set.pipeline.rebuild(core, mask_config);
	}

	createResources(core);

	TOAST_INFO(
	    "Render",
	    "ShadowPass ready: {} cascades at {}x{}, {} punctual layers at {}x{} ({})",
	    shadows::k_cascade_count,
	    shadows::cascadeResolution(),
	    shadows::cascadeResolution(),
	    shadows::k_punctual_layer_count,
	    shadows::punctualResolution(),
	    shadows::punctualResolution(),
	    vk::to_string(m_format)
	);
}

auto ShadowPass::pipelineSetFor(uint32_t view_mask) const -> const PipelineSet* {
	const auto it = std::ranges::find(m_pipeline_sets, view_mask, &PipelineSet::view_mask);
	return it != m_pipeline_sets.end() ? &*it : nullptr;
}

void ShadowPass::createShadowMap(
    const VulkanCore& core, ShadowMap& map, uint32_t resolution, std::span<const uint32_t> group_layers,
    std::string_view debug_name
) {
	ZoneScoped;
	const auto& device = core.getDevice();

	uint32_t layer_count = 0;
	for (const uint32_t count : group_layers) {
		layer_count += count;
	}

	map.resolution = resolution;
	map.layer_count = layer_count;
	map.layout = vk::ImageLayout::eUndefined;

	vk::ImageCreateInfo image_ci {};
	image_ci.imageType = vk::ImageType::e2D;
	image_ci.format = m_format;
	image_ci.extent = vk::Extent3D {resolution, resolution, 1};
	image_ci.mipLevels = 1;
	image_ci.arrayLayers = layer_count;
	image_ci.samples = vk::SampleCountFlagBits::e1;
	image_ci.tiling = vk::ImageTiling::eOptimal;
	image_ci.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled;
	image_ci.sharingMode = vk::SharingMode::eExclusive;
	image_ci.initialLayout = vk::ImageLayout::eUndefined;

	vma::AllocationCreateInfo allocation_ci {};
	allocation_ci.usage = vma::MemoryUsage::eAutoPreferDevice;

	map.image.emplace(core.getAllocator().createImage(image_ci, allocation_ci));
	setDebugName(core, **map.image, std::string(debug_name));

	vk::ImageViewCreateInfo array_view_ci {};
	array_view_ci.image = **map.image;
	array_view_ci.viewType = vk::ImageViewType::e2DArray;
	array_view_ci.format = m_format;
	array_view_ci.subresourceRange = shadowLayerRange(0, layer_count);
	map.array_view.emplace(device, array_view_ci);
	setDebugName(core, **map.array_view, std::format("{} ArrayView", debug_name));

	// One render-target view per group, plus the array view shaders sample. SV_ViewID indexes relative to the
	// view's base layer, which is why a group's layers must be consecutive
	map.groups.clear();
	map.groups.reserve(group_layers.size());

	uint32_t base_layer = 0;
	for (const uint32_t count : group_layers) {
		LayerGroup group;
		group.base_layer = base_layer;
		group.layer_count = count;
		group.view_mask = (1u << count) - 1u;
		// A fresh image holds undefined depth, which reads as an arbitrary occluder until cleared
		group.dirty = true;

		vk::ImageViewCreateInfo view_ci {};
		view_ci.image = **map.image;
		// Array view even for one layer - every group here is a multiview pass
		view_ci.viewType = vk::ImageViewType::e2DArray;
		view_ci.format = m_format;
		view_ci.subresourceRange = shadowLayerRange(base_layer, count);
		group.view.emplace(device, view_ci);
		setDebugName(core, **group.view, std::format("{} GroupView[{}..{}]", debug_name, base_layer, base_layer + count - 1));

		map.groups.push_back(std::move(group));
		base_layer += count;
	}
}

void ShadowPass::createResources(const VulkanCore& core) {
	ZoneScoped;
	const auto& device = core.getDevice();
	const auto& layouts = m_shader_layout.getDescriptorSetLayouts();
	if (layouts.empty()) {
		TOAST_ERROR("Render", "ShadowPass shader layout has no descriptor sets");
		return;
	}

	m_sampler = createShadowSampler(core, m_format);
	setDebugName(core, *m_sampler, "ShadowPass Sampler");

	const vk::DescriptorSetLayout set_layout = *layouts[0];
	const vk::DescriptorPool pool = VulkanRenderer::instance->getDescriptorPoolHandle();

	// Once, not per frame index - re-acquiring inside the loop repeats the same cache lookup
	const auto uid = assets::resolveURI("core://shaders/shadow_depth.slang");
	const auto shader = uid.has_value() ? ShaderCache::get().acquire(*uid) : nullptr;
	if (!shader) {
		TOAST_ERROR("Render", "ShadowPass shader disappeared between pipeline creation and resource setup");
		return;
	}

	m_targets.resize(VulkanRenderer::k_frames_in_flight);
	m_descriptor_sets.clear();
	m_descriptor_sets.reserve(VulkanRenderer::k_frames_in_flight);

	for (uint32_t i = 0; i < VulkanRenderer::k_frames_in_flight; ++i) {
		auto& target = m_targets[i];

		// One multiview group: the views differ only in their matrix, and four passes submitted the same
		// geometry four times
		const std::array cascade_groups {shadows::k_cascade_count};
		createShadowMap(
		    core, target.cascades, shadows::cascadeResolution(), cascade_groups, std::format("ShadowPass Cascades[{}]", i)
		);

		// Spots stay single-view: two can want different sub-rect resolutions, and multiview has one viewport
		// per pass. A cube's six faces share one resolution, so each cube is a group
		std::vector<uint32_t> punctual_groups(shadows::k_max_spot_shadows, 1u);
		punctual_groups.insert(punctual_groups.end(), shadows::k_max_point_shadows, shadows::k_cube_faces);
		createShadowMap(
		    core, target.punctual, shadows::punctualResolution(), punctual_groups, std::format("ShadowPass Punctual[{}]", i)
		);

		vk::BufferCreateInfo buffer_ci {};
		buffer_ci.size = sizeof(ShadowUBO);
		buffer_ci.usage = vk::BufferUsageFlagBits::eUniformBuffer;

		vma::AllocationCreateInfo buffer_alloc_ci {};
		buffer_alloc_ci.usage = vma::MemoryUsage::eAutoPreferHost;
		buffer_alloc_ci.flags = vma::AllocationCreateFlagBits::eMapped | vma::AllocationCreateFlagBits::eHostAccessSequentialWrite;

		target.ubo.gpu_buffer.emplace(core.getAllocator().createBuffer(buffer_ci, buffer_alloc_ci));
		setDebugName(core, **target.ubo.gpu_buffer, std::format("ShadowPass ShadowUBO[{}]", i));

		const vk::DescriptorSetAllocateInfo alloc_info(pool, 1, &set_layout);
		auto allocated = device.allocateDescriptorSets(alloc_info);
		m_descriptor_sets.push_back(std::move(allocated[0]));
		setDebugName(core, *m_descriptor_sets[i], std::format("ShadowPass DescriptorSet[{}]", i));

		DescriptorWriter writer;

		// Bound by declared name, same convention MaterialPass::createFrameSets() uses for set 0
		for (const auto& binding : shader->reflection.bindings) {
			if (binding.set != 0) {
				continue;
			}
			if (binding.name == "gShadow") {
				writer.buffer(
				    *m_descriptor_sets[i], binding.binding, vk::DescriptorType::eUniformBuffer, **target.ubo.gpu_buffer, sizeof(ShadowUBO)
				);
			} else if (binding.name == "gInstances") {
				// ShadowPass's own buffer, not the colour pass's - see RenderFrame::shadow_instance_data
				writer.buffer(
				    *m_descriptor_sets[i],
				    binding.binding,
				    vk::DescriptorType::eStorageBuffer,
				    VulkanRenderer::instance->getShadowInstanceBuffer(i),
				    sizeof(VulkanRenderer::InstanceData) * VulkanRenderer::k_max_instances
				);
			}
		}

		writer.flush(device);
	}
}

auto ShadowPass::getCascadeMapView(uint32_t frame_index) const -> vk::ImageView {
	if (frame_index >= m_targets.size() || !m_targets[frame_index].cascades.array_view.has_value()) {
		return nullptr;
	}
	return **m_targets[frame_index].cascades.array_view;
}

auto ShadowPass::getPunctualMapView(uint32_t frame_index) const -> vk::ImageView {
	if (frame_index >= m_targets.size() || !m_targets[frame_index].punctual.array_view.has_value()) {
		return nullptr;
	}
	return **m_targets[frame_index].punctual.array_view;
}

void ShadowPass::recordMap(vk::CommandBuffer cmd, ShadowMap& map, uint32_t frame_index, bool directional) {
	ZoneScoped;
	const auto* frame = VulkanRenderer::instance->renderingFrame();
	if (frame == nullptr || !map.image.has_value()) {
		return;
	}

	// Nothing samples these while the shaders trace, but MaterialPass still binds them - and a descriptor
	// naming an eUndefined image is invalid even unread, which is why this is not an early return
	if (frame->frame_data.traced_shadow_params.x >= 0.5f && map.layout == vk::ImageLayout::eShaderReadOnlyOptimal) {
		return;
	}

	// Resolved before the barrier, so a map with nothing to record is skipped rather than paying two
	// full-image transitions to do nothing between them
	std::vector<const VulkanRenderer::ShadowView*> layer_views(map.layer_count, nullptr);
	std::vector<uint32_t> layer_view_indices(map.layer_count, 0);
	bool any_layer_recorded = false;
	for (uint32_t i = 0; i < frame->shadows.views.size(); ++i) {
		const auto& candidate = frame->shadows.views[i];
		if (candidate.directional == directional && candidate.layer < map.layer_count) {
			layer_views[candidate.layer] = &candidate;
			layer_view_indices[candidate.layer] = i;
		}
	}
	for (const auto& group : map.groups) {
		if (group.dirty || layer_views[group.base_layer] != nullptr) {
			any_layer_recorded = true;
			break;
		}
	}
	if (!any_layer_recorded) {
		return;
	}

	const vk::ImageMemoryBarrier to_attachment(
	    map.layout == vk::ImageLayout::eShaderReadOnlyOptimal ? vk::AccessFlagBits::eShaderRead : vk::AccessFlags {},
	    vk::AccessFlagBits::eDepthStencilAttachmentWrite,
	    map.layout,
	    vk::ImageLayout::eDepthAttachmentOptimal,
	    VK_QUEUE_FAMILY_IGNORED,
	    VK_QUEUE_FAMILY_IGNORED,
	    **map.image,
	    shadowLayerRange(0, map.layer_count)
	);
	cmd.pipelineBarrier(
	    map.layout == vk::ImageLayout::eShaderReadOnlyOptimal ? vk::PipelineStageFlagBits::eFragmentShader
	                                                          : vk::PipelineStageFlagBits::eTopOfPipe,
	    vk::PipelineStageFlagBits::eEarlyFragmentTests,
	    {},
	    nullptr,
	    nullptr,
	    to_attachment
	);
	map.layout = vk::ImageLayout::eDepthAttachmentOptimal;

	// Whole layer: what gets cleared, regardless of how much of it a view actually draws into
	const vk::Rect2D layer_area({0, 0}, vk::Extent2D {map.resolution, map.resolution});

	constexpr uint32_t k_push_size = 2 * sizeof(uint32_t);

	for (auto& group : map.groups) {
		// All-or-nothing: a partial group would leave SV_ViewID indexing another light's matrix
		bool occupied = layer_views[group.base_layer] != nullptr;
		for (uint32_t i = 1; occupied && i < group.layer_count; ++i) {
			const auto* member = layer_views[group.base_layer + i];
			occupied = member != nullptr && layer_view_indices[group.base_layer + i] == layer_view_indices[group.base_layer] + i;
		}

		// Cleared once, or last frame's depths keep shadowing after a light stops casting. Clearing every
		// frame after that is pure cost, and most of the 16 reserved slots are usually empty
		if (!occupied && !group.dirty) {
			continue;
		}
		group.dirty = occupied;

		const VulkanRenderer::ShadowView* view = occupied ? layer_views[group.base_layer] : nullptr;
		const uint32_t view_index = layer_view_indices[group.base_layer];

		vk::RenderingAttachmentInfo depth_attachment {};
		depth_attachment.imageView = **group.view;
		depth_attachment.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal;
		depth_attachment.loadOp = vk::AttachmentLoadOp::eClear;
		depth_attachment.storeOp = vk::AttachmentStoreOp::eStore;
		depth_attachment.clearValue = vk::ClearValue(vk::ClearDepthStencilValue {1.0f, 0});

		vk::RenderingInfo rendering_info {};
		// Full layer, so what lies outside a shrunken sub-rect is cleared rather than left from a larger frame
		rendering_info.renderArea = layer_area;
		// Ignored when viewMask is non-zero; the mask decides how many layers are written there
		rendering_info.layerCount = 1;
		rendering_info.viewMask = group.view_mask;
		rendering_info.colorAttachmentCount = 0;
		rendering_info.pDepthAttachment = &depth_attachment;

		cmd.beginRendering(rendering_info);
		++m_pass_count;

		const PipelineSet* pipelines = pipelineSetFor(group.view_mask);
		if (view != nullptr && pipelines != nullptr && pipelines->pipeline.isReady()) {
			// A punctual shadow shrinks with distance into a sub-rect at the origin; the shader scales its
			// lookups by GpuLight::shadow_atlas to match
			const uint32_t view_resolution = std::clamp(view->resolution, 1u, map.resolution);
			const vk::Viewport viewport(
			    0.0f, 0.0f, static_cast<float>(view_resolution), static_cast<float>(view_resolution), 0.0f, 1.0f
			);
			const vk::Rect2D scissor({0, 0}, vk::Extent2D {view_resolution, view_resolution});

			cmd.setViewport(0, std::array {viewport});
			cmd.setScissor(0, std::array {scissor});
			cmd.bindDescriptorSets(
			    vk::PipelineBindPoint::eGraphics,
			    *m_shader_layout.getPipelineLayout(),
			    0,
			    std::array<vk::DescriptorSet, 1> {*m_descriptor_sets[frame_index]},
			    {}
			);

			// A multiview group draws the union of what its views see - one pass cannot cull per view. That
			// costs little for cascades, whose near spheres sit inside the far one anyway
			const auto visible = [&](const VulkanRenderer::MeshInstanceProxy& proxy) {
				for (uint32_t i = 0; i < group.layer_count; ++i) {
					const auto* member = layer_views[group.base_layer + i];
					if (member == nullptr) {
						continue;
					}
					const glm::vec3 offset = proxy.bounds_center - glm::vec3(member->cull_sphere);
					const float reach = member->cull_sphere.w + proxy.bounds_radius;
					if (glm::dot(offset, offset) <= (reach * reach)) {
						return true;
					}
				}
				return false;
			};

			// One draw per *run* of adjacent proxies sharing a mesh. The proxy list is pre-sorted and
			// shadow_instance_data is indexed one-to-one with it, so a run of indices is a run of slots
			const auto& proxies = frame->mesh_instances;
			const size_t instance_count = frame->shadow_instance_data.size();
			const vk::Buffer posed_vertices = VulkanRenderer::instance->getPosedVertexBuffer(frame_index);

			const auto posed_offset_of = [&](const VulkanRenderer::MeshInstanceProxy& proxy) {
				return posed_vertices ? proxy.posed_vertex_offset : VulkanRenderer::MeshInstanceProxy::k_no_posed_vertices;
			};

			const auto eligible = [&](size_t index) {
				const auto& proxy = proxies[index];
				return proxy.mesh != nullptr && proxy.mesh->isReady() && visible(proxy);
			};

			// One pipeline for the whole loop - a posed instance is a static draw whose vertices come from
			// somewhere else
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines->pipeline.getPipeline());

			for (size_t i = 0; i < proxies.size() && i < instance_count;) {
				if (!eligible(i)) {
					++i;
					continue;
				}

				const uint32_t posed_offset = posed_offset_of(proxies[i]);
				const bool posed = posed_offset != VulkanRenderer::MeshInstanceProxy::k_no_posed_vertices;

				// A culled proxy ends the run rather than being skipped: instances must be contiguous, and
				// jumping a slot would draw the wrong transform. A posed proxy is always a run of one
				size_t run = 1;
				if (!posed) {
					while (i + run < proxies.size() && i + run < instance_count && eligible(i + run) &&
					       proxies[i + run].mesh == proxies[i].mesh &&
					       posed_offset_of(proxies[i + run]) == VulkanRenderer::MeshInstanceProxy::k_no_posed_vertices) {
						++run;
					}
				}

				const ShadowPushConstants push {
				  .instance_base = static_cast<uint32_t>(i),
				  .view_index = view_index,
				};
				cmd.pushConstants(
				    *m_shader_layout.getPipelineLayout(),
				    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
				    0,
				    k_push_size,
				    &push
				);

				if (posed) {
					proxies[i].mesh->bindPosed(cmd, posed_vertices, posed_offset);
				} else {
					proxies[i].mesh->bind(cmd);
				}
				proxies[i].mesh->draw(cmd, static_cast<uint32_t>(run));
				++m_draw_count;

				i += run;
			}
		}

		cmd.endRendering();
	}

	const vk::ImageMemoryBarrier to_sampled(
	    vk::AccessFlagBits::eDepthStencilAttachmentWrite,
	    vk::AccessFlagBits::eShaderRead,
	    vk::ImageLayout::eDepthAttachmentOptimal,
	    vk::ImageLayout::eShaderReadOnlyOptimal,
	    VK_QUEUE_FAMILY_IGNORED,
	    VK_QUEUE_FAMILY_IGNORED,
	    **map.image,
	    shadowLayerRange(0, map.layer_count)
	);
	cmd.pipelineBarrier(
	    vk::PipelineStageFlagBits::eLateFragmentTests, vk::PipelineStageFlagBits::eFragmentShader, {}, nullptr, nullptr, to_sampled
	);
	map.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
}

void ShadowPass::recordPre(vk::CommandBuffer cmd, uint32_t frame_index, uint32_t image_index) {
	ZoneScoped;
	(void)image_index;

	// Every mask, not just one - a missing cascade pipeline leaves the directional map clearing itself and
	// nothing else, which reads as "shadows stopped working" with no error anywhere
	const bool pipelines_ready = !m_pipeline_sets.empty() && std::ranges::all_of(m_pipeline_sets, [](const PipelineSet& set) {
		return set.pipeline.isReady();
	});
	if (!isEnabled() || !pipelines_ready || frame_index >= m_targets.size()) {
		return;
	}

	const auto* frame = VulkanRenderer::instance->renderingFrame();
	if (frame == nullptr) {
		return;
	}

	auto& target = m_targets[frame_index];

	m_draw_count = 0;
	m_pass_count = 0;

	if (target.ubo.gpu_buffer.has_value()) {
		ShadowUBO ubo {};
		const size_t view_count = std::min<size_t>(frame->shadows.matrices.size(), shadows::k_max_shadow_views);
		std::copy_n(frame->shadows.matrices.begin(), view_count, ubo.view_projection.begin());

		const auto& allocation = target.ubo.gpu_buffer->getAllocation();
		if (auto* mapped = allocation.getInfo().pMappedData) {
			std::memcpy(mapped, &ubo, sizeof(ShadowUBO));
			allocation.flush(0, sizeof(ShadowUBO));
		}
	}

	recordMap(cmd, target.cascades, frame_index, true);
	recordMap(cmd, target.punctual, frame_index, false);
}

}
