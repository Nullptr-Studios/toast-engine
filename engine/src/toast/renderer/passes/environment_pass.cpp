/**
 * @file environment_pass.cpp
 * @author dario
 * @date 03/08/2026
 */

#include "environment_pass.hpp"

#include "../cube_face_basis.hpp"
#include "../shader_cache.hpp"
#include "../vulkan_core.hpp"
#include "../vulkan_debug.hpp"
#include "../vulkan_renderer.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <toast/assets/asset_manager.hpp>
#include <toast/assets/assets.hpp>
#include <toast/log.hpp>

namespace renderer {

namespace {

constexpr uint32_t k_sky_size = 256;
constexpr uint32_t k_irradiance_size = 32;
constexpr uint32_t k_prefiltered_size = 128;

/// @brief Roughness levels in the prefiltered chain
///
/// Five gives mirror, slightly-rough, half-rough, rough and fully-rough. More levels buy little: the
/// coarsest ones are already so blurred that the difference between them is not visible
constexpr uint32_t k_prefiltered_mips = 5;

}

EnvironmentPass::EnvironmentPass(const VulkanCore& core, vk::Format hdr_format, vk::Format depth_format)
    : m_core(&core),
      m_format(hdr_format),
      m_depth_format(depth_format) {
	const auto uid = assets::resolveURI("core://shaders/environment.slang");
	const auto shader = uid.has_value() ? ShaderCache::get().acquire(*uid) : nullptr;
	if (!shader) {
		TOAST_ERROR("Render", "EnvironmentPass shader core://shaders/environment.slang unavailable; ambient stays uniform");
		setEnabled(false);
		return;
	}

	m_shader_layout.rebuild(core, shader->reflection, "EnvironmentPass");

	// Linear between mips: the prefiltered chain stores discrete roughness levels and a surface's roughness
	// lands between them, so without it roughness steps visibly
	const auto sampler_ci = linearClampMippedSamplerInfo(VK_LOD_CLAMP_NONE);
	m_sampler = vk::raii::Sampler(core.getDevice(), sampler_ci);
	setDebugName(core, *m_sampler, "EnvironmentPass Sampler");

	m_prefiltered_mips = k_prefiltered_mips;

	m_sky.create(core, m_format, k_sky_size, 1, "EnvironmentPass Sky");
	m_irradiance.create(core, m_format, k_irradiance_size, 1, "EnvironmentPass Irradiance");
	m_prefiltered.create(core, m_format, k_prefiltered_size, k_prefiltered_mips, "EnvironmentPass Prefiltered");

	createPipelines(core);
	// Before createDescriptors: the placeholder is what binding 1 points at until something is imported
	createEquirectPlaceholder(core);
	createDescriptors(core);
}

void EnvironmentPass::createPipelines(const VulkanCore& core) {
	const auto uid = assets::resolveURI("core://shaders/environment.slang");
	const auto shader = uid.has_value() ? ShaderCache::get().acquire(*uid) : nullptr;
	if (!shader) {
		return;
	}

	VulkanPipeline::Config config;
	config.pipeline_type = VulkanPipeline::PipelineType::graphics;
	config.color_format = m_format;
	// Viewport and scissor are dynamic, so one pipeline covers every face and mip size
	config.extent = vk::Extent2D {1, 1};
	config.shader_spirv = shader->spirv;
	config.pipeline_layout = *m_shader_layout.getPipelineLayout();
	config.vertex_bindings = {};
	config.vertex_attributes = {};
	config.cull_mode = vk::CullModeFlagBits::eNone;
	config.depth_test = false;
	config.depth_write = false;

	config.debug_name = "EnvironmentPass Sky";
	config.fragment_entry = "fragmentSky";
	m_sky_pipeline.rebuild(core, config);

	config.debug_name = "EnvironmentPass Equirect";
	config.fragment_entry = "fragmentEquirect";
	m_equirect_pipeline.rebuild(core, config);

	config.debug_name = "EnvironmentPass Irradiance";
	config.fragment_entry = "fragmentIrradiance";
	m_irradiance_pipeline.rebuild(core, config);

	config.debug_name = "EnvironmentPass Prefilter";
	config.fragment_entry = "fragmentPrefilter";
	m_prefilter_pipeline.rebuild(core, config);

	// The skybox is the odd one out: it draws into the scene target alongside the geometry rather than into a
	// cubemap face, so it needs the depth attachment's format to be pipeline-compatible with the scene pass.
	// It runs before any geometry and neither tests nor writes depth - the meshes simply draw over it
	config.debug_name = "EnvironmentPass Skybox";
	config.vertex_entry = "vertexSkybox";
	config.fragment_entry = "fragmentSkybox";
	config.depth_format = m_depth_format;
	config.depth_test = true;
	config.depth_write = false;
	config.depth_compare = vk::CompareOp::eLessOrEqual;
	// Declared, not written (write_extra_color stays false): the sky is not a surface, and the cleared
	// zero-length normal already says so. The three pipelines above render into cubemap faces instead and must
	// not carry it - their scope has one attachment
	config.extra_color_formats = worldStageExtraColorFormats();
	m_skybox_pipeline.rebuild(core, config);
}

namespace {

/// @brief Format imported environments are uploaded in
///
constexpr vk::Format k_equirect_format = vk::Format::eR32G32B32A32Sfloat;

}

void EnvironmentPass::createEquirectPlaceholder(const VulkanCore& core) {
	vk::SamplerCreateInfo sampler_ci {};
	sampler_ci.magFilter = vk::Filter::eLinear;
	sampler_ci.minFilter = vk::Filter::eLinear;
	// Repeat on U: longitude wraps, and the seam behind the camera falls exactly on the edge. Clamping there
	// smears the last column of texels across it
	sampler_ci.addressModeU = vk::SamplerAddressMode::eRepeat;
	sampler_ci.addressModeV = vk::SamplerAddressMode::eClampToEdge;
	sampler_ci.addressModeW = vk::SamplerAddressMode::eClampToEdge;
	sampler_ci.mipmapMode = vk::SamplerMipmapMode::eNearest;
	m_equirect_sampler = vk::raii::Sampler(core.getDevice(), sampler_ci);
	setDebugName(core, *m_equirect_sampler, "EnvironmentPass EquirectSampler");

	const assets::HdrImage placeholder {
	  .width = 1, .height = 1, .pixels = {0.0f, 0.0f, 0.0f, 1.0f}
	};
	std::ignore = uploadEquirect(placeholder);
}

auto EnvironmentPass::uploadEquirect(const assets::HdrImage& image) -> bool {
	if (m_core == nullptr || !image.valid()) {
		return false;
	}

	// Anything in flight may still be sampling the previous image through the descriptor set
	m_core->getDevice().waitIdle();

	m_equirect_view.reset();
	m_equirect_image.reset();

	vk::ImageCreateInfo image_ci {};
	image_ci.imageType = vk::ImageType::e2D;
	image_ci.format = k_equirect_format;
	image_ci.extent = vk::Extent3D {image.width, image.height, 1};
	image_ci.mipLevels = 1;
	image_ci.arrayLayers = 1;
	image_ci.samples = vk::SampleCountFlagBits::e1;
	image_ci.tiling = vk::ImageTiling::eOptimal;
	image_ci.usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
	image_ci.sharingMode = vk::SharingMode::eExclusive;
	image_ci.initialLayout = vk::ImageLayout::eUndefined;

	vma::AllocationCreateInfo allocation_ci {};
	allocation_ci.usage = vma::MemoryUsage::eAutoPreferDevice;

	m_equirect_image.emplace(m_core->getAllocator().createImage(image_ci, allocation_ci));
	setDebugName(*m_core, **m_equirect_image, "EnvironmentPass Equirectangular");

	const vk::DeviceSize bytes = image.pixels.size() * sizeof(float);

	vk::BufferCreateInfo staging_ci {};
	staging_ci.size = bytes;
	staging_ci.usage = vk::BufferUsageFlagBits::eTransferSrc;
	staging_ci.sharingMode = vk::SharingMode::eExclusive;

	vma::AllocationCreateInfo staging_alloc {};
	staging_alloc.usage = vma::MemoryUsage::eAuto;
	staging_alloc.flags = vma::AllocationCreateFlagBits::eHostAccessSequentialWrite | vma::AllocationCreateFlagBits::eMapped;

	auto staging = m_core->getAllocator().createBuffer(staging_ci, staging_alloc);
	auto* mapped = staging.getAllocation().getInfo().pMappedData;
	if (mapped == nullptr) {
		return false;
	}
	std::memcpy(mapped, image.pixels.data(), bytes);
	staging.getAllocation().flush(0, bytes);

	const auto& device = m_core->getDevice();
	const vk::CommandPoolCreateInfo pool_ci(vk::CommandPoolCreateFlagBits::eTransient, m_core->getGraphicsQueueFamilyIndex());
	const vk::raii::CommandPool pool(device, pool_ci);
	const vk::CommandBufferAllocateInfo alloc_info(*pool, vk::CommandBufferLevel::ePrimary, 1);
	vk::raii::CommandBuffers buffers(device, alloc_info);
	auto& cmd = buffers.front();

	const vk::ImageSubresourceRange range(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);

	cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

	recordUndefinedToTransferDst(cmd, **m_equirect_image, range);

	vk::BufferImageCopy copy {};
	copy.imageSubresource = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
	copy.imageExtent = vk::Extent3D {image.width, image.height, 1};
	cmd.copyBufferToImage(*staging, **m_equirect_image, vk::ImageLayout::eTransferDstOptimal, copy);

	recordTransferDstToShaderRead(cmd, **m_equirect_image, range);

	cmd.end();

	submitAndWait(device, m_core->getGraphicsQueue(), *cmd);

	vk::ImageViewCreateInfo view_ci {};
	view_ci.image = **m_equirect_image;
	view_ci.viewType = vk::ImageViewType::e2D;
	view_ci.format = k_equirect_format;
	view_ci.subresourceRange = range;
	m_equirect_view.emplace(device, view_ci);
	setDebugName(*m_core, **m_equirect_view, "EnvironmentPass EquirectangularView");

	writeEquirectSet(*m_core);
	return true;
}

auto EnvironmentPass::setEnvironmentMap(std::string_view uri) -> bool {
	if (uri.empty()) {
		m_environment_uri.clear();
		m_has_equirect = false;
		m_precompute_pending = true;
		TOAST_INFO("Render", "Environment map cleared; returning to the generated sky");
		return true;
	}

	const auto bytes = assets::AssetManager::get().loadBytes(uri);
	if (!bytes.has_value()) {
		TOAST_ERROR("Render", "Environment map {} could not be read", uri);
		return false;
	}

	const auto image = assets::decodeHdr(*bytes);
	if (!image.valid()) {
		TOAST_ERROR("Render", "Environment map {} is not a readable HDR", uri);
		return false;
	}

	if (!uploadEquirect(image)) {
		TOAST_ERROR("Render", "Environment map {} could not be uploaded", uri);
		return false;
	}

	m_environment_uri = std::string(uri);
	m_has_equirect = true;
	// Rebuilds all three cubemaps, so the skybox and both convolutions move to the new sky together
	m_precompute_pending = true;

	TOAST_INFO("Render", "Environment map loaded from {} ({}x{})", uri, image.width, image.height);
	return true;
}

void EnvironmentPass::createDescriptors(const VulkanCore& core) {
	const auto& layouts = m_shader_layout.getDescriptorSetLayouts();
	if (layouts.empty() || !m_sky.isReady()) {
		return;
	}

	const vk::DescriptorSetLayout set_layout = *layouts[0];
	const vk::DescriptorSetAllocateInfo alloc_info(VulkanRenderer::instance->getDescriptorPoolHandle(), 1, &set_layout);
	auto allocated = core.getDevice().allocateDescriptorSets(alloc_info);
	m_sky_source_set = std::move(allocated[0]);
	setDebugName(core, *m_sky_source_set, "EnvironmentPass SkySourceSet");

	vk::DescriptorImageInfo image_info {};
	image_info.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
	image_info.imageView = m_sky.cubeView();
	image_info.sampler = *m_sampler;

	// Binding 1 is the equirectangular source. Written even when nothing has been imported - the shader's
	// layout declares it either way, and a descriptor left unwritten is undefined on any draw using the layout
	vk::DescriptorImageInfo equirect_info {};
	equirect_info.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
	equirect_info.imageView = m_equirect_view.has_value() ? **m_equirect_view : vk::ImageView {};
	equirect_info.sampler = *m_equirect_sampler;

	const std::array writes {
	  vk::WriteDescriptorSet(*m_sky_source_set, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &image_info),
	  vk::WriteDescriptorSet(*m_sky_source_set, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &equirect_info)
	};
	core.getDevice().updateDescriptorSets(writes, {});

	const vk::DescriptorSetAllocateInfo equirect_alloc(VulkanRenderer::instance->getDescriptorPoolHandle(), 1, &set_layout);
	auto equirect_allocated = core.getDevice().allocateDescriptorSets(equirect_alloc);
	m_equirect_set = std::move(equirect_allocated[0]);
	setDebugName(core, *m_equirect_set, "EnvironmentPass EquirectSet");
	writeEquirectSet(core);
}

void EnvironmentPass::writeEquirectSet(const VulkanCore& core) {
	if (*m_equirect_set == VK_NULL_HANDLE || !m_equirect_view.has_value()) {
		return;
	}

	// Binding 0 is deliberately the renderer's black cube rather than m_sky - see m_equirect_set
	vk::DescriptorImageInfo cube_info {};
	cube_info.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
	cube_info.imageView = VulkanRenderer::instance->getDefaultCubeView();
	cube_info.sampler = *m_sampler;

	vk::DescriptorImageInfo equirect_info {};
	equirect_info.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
	equirect_info.imageView = **m_equirect_view;
	equirect_info.sampler = *m_equirect_sampler;

	const std::array writes {
	  vk::WriteDescriptorSet(*m_equirect_set, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &cube_info),
	  vk::WriteDescriptorSet(*m_equirect_set, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &equirect_info)
	};
	core.getDevice().updateDescriptorSets(writes, {});
}

void EnvironmentPass::renderFace(
    vk::CommandBuffer cmd, const VulkanPipeline& pipeline, vk::DescriptorSet set, const CubemapTarget& target, uint32_t mip,
    uint32_t face, const Params& params
) {
	const uint32_t mip_size = target.mipSize(mip);
	const vk::Extent2D extent {mip_size, mip_size};

	vk::RenderingAttachmentInfo attachment {};
	attachment.imageView = target.faceView(mip, face);
	attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	attachment.loadOp = vk::AttachmentLoadOp::eDontCare;
	attachment.storeOp = vk::AttachmentStoreOp::eStore;

	vk::RenderingInfo rendering_info {};
	rendering_info.renderArea = vk::Rect2D({0, 0}, extent);
	rendering_info.layerCount = 1;
	rendering_info.colorAttachmentCount = 1;
	rendering_info.pColorAttachments = &attachment;

	cmd.beginRendering(rendering_info);

	const vk::Viewport viewport(0.0f, 0.0f, static_cast<float>(mip_size), static_cast<float>(mip_size), 0.0f, 1.0f);
	cmd.setViewport(0, std::array {viewport});
	cmd.setScissor(0, std::array {vk::Rect2D({0, 0}, extent)});

	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.getPipeline());
	if (set) {
		cmd.bindDescriptorSets(
		    vk::PipelineBindPoint::eGraphics, *m_shader_layout.getPipelineLayout(), 0, std::array<vk::DescriptorSet, 1> {set}, {}
		);
	}
	cmd.pushConstants(
	    *m_shader_layout.getPipelineLayout(),
	    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
	    0,
	    sizeof(Params),
	    &params
	);
	cmd.draw(3, 1, 0, 0);

	cmd.endRendering();
}

void EnvironmentPass::recordPre(vk::CommandBuffer cmd, uint32_t frame_index, uint32_t image_index) {
	(void)frame_index;
	(void)image_index;

	// Once. The environment is static, so re-running this every frame would burn a few milliseconds
	// producing identical results. Regenerating on a sky change is a matter of clearing this flag
	// Every pipeline, not just the sky one: the pending flag clears on the first run, so a convolution whose
	// pipeline was not ready yet would be skipped permanently and leave its cubemap at whatever the allocation
	// happened to contain - which reads as black, and only for the surfaces that sample that map
	if (!m_precompute_pending || !isEnabled() || !m_sky.isReady() || !m_sky_pipeline.isReady() ||
	    !m_irradiance_pipeline.isReady() || !m_prefilter_pipeline.isReady()) {
		return;
	}
	m_precompute_pending = false;
	m_has_data = true;

	const auto make_params = [this](uint32_t face, float roughness) {
		const auto basis = cubeFaceBasis(face);
		Params params {};
		params.face_right = glm::vec4(basis.right, 0.0f);
		params.face_up = glm::vec4(basis.up, 0.0f);
		params.face_forward = glm::vec4(basis.forward, 0.0f);
		params.roughness = roughness;
		params.intensity = m_intensity;
		return params;
	};

	// --- Sky ---
	m_sky.transition(
	    cmd,
	    vk::ImageLayout::eColorAttachmentOptimal,
	    vk::AccessFlagBits::eColorAttachmentWrite,
	    vk::PipelineStageFlagBits::eColorAttachmentOutput
	);
	// An imported HDR is projected onto the faces in place of the generated sky. Everything after this point
	// reads the cubemap and is identical either way, which is the whole reason the import is done here rather
	// than by giving the skybox and the convolutions each their own notion of what the sky is
	const bool use_equirect = m_has_equirect && m_equirect_pipeline.isReady() && *m_equirect_set != VK_NULL_HANDLE;
	for (uint32_t face = 0; face < 6; ++face) {
		if (use_equirect) {
			renderFace(cmd, m_equirect_pipeline, *m_equirect_set, m_sky, 0, face, make_params(face, 0.0f));
		} else {
			renderFace(cmd, m_sky_pipeline, nullptr, m_sky, 0, face, make_params(face, 0.0f));
		}
	}
	m_sky.transition(
	    cmd, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, vk::PipelineStageFlagBits::eFragmentShader
	);

	// --- Irradiance: cosine convolution of the sky ---
	m_irradiance.transition(
	    cmd,
	    vk::ImageLayout::eColorAttachmentOptimal,
	    vk::AccessFlagBits::eColorAttachmentWrite,
	    vk::PipelineStageFlagBits::eColorAttachmentOutput
	);
	for (uint32_t face = 0; face < 6; ++face) {
		renderFace(cmd, m_irradiance_pipeline, *m_sky_source_set, m_irradiance, 0, face, make_params(face, 0.0f));
	}
	m_irradiance.transition(
	    cmd, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, vk::PipelineStageFlagBits::eFragmentShader
	);

	// --- Prefiltered radiance: one mip per roughness level ---
	m_prefiltered.transition(
	    cmd,
	    vk::ImageLayout::eColorAttachmentOptimal,
	    vk::AccessFlagBits::eColorAttachmentWrite,
	    vk::PipelineStageFlagBits::eColorAttachmentOutput
	);
	for (uint32_t mip = 0; mip < m_prefiltered.mipLevels(); ++mip) {
		// Mip 0 is a mirror, the last is fully rough; the shader maps a material's roughness back onto this
		const float roughness =
		    m_prefiltered.mipLevels() > 1 ? static_cast<float>(mip) / static_cast<float>(m_prefiltered.mipLevels() - 1) : 0.0f;
		for (uint32_t face = 0; face < 6; ++face) {
			renderFace(cmd, m_prefilter_pipeline, *m_sky_source_set, m_prefiltered, mip, face, make_params(face, roughness));
		}
	}
	m_prefiltered.transition(
	    cmd, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, vk::PipelineStageFlagBits::eFragmentShader
	);

	TOAST_INFO(
	    "Render",
	    "Environment precomputed: sky {}px, irradiance {}px, prefiltered {}px x {} roughness levels",
	    m_sky.size(),
	    m_irradiance.size(),
	    m_prefiltered.size(),
	    m_prefiltered.mipLevels()
	);
}

void EnvironmentPass::record(vk::CommandBuffer cmd, uint32_t frame_index, uint32_t image_index) {
	(void)frame_index;
	(void)image_index;

	if (!m_has_data || !isEnabled() || !m_skybox_pipeline.isReady() || !*m_sky_source_set) {
		return;
	}

	const auto* frame = VulkanRenderer::instance->renderingFrame();
	if (frame == nullptr) {
		return;
	}

	// Camera basis in world space. The view matrix maps world to view, so its inverse's columns are the axes
	// the camera looks along - and Vulkan's view space looks down -Z, hence the negated third column
	const glm::mat4 inverse_view = glm::inverse(frame->frame_data.view);
	const glm::vec3 right(inverse_view[0]);
	const glm::vec3 up(inverse_view[1]);
	const glm::vec3 forward(-glm::vec3(inverse_view[2]));

	// Recovered from the projection rather than passed alongside it, so a camera whose FOV or aspect changes
	// needs nothing plumbed through: [1][1] is 1/tan(fovY/2) and [0][0] folds in the aspect ratio.
	// Magnitudes only - Camera negates [1][1] for Vulkan's downward clip-space y, and that flip is applied
	// deliberately below rather than being inherited here (inheriting it renders the sky upside down)
	const float proj_yy = std::abs(frame->frame_data.projection[1][1]);
	const float proj_xx = std::abs(frame->frame_data.projection[0][0]);
	const float tan_half_fov_y = proj_yy != 0.0f ? 1.0f / proj_yy : 1.0f;
	const float tan_half_fov_x = proj_xx != 0.0f ? 1.0f / proj_xx : tan_half_fov_y;

	Params params {};
	params.face_right = glm::vec4(right * tan_half_fov_x, 0.0f);
	// Negated: the triangle's y follows clip space, which points down in Vulkan, so the top of the screen is
	// y = -1 and has to resolve to world up
	params.face_up = glm::vec4(up * -tan_half_fov_y, 0.0f);
	params.face_forward = glm::vec4(forward, 0.0f);
	params.roughness = 0.0f;
	params.intensity = m_intensity;

	const vk::Extent2D extent {
	  static_cast<uint32_t>(frame->viewport_extent.x),
	  static_cast<uint32_t>(frame->viewport_extent.y),
	};
	if (extent.width == 0 || extent.height == 0) {
		return;
	}

	// Already inside the scene's rendering scope - the pass list records between one beginRendering and its
	// endRendering, so this only sets its own state and draws
	const vk::Viewport viewport(0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f);
	cmd.setViewport(0, std::array {viewport});
	cmd.setScissor(0, std::array {vk::Rect2D({0, 0}, extent)});

	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, m_skybox_pipeline.getPipeline());
	cmd.bindDescriptorSets(
	    vk::PipelineBindPoint::eGraphics,
	    *m_shader_layout.getPipelineLayout(),
	    0,
	    std::array<vk::DescriptorSet, 1> {*m_sky_source_set},
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
}

auto EnvironmentPass::getIrradianceView() const -> vk::ImageView {
	return m_irradiance.cubeView();
}

auto EnvironmentPass::getPrefilteredView() const -> vk::ImageView {
	return m_prefiltered.cubeView();
}

}
