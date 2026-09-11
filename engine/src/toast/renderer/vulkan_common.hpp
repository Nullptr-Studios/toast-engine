#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <vulkan-memory-allocator-hpp/vk_mem_alloc_raii.hpp>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_raii.hpp>

struct FrameResources {
	std::optional<vma::raii::Buffer> staging_buffer;
	std::optional<vma::raii::Buffer> gpu_buffer;
	vk::raii::DescriptorSet descriptor_set = nullptr;
};

namespace renderer {

/// @brief The probe grid a set of irradiance coefficients was baked for
///
/// Here rather than on ReflectionProbePass, which would close an include cycle. Compared on load: a volume
/// that moved since interpolates probes from positions they were never captured at, and that reads as light
/// leaking rather than as stale data
struct ShGridKey {
	glm::vec3 min_corner {0.0f};
	glm::vec3 extents {0.0f};
	glm::uvec3 counts {0};
};

/// @brief Format of the world stage's geometry buffer: world normal in .xyz, perceptual roughness in .w
///
/// Here rather than on VulkanRenderer because every world-stage pipeline declares it, including the ones
/// that write nothing to it
inline constexpr vk::Format k_scene_normal_format = vk::Format::eR16G16B16A16Sfloat;

/// @brief Format of the indirect-diffuse attachment: the ambient/IBL diffuse term, before it is summed in
///
/// Forward shading sums everything into one colour a post pass cannot take apart again, and occlusion has
/// to scale this term alone: `final = color - indirect + indirect*ao`. Packed rather than half-float
/// because indirect radiance is non-negative and low-frequency
inline constexpr vk::Format k_scene_indirect_format = vk::Format::eB10G11R11UfloatPack32;

/// @brief Colour attachments beyond the scene colour that the world stage renders into
///
/// A pipeline's attachment list must match its scope's, and omitting this fails at draw time naming the
/// pipeline rather than the omission
///
/// @warning **RenderStage::world only.** The overlay scope has one attachment, so handing them this is the
///          same mismatch in reverse. Check `stage()`, not whether the pass looks like scene geometry
[[nodiscard]]
inline auto worldStageExtraColorFormats() -> std::vector<vk::Format> {
	return {k_scene_normal_format, k_scene_indirect_format};
}

/// @brief The whole of a single-mip, single-layer colour image
[[nodiscard]]
inline auto colorSubresourceRange() -> vk::ImageSubresourceRange {
	return {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
}

/// @brief Submits @p cmd on @p queue and blocks until it completes
///
/// Blocking is the point: every caller runs at construction or on an explicit user action, never per frame
inline void submitAndWait(const vk::raii::Device& device, vk::Queue queue, vk::CommandBuffer cmd) {
	const vk::raii::Fence fence(device, vk::FenceCreateInfo {});

	vk::SubmitInfo submit {};
	submit.commandBufferCount = 1;
	submit.pCommandBuffers = &cmd;
	queue.submit(submit, *fence);

	std::ignore = device.waitForFences(*fence, VK_TRUE, std::numeric_limits<uint64_t>::max());
}

/// @brief Barriers @p image from undefined into a transfer destination, ready to be written
///
/// Pairs with recordTransferDstToShaderRead() - together, the one-shot upload transition
inline void recordUndefinedToTransferDst(vk::CommandBuffer cmd, vk::Image image, const vk::ImageSubresourceRange& range) {
	vk::ImageMemoryBarrier to_dst {};
	to_dst.oldLayout = vk::ImageLayout::eUndefined;
	to_dst.newLayout = vk::ImageLayout::eTransferDstOptimal;
	to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_dst.image = image;
	to_dst.subresourceRange = range;
	to_dst.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, nullptr, nullptr, to_dst);
}

/// @brief Barriers @p image from a written transfer destination into a shader-readable layout
inline void recordTransferDstToShaderRead(vk::CommandBuffer cmd, vk::Image image, const vk::ImageSubresourceRange& range) {
	vk::ImageMemoryBarrier to_read {};
	to_read.oldLayout = vk::ImageLayout::eTransferDstOptimal;
	to_read.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
	to_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_read.image = image;
	to_read.subresourceRange = range;
	to_read.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	to_read.dstAccessMask = vk::AccessFlagBits::eShaderRead;
	cmd.pipelineBarrier(
	    vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, nullptr, nullptr, to_read
	);
}

/// @brief Sampler for reading a full-screen target: bilinear, clamped, no mip chain
///
/// Clamped so an edge tap cannot wrap to the opposite side; linear because these techniques work by
/// sampling *between* texels
[[nodiscard]]
inline auto linearClampSamplerInfo() -> vk::SamplerCreateInfo {
	vk::SamplerCreateInfo sampler_ci {};
	sampler_ci.magFilter = vk::Filter::eLinear;
	sampler_ci.minFilter = vk::Filter::eLinear;
	sampler_ci.addressModeU = vk::SamplerAddressMode::eClampToEdge;
	sampler_ci.addressModeV = vk::SamplerAddressMode::eClampToEdge;
	sampler_ci.addressModeW = vk::SamplerAddressMode::eClampToEdge;
	sampler_ci.mipmapMode = vk::SamplerMipmapMode::eNearest;
	return sampler_ci;
}

/// @brief Point-sampled counterpart of linearClampSamplerInfo()
///
/// For depth and the geometry buffer: a value filtered across a silhouette describes no surface that
/// exists, so a position reconstructed from it starts the ray in mid-air
[[nodiscard]]
inline auto nearestClampSamplerInfo() -> vk::SamplerCreateInfo {
	auto sampler_ci = linearClampSamplerInfo();
	sampler_ci.magFilter = vk::Filter::eNearest;
	sampler_ci.minFilter = vk::Filter::eNearest;
	return sampler_ci;
}

/// @brief linearClampSamplerInfo() with trilinear filtering across @p max_lod mips
///
/// For the roughness chains, where roughness picks the mip and would step visibly at each stored level
/// without blending
[[nodiscard]]
inline auto linearClampMippedSamplerInfo(float max_lod) -> vk::SamplerCreateInfo {
	auto sampler_ci = linearClampSamplerInfo();
	sampler_ci.mipmapMode = vk::SamplerMipmapMode::eLinear;
	sampler_ci.maxLod = max_lod;
	return sampler_ci;
}

/// @brief Create info for a 2D colour image that is rendered into and then sampled
///
/// @p usage is added to the attachment/sampled pair rather than replacing it
[[nodiscard]]
inline auto colorTargetImageInfo(vk::Extent2D extent, vk::Format format, vk::ImageUsageFlags usage = {}) -> vk::ImageCreateInfo {
	vk::ImageCreateInfo image_ci {};
	image_ci.imageType = vk::ImageType::e2D;
	image_ci.format = format;
	image_ci.extent = vk::Extent3D {extent.width, extent.height, 1};
	image_ci.mipLevels = 1;
	image_ci.arrayLayers = 1;
	image_ci.samples = vk::SampleCountFlagBits::e1;
	image_ci.tiling = vk::ImageTiling::eOptimal;
	image_ci.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | usage;
	image_ci.sharingMode = vk::SharingMode::eExclusive;
	image_ci.initialLayout = vk::ImageLayout::eUndefined;
	return image_ci;
}

}
