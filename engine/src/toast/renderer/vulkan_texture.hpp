/// @file vulkan_texture.hpp
/// @author dario
/// @date 6/28/2026

#pragma once
#include "ktx.h"
#include "vulkan_common.hpp"
#include "vulkan_resource_base.hpp"

#include <string>
#include <string_view>

namespace renderer {

class VulkanTexture : public IVulkanResource {
public:
	VulkanTexture() = default;

	struct Params {
		vk::Format format;
		vk::Extent3D extent;
		uint32_t mip_levels = 1;
		uint32_t layer_count = 1;
		bool is_cubemap = false;
	};

	void create(const VulkanCore& core, Params params, std::string_view debug_name = {});
	void destroy();

	[[nodiscard]]
	auto getImage() const -> vk::Image {
		return **m_image;
	}

	[[nodiscard]]
	auto getView() const -> vk::ImageView {
		return *m_image_view;
	}

	/// @returns the VkFormat the KTX2 carried, which is what decides whether the GPU applies an sRGB decode
	/// on sample - see MaterialPass's colour-space check
	[[nodiscard]]
	auto getFormat() const -> vk::Format {
		return m_params.format;
	}

private:
	std::optional<vma::raii::Image> m_image;
	vk::raii::ImageView m_image_view = nullptr;
	Params m_params;
};

class TextureUpload : public PendingResourceUpload {
public:
	/// the encoded data has no reader once the GPU image exists, so the asset
	/// hands them over rather than keeping a copy alive for the process lifetime
	TextureUpload(VulkanTexture& texture, std::vector<uint8_t> data, std::string_view debug_name = {})
	    : m_texture(&texture),
	      m_data(std::move(data)),
	      m_debug_name(debug_name) { }

	~TextureUpload() override {
		if (m_ktx_texture) {
			ktxTexture2_Destroy(m_ktx_texture);
		}
	}

	void build(const VulkanCore& core) override;
	void record(vk::CommandBuffer cmd) override;

	auto resource() -> IVulkanResource* override { return m_texture; }

private:
	VulkanTexture* m_texture;
	std::vector<uint8_t> m_data;
	std::string m_debug_name;

	ktxTexture2* m_ktx_texture = nullptr;
	vma::raii::Buffer m_staging_buffer = nullptr;
	std::vector<vk::BufferImageCopy> m_copy_regions;
	VulkanTexture::Params m_tex_params {};
};

class RawTextureUpload : public PendingResourceUpload {
public:
	RawTextureUpload(
	    VulkanTexture& texture, std::vector<uint8_t> data, uint32_t width, uint32_t height, vk::Format format,
	    std::string_view debug_name = {}
	)
	    : m_texture(&texture),
	      m_data(std::move(data)),
	      m_width(width),
	      m_height(height),
	      m_format(format),
	      m_debug_name(debug_name) { }

	void build(const VulkanCore& core) override;
	void record(vk::CommandBuffer cmd) override;

	auto resource() -> IVulkanResource* override { return m_texture; }

private:
	VulkanTexture* m_texture = nullptr;
	std::vector<uint8_t> m_data;
	uint32_t m_width = 0;
	uint32_t m_height = 0;
	vk::Format m_format = vk::Format::eUndefined;
	std::string m_debug_name;
	vma::raii::Buffer m_staging_buffer = nullptr;
};

/// @brief Decodes and uploads a KTX2 texture on the calling thread, returning once the GPU has the data
///
/// For renderer-owned textures that have to exist before the first frame. The asynchronous path is the right
/// one for scene assets, but it needs a running render thread to flush its batches and a fully built
/// VulkanRenderer to reclaim them - neither of which is true from inside the constructor
///
/// @returns false when the bytes could not be decoded or the device rejected them; @p texture is left
///          unready and marked with the reason
auto uploadTextureSync(const VulkanCore& core, VulkanTexture& texture, std::vector<uint8_t> data, std::string_view debug_name)
    -> bool;

}
