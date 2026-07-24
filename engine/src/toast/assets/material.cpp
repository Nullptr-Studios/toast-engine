#include "material.hpp"

#include "assets.hpp"
#include "texture.hpp"

#include <algorithm>
#include <format>
#include <toast/renderer/vulkan_renderer.hpp>
#include <toast/renderer/vulkan_sampler.hpp>

namespace assets {

namespace {

auto readAssetUID(const DataValue& v) -> toast::UID {
	if (v.type() == DataType::asset_t || v.type() == DataType::node_t) {
		return static_cast<toast::UID>(v);
	}
	if (v.type() == DataType::string_t) {
		const auto s = static_cast<std::string>(v);
		if (s.size() == 11) {
			return {toast::UID::fromString(s)};
		}
	}
	return toast::UID(uint64_t {0});
}

auto readColor(const DataValue& v) -> glm::vec4 {
	if (v.type() == DataType::color4_t) {
		return static_cast<glm::vec4>(v);
	}
	if (!v.isArray()) {
		return glm::vec4(1.0f);
	}
	glm::vec4 result(1.0f);
	const size_t count = std::min<size_t>(v.size(), 4);
	for (size_t i = 0; i < count; ++i) {
		const DataValue& elem = v[i];
		if (auto d = elem.value<double>()) {
			result[static_cast<int>(i)] = static_cast<float>(*d);
		} else if (auto n = elem.value<int64_t>()) {
			result[static_cast<int>(i)] = static_cast<float>(*n);
		}
	}
	return result;
}

auto readFloat(const DataValue& v, float fallback) -> float {
	if (auto d = v.value<double>()) {
		return static_cast<float>(*d);
	}
	if (auto n = v.value<int64_t>()) {
		return static_cast<float>(*n);
	}
	return fallback;
}

}

Material::Material(const toml::table& table, Handle<Schema> schema) : Data(table, std::move(schema), Data::keep_all_keys) { }

Material::~Material() = default;

auto Material::serialize(SaveMode mode) const -> std::vector<uint8_t> {
	return Data::serialize(mode);
}

auto Material::albedoMap() const -> Handle<Texture> {
	if (!m_root.contains("albedo_map")) {
		m_albedo_handle = {};
		return m_albedo_handle;
	}
	const toast::UID uid = readAssetUID(m_root["albedo_map"]);
	if (uid.data() == 0) {
		m_albedo_handle = {};
		return m_albedo_handle;
	}
	if (m_albedo_handle.uid().data() != uid.data()) {
		m_albedo_handle = assets::load<Texture>(uid);
	}
	return m_albedo_handle;
}

auto Material::normalMap() const -> Handle<Texture> {
	if (!m_root.contains("normal_map")) {
		m_normal_handle = {};
		return m_normal_handle;
	}
	const toast::UID uid = readAssetUID(m_root["normal_map"]);
	if (uid.data() == 0) {
		m_normal_handle = {};
		return m_normal_handle;
	}
	if (m_normal_handle.uid().data() != uid.data()) {
		m_normal_handle = assets::load<Texture>(uid);
	}
	return m_normal_handle;
}

auto Material::color() const -> glm::vec4 {
	if (!m_root.contains("color")) {
		return glm::vec4(1.0f);
	}
	return readColor(m_root["color"]);
}

auto Material::metallic() const -> float {
	if (!m_root.contains("metallic")) {
		return 0.0f;
	}
	return readFloat(m_root["metallic"], 0.0f);
}

auto Material::roughness() const -> float {
	if (!m_root.contains("roughness")) {
		return 0.5f;
	}
	return readFloat(m_root["roughness"], 0.5f);
}

void Material::resolveTextureHandles() {
	const Handle<Texture> albedo_map = albedoMap();
	if (!albedo_map.hasValue()) {
		return;
	}

	// This is called every frame
	if (m_sampler_ready) {
		return;
	}

	const auto& core = renderer::getCore();

	vk::SamplerCreateInfo sampler_info {};

	sampler_info.magFilter = vk::Filter::eLinear;
	sampler_info.minFilter = vk::Filter::eLinear;
	// Clamp, not repeat: these are unique-UV asset textures (albedo/normal), not tiling patterns. Repeat
	// wrap mode plus linear filtering blends texels straddling a UV seam with texels from the opposite edge
	// of the texture, producing a visible seam - most noticeable on normal maps since it directly corrupts
	// the lighting direction right at the seam
	sampler_info.addressModeU = vk::SamplerAddressMode::eClampToEdge;
	sampler_info.addressModeV = vk::SamplerAddressMode::eClampToEdge;
	sampler_info.addressModeW = vk::SamplerAddressMode::eClampToEdge;
	sampler_info.anisotropyEnable = core.supportsSamplerAnisotropy() ? vk::True : vk::False;
	sampler_info.maxAnisotropy = core.supportsSamplerAnisotropy() ? std::min(16.0f, core.maxSamplerAnisotropy()) : 1.0f;
	sampler_info.compareEnable = vk::False;
	sampler_info.compareOp = vk::CompareOp::eAlways;
	sampler_info.unnormalizedCoordinates = vk::False;

	sampler_info.mipmapMode = vk::SamplerMipmapMode::eLinear;
	sampler_info.mipLodBias = 0.0f;
	sampler_info.minLod = 0.0f;
	sampler_info.maxLod = 0.0f;

	m_albedo_sampler = std::make_unique<renderer::VulkanSampler>(
	    core, sampler_info, std::format("Material AlbedoSampler ({})", albedo_map.path())
	);

	m_sampler_ready = true;
}

auto Material::albedoSampler() const -> VkSampler {
	return m_albedo_sampler ? m_albedo_sampler->handle() : VK_NULL_HANDLE;
}

}
