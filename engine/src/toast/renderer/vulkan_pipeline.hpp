/// @file VulkanPipeline.hpp
/// @author dario
/// @date 16/05/2026

#pragma once

#include "vulkan_common.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace renderer {

class VulkanCore;

/**
 * @class VulkanPipeline
 * @brief Wraps Vulkan graphics and compute pipelines with shader compilation
 */
class VulkanPipeline {
public:
	enum class PipelineType : uint8_t {
		graphics,
		compute
	};

	enum class BlendPreset : uint8_t {
		none,             ///< blending disabled
		alpha,            ///< srcAlpha, 1-srcAlpha
		premultiplied,    ///< one, 1-srcAlpha (premultiplied-alpha source)
		additive,         ///< one, one
		multiply,         ///< dstColor, zero
	};

	struct Config {
		PipelineType pipeline_type = PipelineType::graphics;
		std::string debug_name;

		// Render state
		vk::Format color_format = vk::Format::eUndefined;
		std::optional<vk::Format> depth_format;
		vk::Extent2D extent;

		/// @brief Colour attachments beyond the first, in binding order
		///
		/// Not opt-in: every pipeline in a scope must declare the same list, even the ones writing nothing.
		/// Blending never applies - a blended normal describes neither surface
		std::vector<vk::Format> extra_color_formats;

		/// @brief Whether the fragment shader actually writes the extra attachments
		///
		/// Explicit because most pipelines in a G-buffer scope contribute nothing, and declaring without
		/// writing leaves undefined values there
		bool write_extra_color = false;

		/// @brief Vertex-stage-only pipeline writing depth and nothing else
		///
		/// color_format stays Undefined and fragment_entry is never looked up - a shadow-map shader has none
		bool depth_only = false;

		/// @brief Multiview mask; one bit per simultaneous view. Must equal the `viewMask` of every
		/// `vk::RenderingInfo` this pipeline is recorded into - Vulkan compares them exactly, which is why
		/// ShadowPass builds one variant per mask
		///
		/// @note 0 and 1 differ: 0 means not multiview at all and leaves `SV_ViewID` undefined; 1 is a
		/// multiview pass with one view, where it reads 0
		uint32_t view_mask = 0;

		// Shader data TODO: Move this into own shader class
		std::vector<std::byte> shader_spirv;
		std::string vertex_entry = "vertexMain";
		std::string fragment_entry = "fragmentMain";
		std::string compute_entry = "computeMain";

		// Layouts are now provided from the outside
		vk::PipelineLayout pipeline_layout = nullptr;

		// Vertex input state; every graphics pipeline must set this explicitly. Most passes bind a single
		// vertex stream; a skinned mesh pipeline adds a second binding (SkinVertex) alongside the base one
		std::vector<vk::VertexInputBindingDescription> vertex_bindings;
		std::vector<vk::VertexInputAttributeDescription> vertex_attributes;

		vk::PrimitiveTopology topology = vk::PrimitiveTopology::eTriangleList;

		// Raster state
		vk::CullModeFlags cull_mode = vk::CullModeFlagBits::eBack;
		vk::FrontFace front_face = vk::FrontFace::eCounterClockwise;    // Note: Counter-clockwise due to inverted projection matrix

		// Depth/blend state
		bool depth_test = true;
		bool depth_write = true;
		/// Depth is cleared to 1.0 (far), so eLess is right for geometry. A pass that draws *at* the far plane -
		/// a skybox filling whatever the scene left empty - needs eLessOrEqual, or it fails against the clear
		/// value everywhere and never appears
		vk::CompareOp depth_compare = vk::CompareOp::eLess;

		/// @brief Rasterizer depth bias, constant and slope-proportional. Zero on both disables it
		///
		/// Where shadow acne should be fought: it offsets along the polygon's own depth slope, so nothing
		/// shifts laterally. A normal offset in the shader can only move sideways, detaching the shadow
		float depth_bias_constant = 0.0f;
		float depth_bias_slope = 0.0f;
		bool blend_enable = false;    // legacy
		BlendPreset blend_preset = BlendPreset::none;
	};

	VulkanPipeline() = default;
	explicit VulkanPipeline(const VulkanCore& core, const Config& config);
	~VulkanPipeline() = default;

	VulkanPipeline(const VulkanPipeline&) = delete;
	auto operator=(const VulkanPipeline&) -> VulkanPipeline& = delete;
	VulkanPipeline(VulkanPipeline&&) = delete;
	auto operator=(VulkanPipeline&&) -> VulkanPipeline& = delete;

	auto rebuild(const VulkanCore& core, const Config& config) -> void;
	auto reset() -> void;

	[[nodiscard]]
	auto isReady() const -> bool {
		return m_pipeline != nullptr;
	}

	[[nodiscard]]
	auto getPipeline() const -> const vk::raii::Pipeline& {
		return m_pipeline;
	}

private:
	std::optional<vk::raii::ShaderModule> m_shader_module;
	vk::raii::Pipeline m_pipeline = nullptr;
};

}    // namespace renderer
