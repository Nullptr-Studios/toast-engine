/**
 * @file cubemap_target.hpp
 * @author dario
 * @date 14/08/2026
 */

#pragma once

#include "vulkan_common.hpp"

#include <optional>
#include <string_view>
#include <vector>

namespace renderer {
class VulkanCore;

/**
 * @brief A cubemap rendered into one face at a time, plus the layout tracking that needs
 *
 * Shared by `EnvironmentPass` and `ReflectionProbePass`, which both generate six faces and convolve them into
 * a roughness chain. The per-face views exist because dynamic rendering targets one layer at a time.
 * `eCubeCompatible` is the plain cube feature, not `imageCubeArray` - which this device does not enable, and
 * is also why point-light shadows use a 2D array. How a face is drawn is left to the passes; storage and
 * layout are the shared part
 *
 * @note The layout is render-thread state, like PostProcessTarget's.
 */
class CubemapTarget {
public:
	/// @brief (Re)creates a @p size cubemap with @p mip_levels, discarding any previous one
	void create(
	    const VulkanCore& core, vk::Format format, uint32_t size, uint32_t mip_levels, std::string_view debug_name,
	    vk::ImageUsageFlags extra_usage = {}, bool with_face_views = true
	);

	/// @brief Barriers the whole cube - every mip, every face - into @p new_layout, or does nothing if already there
	void
	    transition(vk::CommandBuffer cmd, vk::ImageLayout new_layout, vk::AccessFlags dst_access, vk::PipelineStageFlags dst_stage);

	/// @brief Records that something outside this type transitioned the image
	///
	/// For the one path that has to barrier by hand: a probe's disk readback moves the whole cube to
	/// eTransferSrcOptimal and back on its own single-use command buffer, and the tracked layout has to follow
	/// or the next transition() computes its source state from a stale value
	void setLayout(vk::ImageLayout layout) noexcept;

	/// @returns the single-mip, single-layer attachment view for one face, or null when out of range
	[[nodiscard]]
	auto faceView(uint32_t mip, uint32_t face) const -> vk::ImageView;

	/// @returns the cube view everything downstream samples, or null when nothing was created
	[[nodiscard]]
	auto cubeView() const -> vk::ImageView {
		return m_cube_view.has_value() ? **m_cube_view : vk::ImageView {};
	}

	[[nodiscard]]
	auto image() const -> vk::Image {
		return m_image.has_value() ? **m_image : vk::Image {};
	}

	[[nodiscard]]
	auto size() const noexcept -> uint32_t {
		return m_size;
	}

	/// @returns edge length of @p mip, floored at 1 - the size a face is rendered at in the roughness chain
	[[nodiscard]]
	auto mipSize(uint32_t mip) const noexcept -> uint32_t {
		return std::max(m_size >> mip, 1u);
	}

	[[nodiscard]]
	auto mipLevels() const noexcept -> uint32_t {
		return m_mip_levels;
	}

	[[nodiscard]]
	auto layout() const noexcept -> vk::ImageLayout {
		return m_layout;
	}

	[[nodiscard]]
	auto isReady() const noexcept -> bool {
		return m_image.has_value() && m_cube_view.has_value();
	}

	/// Faces of a cube, and the array layer count every cubemap image uses
	static constexpr uint32_t k_faces = 6;

private:
	std::optional<vma::raii::Image> m_image;
	std::optional<vk::raii::ImageView> m_cube_view;
	/// One per (mip, face), mip-major - the order faceView() indexes
	std::vector<vk::raii::ImageView> m_face_views;

	uint32_t m_size = 0;
	uint32_t m_mip_levels = 1;
	vk::ImageLayout m_layout = vk::ImageLayout::eUndefined;
};

}
