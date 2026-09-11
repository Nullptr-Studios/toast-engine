/**
 * @file post_process_target.hpp
 * @author dario
 * @date 14/08/2026
 */

#pragma once

#include "vulkan_common.hpp"

#include <optional>
#include <string>
#include <string_view>

namespace renderer {
class VulkanCore;

/**
 * @brief The offscreen image a full-screen post pass renders into, plus the layout tracking it needs
 *
 * Every `IPostProcessPass` has the same shape: read the previous view, draw one full-screen triangle into an
 * image of its own, hand that view on. The image, view, extent, layout field and the two barriers around the
 * scope were copied byte for byte into five passes - five places for a barrier to be wrong, and a wrong
 * layout does not fail at the call, it corrupts something later
 *
 * @note The layout is render-thread state. Not synchronised - touch it only from the recording thread
 */
class PostProcessTarget {
public:
	/// @brief (Re)creates the image at @p extent, discarding any previous one
	///
	/// Called from the pass constructor and again on every resize. The layout resets to `eUndefined`, which is
	/// what the first barrier in beginScope() expects to transition away from
	void create(const VulkanCore& core, vk::Extent2D extent, vk::Format format, std::string_view debug_name);

	/// @brief Barriers into a colour attachment, opens the rendering scope, and sets the full-target viewport
	///
	/// The source access mask is conditional on the current layout rather than hardcoded: the first frame
	/// transitions from `eUndefined` with nothing to wait on, and every frame after it from
	/// `eShaderReadOnlyOptimal` with the previous pass's sampling to finish first
	void beginScope(vk::CommandBuffer cmd) const;

	/// @brief Closes the scope and barriers back to `eShaderReadOnlyOptimal` for whatever samples this next
	void endScope(vk::CommandBuffer cmd) const;

	/// @returns the view to hand the next pass in the chain, or null when nothing was created
	[[nodiscard]]
	auto view() const -> vk::ImageView {
		return m_view.has_value() ? **m_view : vk::ImageView {};
	}

	[[nodiscard]]
	auto extent() const noexcept -> vk::Extent2D {
		return m_extent;
	}

	[[nodiscard]]
	auto isReady() const noexcept -> bool {
		return m_image.has_value() && m_view.has_value();
	}

private:
	std::optional<vma::raii::Image> m_image;
	std::optional<vk::raii::ImageView> m_view;
	vk::Extent2D m_extent {};

	/// Render-thread-only. Mutable because the scope calls are logically const - they record commands and do
	/// not change what the target *is* - while still having to track what the recorded barriers left behind
	mutable vk::ImageLayout m_layout = vk::ImageLayout::eUndefined;
};

}
