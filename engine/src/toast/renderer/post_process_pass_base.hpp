/// @file post_process_pass_base.hpp
/// @author dario
/// @date 03/08/2026

#pragma once

#include "vulkan_common.hpp"

#include <atomic>
#include <string_view>

namespace renderer {

/**
 * @brief A full-screen pass that transforms the rendered scene on its way to the screen
 *
 * Passes run in registration order, each reading the previous one's view and returning its own. None writes
 * the output image - the renderer copies the chain's final result there. That uniformity is what makes a pass
 * reorderable by registration alone, disableable without breaking the chain, and the all-disabled case show
 * the raw scene rather than black
 *
 * Each opens its own rendering scope rather than sharing one, which is what lets a single pass render several
 * times into several targets - bloom's mip chain is why the interface is shaped this way
 */
class IPostProcessPass {
public:
	virtual ~IPostProcessPass() = default;

	/// @brief Records this pass's work into @p cmd, outside any rendering scope
	/// @returns this pass's result, which becomes the next pass's input. Returning @p source_view unchanged
	///          means "I did not alter the image"
	virtual auto record(vk::CommandBuffer cmd, uint32_t frame_index, vk::ImageView source_view) -> vk::ImageView = 0;

	/// @brief Called when the output is resized, for passes owning extent-sized targets
	virtual void onResize(vk::Extent2D extent) { }

	/// @returns name shown in the editor's pass visibility popup
	[[nodiscard]]
	virtual auto name() const -> std::string_view {
		return "PostProcess";
	}

	/// Disabled passes are skipped and pass their input straight through
	void setEnabled(bool enabled) noexcept { m_enabled.store(enabled, std::memory_order_relaxed); }

	[[nodiscard]]
	auto isEnabled() const noexcept -> bool {
		return m_enabled.load(std::memory_order_relaxed);
	}

private:
	std::atomic_bool m_enabled {true};
};

}
