/// @file IOutputTarget.hpp
/// @author dario
/// @date 16/05/2026

#pragma once

#include "vulkan_common.hpp"

#include <cstdint>

namespace renderer {

class IOutputTarget {
public:
	virtual ~IOutputTarget() = default;

	[[nodiscard]]
	virtual auto getExtent() const -> vk::Extent2D = 0;

	[[nodiscard]]
	virtual auto getColorFormat() const -> vk::Format = 0;

	[[nodiscard]]
	virtual auto getImageCount() const -> uint32_t = 0;

	[[nodiscard]]
	virtual auto getColorImage(uint32_t index) const -> const vk::Image& = 0;

	[[nodiscard]]
	virtual auto getColorAttachment(uint32_t index) const -> const vk::raii::ImageView& = 0;

	/// @brief Acquires the next image: a real swapchain acquire, or an index rotation for off-screen targets
	virtual auto acquireNextImage(uint64_t timeout, vk::Semaphore image_available, vk::Fence in_flight_fence)
	    -> vk::ResultValue<uint32_t> = 0;

	/// @brief Presents @p image_index. Off-screen targets publish via onImageRenderComplete() instead
	virtual auto present(uint32_t image_index, vk::Semaphore render_finished) -> vk::Result = 0;

	[[nodiscard]]
	virtual auto usesAcquirePresentSemaphores() const -> bool {
		return true;
	}

	/// @brief Transitions @p image_index out of `eColorAttachmentOptimal` into whatever the target needs next
	virtual void recordFinalize(vk::CommandBuffer command_buffer, uint32_t image_index) = 0;

	/// Called once the GPU work for @p image_index has completed; off-screen targets publish the frame here
	virtual void onImageRenderComplete(uint32_t image_index) { (void)image_index; }

	virtual void recreate(vk::Extent2D extent) = 0;
};

}    // namespace renderer
