/// @file compute_pass_base.hpp
/// @author dario
/// @date 18/07/2026.

#pragma once

#include "vulkan_common.hpp"

/**
 * @brief Interface for custom compute passes
 *
 * Symmetric to IRenderPass, but dispatch() isn't tied to a swapchain image (compute doesn't have one, so
 * no imageIndex) and doesn't take a RenderFrame& - implementors fetch the current frame via
 * VulkanRenderer::instance->renderingFrame(), same as MeshPass/DebugPass already do
 */
class IComputePass {
public:
	virtual ~IComputePass() = default;

	/**
	 * @brief Updates the compute pass state for the current frame
	 * @param frame_index The index of the current frame in flight
	 * @param dt The delta time since the last frame
	 */
	virtual void update(uint32_t frame_index, float dt) { }

	/**
	 * @brief Records the compute pass's dispatch(es) for the current frame
	 * @param cmd The command buffer to record commands into
	 * @param frame_index The index of the current frame in flight
	 */
	virtual void dispatch(vk::CommandBuffer cmd, uint32_t frame_index) = 0;
};
