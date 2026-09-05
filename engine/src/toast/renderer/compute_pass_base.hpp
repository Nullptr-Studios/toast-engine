/// @file compute_pass_base.hpp
/// @author dario
/// @date 18/07/2026

#pragma once

#include "vulkan_common.hpp"

/// @brief Interface for custom compute passes
///
/// Symmetric to IRenderPass, minus the swapchain image compute has no notion of. Implementors read the
/// current frame through VulkanRenderer::instance->renderingFrame()
class IComputePass {
public:
	virtual ~IComputePass() = default;

	virtual void update(uint32_t frame_index, float dt) { }

	virtual void dispatch(vk::CommandBuffer cmd, uint32_t frame_index) = 0;
};
