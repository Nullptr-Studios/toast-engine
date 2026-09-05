/// @file VulkanCore.hpp
/// @author dario
/// @date 14/05/2026

#pragma once

#include "vulkan_common.hpp"

#include <algorithm>
#include <external/inc/renderdoc/renderdoc_app.h>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

namespace renderer {

/// @brief Which Nsight Graphics SDK activity (if any) got injected/initialized into this process. Only one
/// activity can be active per process, chosen once at startup via the TOAST_NSIGHT_MODE env var - see
/// VulkanCore::VulkanCore() for the selection logic
enum class NsightMode : uint8_t {
	none,
	graphics_capture,
	gpu_trace
};

/**
 * @brief Represents the suitability score of a Vulkan physical device based on various criteria
 */
struct DeviceScore {
	int total = 0;
	int device_type = 0;
	int memory = 0;
	int limits = 0;
	int features = 0;
	int extensions = 0;
	int vulkan_support = 0;
	int queue_support = 0;

	int graphics_idx = -1;
	int compute_idx = -1;
	int transfer_idx = -1;

	std::vector<std::string> missing_extensions;

	/// @brief Converts the device score to a string representation
	[[nodiscard]]
	auto toString() const noexcept -> std::string;
};

/// @brief Manages Vulkan instance, device initialization, and memory allocation
class VulkanCore {
public:
	VulkanCore(
	    std::span<const char* const> required_instance_extensions, std::span<const char* const> required_device_extensions = {}
	) noexcept;
	~VulkanCore() = default;

	// Prevent copying
	VulkanCore(const VulkanCore&) = delete;
	auto operator=(const VulkanCore&) -> VulkanCore& = delete;

	[[nodiscard]]
	auto getInstance() const noexcept -> const vk::raii::Instance& {
		return m_instance;
	}

	[[nodiscard]]
	auto getDevice() const noexcept -> const vk::raii::Device& {
		return m_device;
	}

	[[nodiscard]]
	auto getPhysicalDevice() const noexcept -> const vk::raii::PhysicalDevice& {
		return m_physical_device;
	}

	[[nodiscard]]
	auto getAllocator() const noexcept -> const vma::raii::Allocator& {
		return *m_allocator;
	}

	[[nodiscard]]
	auto getGraphicsQueueFamilyIndex() const noexcept -> uint32_t {
		return m_graphics_queue_family_index;
	}

	[[nodiscard]]
	auto getComputeQueueFamilyIndex() const noexcept -> uint32_t {
		return m_compute_queue_family_index;
	}

	[[nodiscard]]
	auto getTransferQueueFamilyIndex() const noexcept -> uint32_t {
		return m_transfer_queue_family_index;
	}

	[[nodiscard]]
	auto getGraphicsQueue() const noexcept -> vk::Queue {
		return m_graphics_queue;
	}

	[[nodiscard]]
	auto getComputeQueue() const noexcept -> vk::Queue {
		return m_compute_queue;
	}

	[[nodiscard]]
	auto getTransferQueue() const noexcept -> vk::Queue {
		return m_transfer_queue;
	}

	[[nodiscard]]
	auto getRenderDocAPI() const noexcept -> const RENDERDOC_API_1_6_0* {
		return rdoc_api;
	}

	/// @returns true when ray queries and acceleration structures are available
	///
	/// Every traceable feature keeps its raster path: RT replaces how something is computed, never whether
	/// it exists
	[[nodiscard]]
	auto isRayTracingSupported() const noexcept -> bool {
		return m_ray_tracing_supported;
	}

	/// @returns the device address of @p buffer
	[[nodiscard]]
	auto getBufferAddress(vk::Buffer buffer) const -> vk::DeviceAddress {
		vk::BufferDeviceAddressInfo info {};
		info.buffer = buffer;
		return m_device.getBufferAddress(info);
	}

	/// @returns bytes to allocate so @p needed bytes are reachable at an aligned address
	///
	/// One alignment of slack: the allocation's own address does not satisfy the stricter AS requirement, so
	/// the usable address is rounded up inside it
	[[nodiscard]]
	auto getScratchAllocationSize(vk::DeviceSize needed) const noexcept -> vk::DeviceSize {
		return needed + std::max(m_as_scratch_alignment, 1u);
	}

	/// @returns the first address inside @p buffer a build may use as scratch
	///
	/// Pair with getScratchAllocationSize(), which reserves the slack this rounds into. Getting it wrong
	/// corrupts the build and takes the device out later, rather than failing here
	[[nodiscard]]
	auto getAlignedScratchAddress(vk::Buffer buffer) const -> vk::DeviceAddress {
		const auto alignment = static_cast<vk::DeviceAddress>(std::max(m_as_scratch_alignment, 1u));
		return (getBufferAddress(buffer) + alignment - 1) & ~(alignment - 1);
	}

	/// @returns true when VK_EXT_frame_boundary is enabled, so submits can carry a frame delimiter
	///
	/// The editor never presents, so a graphics debugger has nothing to delimit frames by and can report
	/// seeing no API at all. This supplies the boundary instead
	[[nodiscard]]
	auto isFrameBoundarySupported() const noexcept -> bool {
		return m_frame_boundary_supported;
	}

	/// @brief Which Nsight Graphics activity (if any) is initialized for this process - none if no Nsight
	/// Graphics installation was found, or its SDK init failed
	[[nodiscard]]
	auto getNsightMode() const noexcept -> NsightMode {
		return m_nsight_mode;
	}

#if defined(_WIN32)
	/// @brief Lazily activates Nsight GPU Trace on first call; no-op afterwards and unless mode is gpu_trace
	///
	/// @warning Blocks until the Nsight Graphics host attaches - render thread's F12 handler only, never at
	/// startup, or it hangs the engine waiting for a host that may never come
	void activateNsightGpuTraceIfNeeded() const;
#endif

	/// @brief Whether validation layers are enabled
	[[nodiscard]]
	auto validationEnabled() const noexcept -> bool {
		return m_validation_enabled;
	}

	/// @brief Whether the selected device supports anisotropic filtering
	[[nodiscard]]
	auto supportsSamplerAnisotropy() const noexcept -> bool {
		return m_sampler_anisotropy_supported;
	}

	/// @brief Device limit to clamp requested sampler anisotropy against
	[[nodiscard]]
	auto maxSamplerAnisotropy() const noexcept -> float {
		return m_max_sampler_anisotropy;
	}

	// TODO: UI system should submit on the render thread
	/// @brief Guards graphics queue submission; the render thread and the UI system submit on the same queue
	[[nodiscard]]
	auto graphicsSubmitMutex() const noexcept -> std::mutex& {
		return m_graphics_submit_mutex;
	}

private:
	/// @brief Picks the highest-scoring physical device supporting @p required_device_extensions, and stores
	/// it with its queue family indices
	///
	/// Logs the scoring breakdown and rejection reason for every candidate
	void pickPhysicalDevice(std::span<const char* const> required_device_extensions);
	/**
	 * Creates a Vulkan logical device and initializes the associated memory allocator
	 */
	void createLogicalDeviceAndAllocator(std::span<const char* const> required_device_extensions);

	/// @brief Weighted suitability score for @p device: type, memory, limits, features, extensions, API
	/// version and queue families
	///
	/// Penalties rather than gates, so the total ranks candidates instead of rejecting them
	[[nodiscard]]
	auto calculateDeviceScore(const vk::PhysicalDevice& device, std::span<const char* const> required_device_extensions)
	    -> DeviceScore;

	[[nodiscard]]
	auto checkValidationLayerSupport() -> bool;

#if defined(_WIN32)
	/// @brief Detects an Nsight Graphics installation and injects+initializes whichever activity
	/// TOAST_NSIGHT_MODE selects; sets m_nsight_mode on success. Must run before the VkInstance is created
	void initializeNsightActivity();
#endif

	bool m_validation_enabled = false;

	vk::raii::Context m_context;
	vk::raii::Instance m_instance = nullptr;
#ifndef NDEBUG
	vk::raii::DebugUtilsMessengerEXT m_debug_messenger = nullptr;
#endif

	vk::raii::PhysicalDevice m_physical_device = nullptr;
	vk::raii::Device m_device = nullptr;

	std::optional<vma::raii::Allocator> m_allocator;

	uint32_t m_graphics_queue_family_index = std::numeric_limits<uint32_t>::max();
	uint32_t m_compute_queue_family_index = std::numeric_limits<uint32_t>::max();
	uint32_t m_transfer_queue_family_index = std::numeric_limits<uint32_t>::max();
	vk::Queue m_graphics_queue = nullptr;
	vk::Queue m_compute_queue = nullptr;
	vk::Queue m_transfer_queue = nullptr;

	bool m_sampler_anisotropy_supported = false;
	float m_max_sampler_anisotropy = 1.0f;

	mutable std::mutex m_graphics_submit_mutex;

	RENDERDOC_API_1_6_0* rdoc_api = nullptr;

	// Mutated lazily from activateNsightGpuTraceIfNeeded(), which callers reach through a const VulkanCore*
	// (see VulkanRenderer::m_core) - the activation is logically a one-time cache fill, not a state change
	// callers need to observe through a non-const handle
	mutable NsightMode m_nsight_mode = NsightMode::none;

	/// Whether VK_EXT_frame_boundary was available and enabled on the device
	bool m_frame_boundary_supported = false;

	/// Whether acceleration structures + ray query were available and enabled
	bool m_ray_tracing_supported = false;

	/// minAccelerationStructureScratchOffsetAlignment, queried once when RT is available
	uint32_t m_as_scratch_alignment = 0;
	mutable bool m_nsight_gputrace_activated = false;
};
}
