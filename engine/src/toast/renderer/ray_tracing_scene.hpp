/**
 * @file ray_tracing_scene.hpp
 * @author dario
 * @date 13/08/2026
 */

#pragma once

#include "vulkan_common.hpp"

#include <cstdint>
#include <glm/glm.hpp>
#include <optional>
#include <vector>

namespace renderer {
class VulkanCore;

/**
 * @brief The top-level acceleration structure everything traceable in the scene lives in
 *
 * Deliberately knows nothing about what produced a bottom-level structure: an instance references its
 * geometry by **device address**, so a triangle BLAS from a mesh and a procedural-AABB BLAS from anything
 * brick are the same thing here - two addresses and two transforms. That is why this is a separate class
 * rather than a method walking `mesh_instances`
 *
 * Rebuilt rather than refitted each frame. A refit is cheaper but only valid while topology is unchanged,
 * and destructible geometry changes topology constantly
 */
class RayTracingScene {
public:
	/// @brief One traceable instance, from any geometry source
	struct Instance {
		/// World transform; the top 3x4 is what the acceleration structure stores
		glm::mat4 transform {1.0f};
		/// Address of the bottom-level structure this instance references
		vk::DeviceAddress blas_address = 0;
		/// Readable in the shader via RayQuery's instance id - what a hit uses to find its material
		uint32_t custom_index = 0;
		/// Trace-time filter; a ray with a zero mask bit skips this instance entirely
		uint8_t mask = 0xFF;
	};

	RayTracingScene(const VulkanCore& core, uint32_t frames_in_flight);

	/// @brief Drops last frame's instances; call once per frame before adding any
	void beginFrame();

	/// @brief Adds one instance from any source. Ignored once the per-frame capacity is reached
	void addInstance(const Instance& instance);

	/// @brief Uploads the instance buffer and records the TLAS build for @p frame_index
	///
	/// Records into a compute-capable queue's command buffer - an acceleration-structure build is not
	/// permitted on a transfer-only family
	void build(vk::CommandBuffer cmd, uint32_t frame_index);

	/// @returns the structure to bind for tracing, or null when this frame has no instances
	[[nodiscard]]
	auto getAccelerationStructure(uint32_t frame_index) const -> vk::AccelerationStructureKHR;

	[[nodiscard]]
	auto getInstanceCount() const noexcept -> uint32_t {
		return static_cast<uint32_t>(m_instances.size());
	}

	/// @brief Capacity per frame
	static constexpr uint32_t k_max_instances = 8192;

private:
	/// Per frame in flight, because the build writes while an earlier frame may still be tracing
	struct FrameResources {
		std::optional<vma::raii::Buffer> instance_buffer;
		std::optional<vma::raii::Buffer> tlas_buffer;
		std::optional<vma::raii::Buffer> scratch;
		vk::raii::AccelerationStructureKHR tlas = nullptr;
		uint32_t built_instances = 0;
	};

	const VulkanCore* m_core = nullptr;
	std::vector<FrameResources> m_frames;
	std::vector<Instance> m_instances;

	/// Resolved from the RAII device's dispatcher - a plain vk::CommandBuffer dispatches through vulkan-hpp's
	/// static loader, which carries no extension entry points
	PFN_vkCmdBuildAccelerationStructuresKHR m_build_acceleration_structures = nullptr;
};

}
