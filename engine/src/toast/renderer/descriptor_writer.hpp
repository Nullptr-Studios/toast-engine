/**
 * @file descriptor_writer.hpp
 * @author dario
 * @date 14/08/2026
 */

#pragma once

#include "vulkan_common.hpp"

#include <cstdint>
#include <deque>
#include <span>
#include <vector>

namespace renderer {

/**
 * @brief Accumulates descriptor writes whose backing info structs cannot dangle
 *
 * `VkWriteDescriptorSet` holds raw pointers to info structs that must outlive it, so building them in a
 * `std::vector` means one reallocation leaves every earlier write pointing at freed memory. The infos live in
 * a `std::deque`, whose references survive growth. Array bindings need contiguous infos, so each is a
 * `std::vector` inside that deque. Not thread-safe: build it, fill it, flush it
 */
class DescriptorWriter {
public:
	/// @brief Queues a buffer binding. A null @p buffer is skipped, matching what the call sites already did
	auto buffer(
	    vk::DescriptorSet set, uint32_t binding, vk::DescriptorType type, vk::Buffer buffer, vk::DeviceSize range = VK_WHOLE_SIZE,
	    vk::DeviceSize offset = 0
	) -> DescriptorWriter&;

	/// @brief Queues a single combined image sampler (or any image-typed descriptor)
	auto image(
	    vk::DescriptorSet set, uint32_t binding, vk::DescriptorType type, vk::Sampler sampler, vk::ImageView view,
	    vk::ImageLayout layout
	) -> DescriptorWriter&;

	/// @brief Queues a descriptor *array*, whose infos must be contiguous
	auto
	    imageArray(vk::DescriptorSet set, uint32_t binding, vk::DescriptorType type, std::span<const vk::DescriptorImageInfo> infos)
	        -> DescriptorWriter&;

	/// @brief Queues an acceleration structure, which is written through a pNext rather than an info pointer
	auto accelerationStructure(vk::DescriptorSet set, uint32_t binding, vk::AccelerationStructureKHR structure)
	    -> DescriptorWriter&;

	/// @brief Applies everything queued, then clears. A no-op when nothing was queued
	void flush(const vk::raii::Device& device);

private:
	std::vector<vk::WriteDescriptorSet> m_writes;

	// Deques, not vectors: a write holds a pointer into these, and a deque never moves an element it has
	// already handed out
	std::deque<vk::DescriptorBufferInfo> m_buffer_infos;
	std::deque<std::vector<vk::DescriptorImageInfo>> m_image_infos;
	std::deque<vk::AccelerationStructureKHR> m_structures;
	std::deque<vk::WriteDescriptorSetAccelerationStructureKHR> m_structure_writes;
};

}
