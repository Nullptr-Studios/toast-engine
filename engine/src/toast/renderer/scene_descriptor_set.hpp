/**
 * @file scene_descriptor_set.hpp
 * @author dario
 * @date 29/08/2026
 *
 * @brief Descriptor set 0: the scene resources the engine binds, never a material
 */

#pragma once

#include "shader_reflection.hpp"
#include "vulkan_common.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace renderer {
class VulkanCore;

/**
 * @brief One descriptor set per frame in flight, holding whatever set-0 bindings a shader declares
 *
 * Written by *name* out of shader reflection, so a shader opts into as many as it reads. Shared rather than
 * per-pass because a second copy of this name table stops matching the first the next time a binding is added
 */
class SceneDescriptorSets {
public:
	/// @brief Allocates one set per frame from @p layout and writes every set-0 binding @p reflection declares
	///
	/// Safe to call again after a pipeline rebuild; the previous sets are released
	void create(const VulkanCore& core, const ShaderReflection& reflection, vk::DescriptorSetLayout layout, std::string_view owner);

	/// @brief Re-points the TLAS binding at this frame's acceleration structure
	///
	/// The one set-0 binding that cannot be written once - the handle changes as geometry does. Call before
	/// the bind, since the draws consume the set
	void updateTlas(uint32_t frame_index);

	[[nodiscard]]
	auto get(uint32_t frame_index) const -> vk::DescriptorSet {
		return frame_index < m_sets.size() ? *m_sets[frame_index] : vk::DescriptorSet {};
	}

	[[nodiscard]]
	auto empty() const noexcept -> bool {
		return m_sets.empty();
	}

private:
	const VulkanCore* m_core = nullptr;
	std::string m_owner;

	std::vector<vk::raii::DescriptorSet> m_sets;

	/// Set when the shader declares gScene; absent on a device without ray query
	std::optional<uint32_t> m_scene_binding;

	/// Last TLAS handle written per frame, so an unchanged structure costs no descriptor write
	std::vector<vk::AccelerationStructureKHR> m_bound_tlas;
};

}
