/**
 * @file skinning_pass.hpp
 * @author dario
 * @date 13/08/2026
 */

#pragma once

#include "../shader_layout.hpp"
#include "../vulkan_common.hpp"
#include "../vulkan_pipeline.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace renderer {
class VulkanCore;
class VulkanMesh;

/**
 * @brief Poses every skinned instance into a shared vertex buffer, once per frame
 *
 * Replaces the per-pass `vertexMainSkinned` copies: a character used to be skinned once per pass, and any
 * divergence showed up as the prepass rejecting the pose the main pass drew. A BLAS reads a vertex buffer
 * too, so skinning in the vertex shader also left it stuck in bind pose
 *
 * Recorded on the **graphics** command buffer, not through IComputePass - the next passes consume its output
 * from the same buffer, so one barrier orders it. The compute queue would need a semaphore and a buffer
 * ownership transfer instead
 *
 * Output is world space (glTF ignores a skinned node's own transform), so posed proxies get an identity model
 * matrix and everything downstream treats them as static geometry
 */
class SkinningPass {
public:
	explicit SkinningPass(const VulkanCore& core);

	/// @brief Records one dispatch per skinned instance in the frame, plus the barrier its readers need
	void record(vk::CommandBuffer cmd, uint32_t frame_index);

	/// @returns the buffer holding this frame's posed vertices, bound as vertex stream 0 for skinned draws
	[[nodiscard]]
	auto getPosedVertexBuffer(uint32_t frame_index) const -> vk::Buffer;

	/// @returns instances posed last frame, for the debug panel
	[[nodiscard]]
	auto getPosedInstanceCount() const noexcept -> uint32_t {
		return m_posed_instances;
	}

	[[nodiscard]]
	auto isReady() const noexcept -> bool {
		return m_pipeline.isReady();
	}

	/// @brief Capacity of the posed buffer, in vertices
	///
	/// Every *instance* of a skinned mesh owns a slice, not every mesh - two characters sharing a rig hold
	/// different poses. Sized for the handful of skinned meshes a scene is expected to carry,
	/// not for the whole scene
	static constexpr uint32_t k_max_posed_vertices = 1u << 19;

private:
	/// @brief Mirrors skinning.slang's PushConstants
	struct PushConstants {
		uint32_t vertex_count = 0;
		uint32_t output_base = 0;
		uint32_t joint_offset = 0;
		uint32_t _pad0 = 0;
	};

	/// @brief Descriptor sets for one source mesh, one per frame in flight
	///
	/// Keyed by mesh rather than by instance: the source vertex and skin streams are the mesh's, and only the
	/// output slice and joint slice differ between two instances of it - both of which are push constants
	struct MeshResources {
		std::vector<vk::raii::DescriptorSet> sets;
	};

	auto ensureMeshResources(const VulkanMesh* mesh) -> MeshResources*;

	const VulkanCore* m_core = nullptr;

	ShaderLayout m_shader_layout;
	VulkanPipeline m_pipeline;

	std::vector<std::optional<vma::raii::Buffer>> m_posed_buffers;
	std::unordered_map<const VulkanMesh*, MeshResources> m_mesh_resources;

	uint32_t m_posed_instances = 0;
};

}
