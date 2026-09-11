/**
 * @file skinned_blas_pool.hpp
 * @author dario
 * @date 13/08/2026
 */

#pragma once

#include "vulkan_common.hpp"

#include <cstdint>
#include <optional>
#include <unordered_map>

namespace renderer {
class VulkanCore;
class VulkanMesh;

/**
 * @brief One bottom-level acceleration structure per skinned *instance*, refit each frame
 *
 * A static mesh's BLAS lives on the VulkanMesh - every instance shares its vertex buffer and differs only by
 * the TLAS transform. A posed instance has no such transform, since its vertices are already world-space in
 * its own slice of the posed buffer, so two characters sharing a rig are two pieces of geometry
 *
 * Refit rather than rebuilt, because skinning moves vertices without changing topology; a full rebuild only
 * happens when the mesh or its counts change. Keyed by node UID, since the proxy list is a per-frame snapshot
 * and a structure that persists needs an identity that persists too
 */
class SkinnedBlasPool {
public:
	explicit SkinnedBlasPool(const VulkanCore& core);

	/// @brief Marks every entry unused; call once per frame before any recordFor()
	void beginFrame();

	/// @brief Ensures this instance has a structure and records its build or refit
	///
	/// @param posed_vertices SkinningPass's output buffer for this frame
	/// @param posed_vertex_offset This instance's slice of it, in vertices
	/// @returns the device address to reference from the TLAS, or 0 when nothing could be built
	auto recordFor(
	    vk::CommandBuffer cmd, uint64_t node_uid, const VulkanMesh& mesh, vk::Buffer posed_vertices, uint32_t posed_vertex_offset
	) -> vk::DeviceAddress;

	/// @brief Releases entries no instance asked for this frame; call after the last recordFor()
	///
	/// A character that despawns would otherwise keep its structure and scratch buffer alive forever. Deferred
	/// by k_retire_frames so a proxy that vanishes for one frame - culled from the snapshot, briefly unloaded -
	/// does not pay a full rebuild to come back
	void endFrame();

	/// @returns the device address recorded for @p node_uid this frame, or 0 if it has none
	///
	/// Separate from recordFor()'s return so the TLAS loop does not have to hold onto every address the build
	/// loop produced - the two walk the same list for different reasons
	[[nodiscard]]
	auto addressFor(uint64_t node_uid) const -> vk::DeviceAddress;

	/// @returns structures recorded this frame, for the debug panel
	[[nodiscard]]
	auto getRecordedCount() const noexcept -> uint32_t {
		return m_recorded;
	}

private:
	/// Frames an untouched entry survives before its memory is released
	static constexpr uint64_t k_retire_frames = 8;

	struct Entry {
		/// What the structure was built from, so a mesh swapped under one node forces a rebuild
		const VulkanMesh* mesh = nullptr;
		uint32_t vertex_count = 0;
		uint32_t primitive_count = 0;

		std::optional<vma::raii::Buffer> blas_buffer;
		std::optional<vma::raii::Buffer> scratch;
		vk::raii::AccelerationStructureKHR blas = nullptr;
		vk::DeviceAddress address = 0;
		vk::DeviceAddress scratch_address = 0;

		/// False until a full build has been recorded; a refit against a structure that never existed is
		/// undefined, so the first frame of an instance is always a build
		bool built = false;
		uint64_t last_used_frame = 0;
	};

	/// @returns the entry for @p node_uid, (re)allocating it when the geometry it was built for changed
	auto ensureEntry(uint64_t node_uid, const VulkanMesh& mesh) -> Entry*;

	const VulkanCore* m_core = nullptr;
	std::unordered_map<uint64_t, Entry> m_entries;

	uint64_t m_frame = 0;
	uint32_t m_recorded = 0;

	/// Resolved from the RAII device's dispatcher - a plain vk::CommandBuffer dispatches through vulkan-hpp's
	/// static loader, which carries no extension entry points
	PFN_vkCmdBuildAccelerationStructuresKHR m_build_acceleration_structures = nullptr;
};

}
