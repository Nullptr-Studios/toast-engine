/**
 * @file skinned_blas_pool.cpp
 * @author dario
 * @date 13/08/2026
 */

#include "skinned_blas_pool.hpp"

#include "vertex.hpp"
#include "vulkan_core.hpp"
#include "vulkan_debug.hpp"
#include "vulkan_mesh.hpp"

#include <algorithm>
#include <format>
#include <toast/log.hpp>
#include <tracy/Tracy.hpp>

namespace renderer {

SkinnedBlasPool::SkinnedBlasPool(const VulkanCore& core) : m_core(&core) {
	m_build_acceleration_structures = core.getDevice().getDispatcher()->vkCmdBuildAccelerationStructuresKHR;
}

void SkinnedBlasPool::beginFrame() {
	++m_frame;
	m_recorded = 0;
}

auto SkinnedBlasPool::ensureEntry(uint64_t node_uid, const VulkanMesh& mesh) -> Entry* {
	ZoneScoped;
	auto& entry = m_entries[node_uid];
	entry.last_used_frame = m_frame;

	const uint32_t vertex_count = mesh.getVertexCount();
	const uint32_t primitive_count = mesh.getIndexCount() / 3;
	if (vertex_count == 0 || primitive_count == 0) {
		return nullptr;
	}

	// Already sized for this geometry - keep the structure so the caller can refit it
	if (entry.mesh == &mesh && entry.vertex_count == vertex_count && entry.primitive_count == primitive_count &&
	    *entry.blas != VK_NULL_HANDLE) {
		return &entry;
	}

	const auto& device = m_core->getDevice();

	// Sized against the *posed* buffer's layout, not the mesh's own. The geometry addresses are patched in
	// per frame by recordFor(); only the sizes matter here, and they depend on counts and strides alone
	vk::AccelerationStructureGeometryTrianglesDataKHR triangles {};
	triangles.vertexFormat = vk::Format::eR32G32B32Sfloat;
	triangles.vertexStride = sizeof(Vertex);
	triangles.maxVertex = vertex_count - 1;
	triangles.indexType = vk::IndexType::eUint32;
	triangles.indexData.deviceAddress = m_core->getBufferAddress(mesh.getIndexBuffer());

	vk::AccelerationStructureGeometryKHR geometry {};
	geometry.geometryType = vk::GeometryTypeKHR::eTriangles;
	geometry.geometry.triangles = triangles;
	geometry.flags = vk::GeometryFlagBitsKHR::eOpaque;

	vk::AccelerationStructureBuildGeometryInfoKHR build {};
	build.type = vk::AccelerationStructureTypeKHR::eBottomLevel;
	// eAllowUpdate is what makes the per-frame refit legal at all - without it a build in eUpdate mode is
	// invalid usage rather than a slower path
	build.flags =
	    vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace | vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate;
	build.mode = vk::BuildAccelerationStructureModeKHR::eBuild;
	build.geometryCount = 1;
	build.pGeometries = &geometry;

	const auto sizes =
	    device.getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice, build, primitive_count);

	vk::BufferCreateInfo blas_ci {};
	blas_ci.size = sizes.accelerationStructureSize;
	blas_ci.usage = vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress;
	blas_ci.sharingMode = vk::SharingMode::eExclusive;

	vma::AllocationCreateInfo blas_alloc {};
	blas_alloc.usage = vma::MemoryUsage::eAutoPreferDevice;

	// Released before the buffer it is a view onto, same ordering VulkanMesh::destroy() keeps
	entry.blas = nullptr;
	entry.blas_buffer.emplace(m_core->getAllocator().createBuffer(blas_ci, blas_alloc));
	setDebugName(*m_core, **entry.blas_buffer, std::format("SkinnedBlas[{:016x}]", node_uid));

	vk::AccelerationStructureCreateInfoKHR as_ci {};
	as_ci.buffer = **entry.blas_buffer;
	as_ci.size = sizes.accelerationStructureSize;
	as_ci.type = vk::AccelerationStructureTypeKHR::eBottomLevel;
	entry.blas = vk::raii::AccelerationStructureKHR(device, as_ci);

	vk::AccelerationStructureDeviceAddressInfoKHR address_info {};
	address_info.accelerationStructure = *entry.blas;
	entry.address = device.getAccelerationStructureAddressKHR(address_info);

	// One scratch buffer kept for the entry's lifetime rather than handed back per build, because unlike a
	// static mesh this is rebuilt or refit every single frame. Sized for the larger of the two: a refit needs
	// updateScratchSize, and the first frame - plus any frame the geometry changed - needs buildScratchSize
	const vk::DeviceSize needed_scratch =
	    m_core->getScratchAllocationSize(std::max(sizes.buildScratchSize, sizes.updateScratchSize));

	vk::BufferCreateInfo scratch_ci {};
	scratch_ci.size = needed_scratch;
	scratch_ci.usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;
	scratch_ci.sharingMode = vk::SharingMode::eExclusive;

	vma::AllocationCreateInfo scratch_alloc {};
	scratch_alloc.usage = vma::MemoryUsage::eAutoPreferDevice;
	entry.scratch.emplace(m_core->getAllocator().createBuffer(scratch_ci, scratch_alloc));

	// Aligned up from an over-allocated buffer. minAccelerationStructureScratchOffsetAlignment is stricter
	// than a storage buffer's own alignment, and a misaligned scratch address does not fail at the call - it
	// corrupts the build and takes the device out later
	entry.scratch_address = m_core->getAlignedScratchAddress(**entry.scratch);

	entry.mesh = &mesh;
	entry.vertex_count = vertex_count;
	entry.primitive_count = primitive_count;
	entry.built = false;

	return &entry;
}

auto SkinnedBlasPool::recordFor(
    vk::CommandBuffer cmd, uint64_t node_uid, const VulkanMesh& mesh, vk::Buffer posed_vertices, uint32_t posed_vertex_offset
) -> vk::DeviceAddress {
	ZoneScoped;
	if (m_core == nullptr || m_build_acceleration_structures == nullptr || !posed_vertices) {
		return 0;
	}

	Entry* entry = ensureEntry(node_uid, mesh);
	if (entry == nullptr || *entry->blas == VK_NULL_HANDLE) {
		return 0;
	}

	const auto& device = m_core->getDevice();

	const vk::DeviceAddress posed_address = m_core->getBufferAddress(posed_vertices);

	vk::AccelerationStructureGeometryTrianglesDataKHR triangles {};
	triangles.vertexFormat = vk::Format::eR32G32B32Sfloat;
	// The slice this instance owns. Offsetting the address rather than using firstVertex keeps the mesh's
	// shared index buffer valid unchanged - its indices are relative to vertex 0 of whatever stream is bound,
	// which is exactly what bindPosed() does for the raster path
	triangles.vertexData.deviceAddress = posed_address + (static_cast<vk::DeviceSize>(posed_vertex_offset) * sizeof(Vertex));
	triangles.vertexStride = sizeof(Vertex);
	triangles.maxVertex = entry->vertex_count - 1;
	triangles.indexType = vk::IndexType::eUint32;

	triangles.indexData.deviceAddress = m_core->getBufferAddress(mesh.getIndexBuffer());

	vk::AccelerationStructureGeometryKHR geometry {};
	geometry.geometryType = vk::GeometryTypeKHR::eTriangles;
	geometry.geometry.triangles = triangles;
	geometry.flags = vk::GeometryFlagBitsKHR::eOpaque;

	vk::AccelerationStructureBuildGeometryInfoKHR build {};
	build.type = vk::AccelerationStructureTypeKHR::eBottomLevel;
	build.flags =
	    vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace | vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate;
	// In-place: source and destination are the same structure, which is what an update is permitted to do and
	// what avoids carrying a second copy of every character's BVH
	build.mode = entry->built ? vk::BuildAccelerationStructureModeKHR::eUpdate : vk::BuildAccelerationStructureModeKHR::eBuild;
	build.srcAccelerationStructure = entry->built ? *entry->blas : vk::AccelerationStructureKHR {};
	build.dstAccelerationStructure = *entry->blas;
	build.geometryCount = 1;
	build.pGeometries = &geometry;
	build.scratchData.deviceAddress = entry->scratch_address;

	vk::AccelerationStructureBuildRangeInfoKHR range {};
	range.primitiveCount = entry->primitive_count;
	const auto* range_ptr = reinterpret_cast<const VkAccelerationStructureBuildRangeInfoKHR*>(&range);

	m_build_acceleration_structures(
	    static_cast<VkCommandBuffer>(cmd),
	    1,
	    reinterpret_cast<const VkAccelerationStructureBuildGeometryInfoKHR*>(&build),
	    &range_ptr
	);

	entry->built = true;
	++m_recorded;
	return entry->address;
}

auto SkinnedBlasPool::addressFor(uint64_t node_uid) const -> vk::DeviceAddress {
	const auto it = m_entries.find(node_uid);
	// built matters as well as existence: an entry allocated this frame but whose build was not recorded has
	// a valid address pointing at an uninitialised structure, and tracing that is undefined
	if (it == m_entries.end() || !it->second.built || it->second.last_used_frame != m_frame) {
		return 0;
	}
	return it->second.address;
}

void SkinnedBlasPool::endFrame() {
	if (m_entries.empty()) {
		return;
	}

	std::erase_if(m_entries, [this](const auto& pair) { return m_frame > pair.second.last_used_frame + k_retire_frames; });
}

}
