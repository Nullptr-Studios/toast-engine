/// @file VulkanMesh.hpp
/// @author dario
/// @date 07/06/2026

#pragma once

#include "glm/glm.hpp"
#include "vertex.hpp"
#include "vulkan_common.hpp"
#include "vulkan_resource_base.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <toast/assets/core_types.hpp>

namespace renderer {
class VulkanCore;

auto vertexBindingDescription() -> vk::VertexInputBindingDescription;

auto vertexAttributeDescriptions() -> std::array<vk::VertexInputAttributeDescription, 5>;

/// @brief GPU mesh with vertex and index buffers stored in VRAM
class VulkanMesh : public IVulkanResource {
public:
	VulkanMesh() = default;

	struct UploadData {
		std::span<const Vertex> vertices;
		std::span<const uint32_t> indices;
		std::span<const SkinVertex> skin_vertices;    ///< empty for a static mesh
	};

	void create(
	    const renderer::VulkanCore& core, UploadData data, uint32_t graphics_queue_family_index,
	    uint32_t transfer_queue_family_index, std::string_view debug_name = {}
	);

	void destroy();

	/// @brief Binds vertex buffer 0 (and the index buffer). Use for a static-pipeline draw
	void bind(vk::CommandBuffer cmd) const;

	/// @brief Binds this mesh's index buffer, with @p posed_vertices as the vertex stream instead of its own
	///
	/// SkinningPass already posed this instance into a slice of the shared buffer, so the pipeline is the
	/// ordinary static one and only the vertex source differs from bind()
	///
	/// @param posed_vertex_offset First vertex of this instance's slice, in vertices
	void bindPosed(vk::CommandBuffer cmd, vk::Buffer posed_vertices, uint32_t posed_vertex_offset) const;

	[[nodiscard]]
	auto getVertexCount() const noexcept -> uint32_t {
		return m_vertex_count;
	}

	/// @returns the mesh's own (bind-pose) vertex stream, read by SkinningPass; null when not uploaded
	[[nodiscard]]
	auto getVertexBuffer() const -> vk::Buffer {
		return m_vertex_buffer.has_value() ? **m_vertex_buffer : vk::Buffer {};
	}

	/// @returns the per-vertex joint influences, or null for a static mesh
	[[nodiscard]]
	auto getSkinVertexBuffer() const -> vk::Buffer {
		return m_skin_vertex_buffer.has_value() ? **m_skin_vertex_buffer : vk::Buffer {};
	}

	/// @returns the index buffer, shared by every instance of this mesh; null when not uploaded
	///
	/// Shared with every posed instance's BLAS too - skinning moves vertices and never changes topology
	[[nodiscard]]
	auto getIndexBuffer() const -> vk::Buffer {
		return m_index_buffer.has_value() ? **m_index_buffer : vk::Buffer {};
	}

	[[nodiscard]]
	auto getIndexCount() const noexcept -> uint32_t {
		return m_index_count;
	}

	/// @param instance_count Proxies sharing this mesh and material, drawn as one instanced call
	void draw(vk::CommandBuffer cmd, uint32_t instance_count = 1) const;

	[[nodiscard]]
	auto isSkinned() const noexcept -> bool {
		return m_skin_vertex_buffer.has_value();
	}

	void recordUpload(
	    vk::CommandBuffer cmd, vk::Buffer staging_buffer, vk::DeviceSize vertex_offset, vk::DeviceSize index_offset,
	    vk::DeviceSize skin_vertex_offset
	) const;

	/// @brief Allocates this mesh's bottom-level acceleration structure and the scratch buffer to build it
	///
	/// Split from the record step: sizes are queried host-side first, and the scratch has to outlive the
	/// command buffer consuming it. No-op without ray query, so such a device simply has no BLAS
	///
	/// @returns scratch buffer the caller must keep alive until the build completes, or nullopt
	auto createAccelerationStructure(const VulkanCore& core) -> std::optional<vma::raii::Buffer>;

	/// @brief Records the BLAS build; the vertex and index copies must already be recorded and barriered
	void recordBuildAccelerationStructure(vk::CommandBuffer cmd) const;

	/// @returns true once this mesh has a bottom-level acceleration structure
	[[nodiscard]]
	auto hasAccelerationStructure() const noexcept -> bool {
		return m_blas_address != 0;
	}

	/// @returns device address of this mesh's BLAS, or 0 when it has none
	///
	/// By address, not handle, so the TLAS build knows nothing about what produced the bottom level
	[[nodiscard]]
	auto getAccelerationStructureAddress() const noexcept -> vk::DeviceAddress {
		return m_blas_address;
	}

private:
	std::optional<vma::raii::Buffer> m_vertex_buffer;
	std::optional<vma::raii::Buffer> m_index_buffer;
	std::optional<vma::raii::Buffer> m_skin_vertex_buffer;

	vk::DeviceSize m_vertex_size = 0;
	vk::DeviceSize m_index_size = 0;
	vk::DeviceSize m_skin_vertex_size = 0;

	uint32_t m_vertex_count = 0;
	uint32_t m_index_count = 0;

	/// Backing storage for the BLAS; the acceleration structure is a view onto this buffer
	std::optional<vma::raii::Buffer> m_blas_buffer;
	vk::raii::AccelerationStructureKHR m_blas = nullptr;
	vk::DeviceAddress m_blas_address = 0;
	vk::AccelerationStructureBuildGeometryInfoKHR m_blas_build_info {};
	vk::AccelerationStructureGeometryKHR m_blas_geometry {};
	uint32_t m_blas_primitive_count = 0;
	/// Only valid between createAccelerationStructure() and the build's fence signalling
	mutable vk::DeviceAddress m_blas_scratch_address = 0;

	/// @brief Resolved from the RAII device's dispatcher, because the record step has a plain command buffer
	///
	/// vulkan-hpp dispatches those through its *static* loader, which has no extension entry points -
	/// buildAccelerationStructuresKHR there links against a symbol that does not exist
	PFN_vkCmdBuildAccelerationStructuresKHR m_build_acceleration_structures = nullptr;

	friend class MeshUpload;
};

class MeshUpload : public PendingResourceUpload {
public:
	/// @param source Handle to the asset owning @p data
	MeshUpload(VulkanMesh& mesh, VulkanMesh::UploadData data, assets::HandleBase source, std::string_view debug_name = {});

	VulkanMesh* mesh;

	VulkanMesh::UploadData data;
	assets::HandleBase source;
	std::string debug_name;

	/// One buffer for vertices, indices and skin data - recordUpload() copies out of it at three offsets
	vma::raii::Buffer vertex_staging = nullptr;

	void build(const VulkanCore& core) override;

	void record(vk::CommandBuffer cmd) override;

	auto resource() -> IVulkanResource* override { return mesh; }
};

}
