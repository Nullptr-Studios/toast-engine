/**
 * @file ray_tracing_scene.cpp
 * @author dario
 * @date 13/08/2026
 */

#include "ray_tracing_scene.hpp"

#include "vulkan_core.hpp"
#include "vulkan_debug.hpp"

#include <cstring>
#include <format>
#include <toast/log.hpp>
#include <tracy/Tracy.hpp>

namespace renderer {

RayTracingScene::RayTracingScene(const VulkanCore& core, uint32_t frames_in_flight) : m_core(&core) {
	ZoneScoped;
	if (!core.isRayTracingSupported()) {
		return;
	}

	m_build_acceleration_structures = core.getDevice().getDispatcher()->vkCmdBuildAccelerationStructuresKHR;
	m_instances.reserve(k_max_instances);
	m_frames.resize(frames_in_flight);

	for (uint32_t i = 0; i < frames_in_flight; ++i) {
		auto& frame = m_frames[i];

		// Host-visible: rewritten every frame from the CPU, and a staging copy would cost more than the write
		vk::BufferCreateInfo instance_ci {};
		instance_ci.size = sizeof(vk::AccelerationStructureInstanceKHR) * k_max_instances;
		instance_ci.usage =
		    vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress;
		instance_ci.sharingMode = vk::SharingMode::eExclusive;

		vma::AllocationCreateInfo instance_alloc {};
		instance_alloc.usage = vma::MemoryUsage::eAuto;
		instance_alloc.flags = vma::AllocationCreateFlagBits::eMapped | vma::AllocationCreateFlagBits::eHostAccessSequentialWrite;

		frame.instance_buffer.emplace(core.getAllocator().createBuffer(instance_ci, instance_alloc));
		setDebugName(core, **frame.instance_buffer, std::format("RayTracingScene Instances[{}]", i));
	}

	TOAST_INFO("Render", "RayTracingScene ready ({} instances max)", k_max_instances);
}

void RayTracingScene::beginFrame() {
	m_instances.clear();
}

void RayTracingScene::addInstance(const Instance& instance) {
	// Dropped rather than grown mid-frame: the instance buffer is sized once, and reallocating it while a
	// previous frame may still be tracing against it is not something a capacity overrun should trigger
	if (instance.blas_address == 0 || m_instances.size() >= k_max_instances) {
		return;
	}
	m_instances.push_back(instance);
}

auto RayTracingScene::getAccelerationStructure(uint32_t frame_index) const -> vk::AccelerationStructureKHR {
	if (frame_index >= m_frames.size() || m_frames[frame_index].built_instances == 0) {
		return {};
	}
	return *m_frames[frame_index].tlas;
}

void RayTracingScene::build(vk::CommandBuffer cmd, uint32_t frame_index) {
	ZoneScoped;
	if (m_core == nullptr || frame_index >= m_frames.size() || m_build_acceleration_structures == nullptr) {
		return;
	}

	auto& frame = m_frames[frame_index];
	frame.built_instances = 0;

	if (m_instances.empty() || !frame.instance_buffer.has_value()) {
		return;
	}

	const auto& device = m_core->getDevice();

	// Pack into the layout the driver reads. The transform is the top 3x4 of the world matrix in *row-major*
	// order, which is the transpose of how glm stores it - getting this wrong scatters geometry rather than
	// failing, so it is written out explicitly rather than memcpy'd
	auto* mapped = static_cast<vk::AccelerationStructureInstanceKHR*>(frame.instance_buffer->getAllocation().getInfo().pMappedData);
	if (mapped == nullptr) {
		return;
	}

	for (size_t i = 0; i < m_instances.size(); ++i) {
		const auto& source = m_instances[i];
		vk::AccelerationStructureInstanceKHR out {};
		for (int row = 0; row < 3; ++row) {
			for (int column = 0; column < 4; ++column) {
				out.transform.matrix[static_cast<size_t>(row)][static_cast<size_t>(column)] = source.transform[column][row];
			}
		}
		out.instanceCustomIndex = source.custom_index & 0xFFFFFFu;
		out.mask = source.mask;
		out.instanceShaderBindingTableRecordOffset = 0;
		// Ray query has no any-hit stage to run, so every instance is opaque as far as traversal is concerned
		out.flags = static_cast<uint8_t>(vk::GeometryInstanceFlagBitsKHR::eForceOpaque);
		out.accelerationStructureReference = source.blas_address;
		mapped[i] = out;
	}
	frame.instance_buffer->getAllocation().flush(0, sizeof(vk::AccelerationStructureInstanceKHR) * m_instances.size());

	vk::AccelerationStructureGeometryInstancesDataKHR instances_data {};
	instances_data.arrayOfPointers = VK_FALSE;
	instances_data.data.deviceAddress = m_core->getBufferAddress(**frame.instance_buffer);

	vk::AccelerationStructureGeometryKHR geometry {};
	geometry.geometryType = vk::GeometryTypeKHR::eInstances;
	geometry.geometry.instances = instances_data;

	const auto instance_count = static_cast<uint32_t>(m_instances.size());

	vk::AccelerationStructureBuildGeometryInfoKHR build {};
	build.type = vk::AccelerationStructureTypeKHR::eTopLevel;
	build.flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
	build.mode = vk::BuildAccelerationStructureModeKHR::eBuild;
	build.geometryCount = 1;
	build.pGeometries = &geometry;

	const auto sizes =
	    device.getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice, build, instance_count);

	// Reallocated only when it has to grow. The instance count changes every frame in a destructible scene,
	// and rebuilding the backing buffer each time would churn allocations for no benefit
	const bool needs_storage =
	    !frame.tlas_buffer.has_value() || frame.tlas_buffer->getAllocation().getInfo().size < sizes.accelerationStructureSize;
	if (needs_storage) {
		vk::BufferCreateInfo tlas_ci {};
		tlas_ci.size = sizes.accelerationStructureSize;
		tlas_ci.usage = vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress;
		tlas_ci.sharingMode = vk::SharingMode::eExclusive;

		vma::AllocationCreateInfo tlas_alloc {};
		tlas_alloc.usage = vma::MemoryUsage::eAutoPreferDevice;

		frame.tlas = nullptr;
		frame.tlas_buffer.emplace(m_core->getAllocator().createBuffer(tlas_ci, tlas_alloc));
		setDebugName(*m_core, **frame.tlas_buffer, std::format("RayTracingScene TLAS[{}]", frame_index));
	}

	if (needs_storage || *frame.tlas == VK_NULL_HANDLE) {
		vk::AccelerationStructureCreateInfoKHR as_ci {};
		as_ci.buffer = **frame.tlas_buffer;
		as_ci.size = sizes.accelerationStructureSize;
		as_ci.type = vk::AccelerationStructureTypeKHR::eTopLevel;
		frame.tlas = vk::raii::AccelerationStructureKHR(device, as_ci);
	}

	// Same alignment requirement as a BLAS build - a misaligned scratch address is undefined behaviour that
	// surfaces as a device loss later rather than an error here
	const vk::DeviceSize needed_scratch = m_core->getScratchAllocationSize(sizes.buildScratchSize);
	if (!frame.scratch.has_value() || frame.scratch->getAllocation().getInfo().size < needed_scratch) {
		vk::BufferCreateInfo scratch_ci {};
		scratch_ci.size = needed_scratch;
		scratch_ci.usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;
		scratch_ci.sharingMode = vk::SharingMode::eExclusive;

		vma::AllocationCreateInfo scratch_alloc {};
		scratch_alloc.usage = vma::MemoryUsage::eAutoPreferDevice;
		frame.scratch.emplace(m_core->getAllocator().createBuffer(scratch_ci, scratch_alloc));
	}

	build.dstAccelerationStructure = *frame.tlas;
	build.scratchData.deviceAddress = m_core->getAlignedScratchAddress(**frame.scratch);

	// The build reads the instance buffer this frame just wrote from the host
	vk::MemoryBarrier2 host_write {};
	host_write.srcStageMask = vk::PipelineStageFlagBits2::eHost;
	host_write.srcAccessMask = vk::AccessFlagBits2::eHostWrite;
	host_write.dstStageMask = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR;
	host_write.dstAccessMask = vk::AccessFlagBits2::eAccelerationStructureReadKHR;

	vk::DependencyInfo host_dependency {};
	host_dependency.memoryBarrierCount = 1;
	host_dependency.pMemoryBarriers = &host_write;
	cmd.pipelineBarrier2(host_dependency);

	vk::AccelerationStructureBuildRangeInfoKHR range {};
	range.primitiveCount = instance_count;
	const auto* range_ptr = reinterpret_cast<const VkAccelerationStructureBuildRangeInfoKHR*>(&range);

	m_build_acceleration_structures(
	    static_cast<VkCommandBuffer>(cmd),
	    1,
	    reinterpret_cast<const VkAccelerationStructureBuildGeometryInfoKHR*>(&build),
	    &range_ptr
	);

	// Anything tracing this frame has to see the finished structure
	vk::MemoryBarrier2 build_done {};
	build_done.srcStageMask = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR;
	build_done.srcAccessMask = vk::AccessFlagBits2::eAccelerationStructureWriteKHR;
	build_done.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eComputeShader;
	build_done.dstAccessMask = vk::AccessFlagBits2::eAccelerationStructureReadKHR;

	vk::DependencyInfo build_dependency {};
	build_dependency.memoryBarrierCount = 1;
	build_dependency.pMemoryBarriers = &build_done;
	cmd.pipelineBarrier2(build_dependency);

	frame.built_instances = instance_count;
}

}
