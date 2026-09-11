/**
 * @file skinning_pass.cpp
 * @author dario
 * @date 13/08/2026
 */

#include "skinning_pass.hpp"

#include "../descriptor_writer.hpp"
#include "../shader_cache.hpp"
#include "../vertex.hpp"
#include "../vulkan_core.hpp"
#include "../vulkan_debug.hpp"
#include "../vulkan_mesh.hpp"
#include "../vulkan_renderer.hpp"

#include <array>
#include <format>
#include <toast/assets/assets.hpp>
#include <toast/log.hpp>

namespace renderer {

namespace {

/// Must match skinning.slang's [numthreads(64,1,1)]
constexpr uint32_t k_group_size = 64;

}

SkinningPass::SkinningPass(const VulkanCore& core) : m_core(&core) {
	const auto uid = assets::resolveURI("core://shaders/skinning.slang");
	const auto shader = uid.has_value() ? ShaderCache::get().acquire(*uid) : nullptr;
	if (!shader) {
		TOAST_ERROR("Render", "SkinningPass shader core://shaders/skinning.slang unavailable, skinned meshes will not deform");
		return;
	}

	m_shader_layout.rebuild(core, shader->reflection, "SkinningPass");

	VulkanPipeline::Config config;
	config.pipeline_type = VulkanPipeline::PipelineType::compute;
	config.debug_name = "SkinningPass";
	config.shader_spirv = shader->spirv;
	config.pipeline_layout = *m_shader_layout.getPipelineLayout();
	config.compute_entry = "computeMain";
	m_pipeline.rebuild(core, config);

	m_posed_buffers.resize(VulkanRenderer::k_frames_in_flight);
	for (uint32_t i = 0; i < VulkanRenderer::k_frames_in_flight; ++i) {
		vk::BufferCreateInfo buffer_ci {};
		buffer_ci.size = static_cast<vk::DeviceSize>(k_max_posed_vertices) * sizeof(Vertex);
		// eVertexBuffer because every raster pass binds this as stream 0 for a skinned draw; eStorageBuffer
		// because this pass writes it. The acceleration-structure flags are here for the BLAS refit that
		// follows in R2 - adding usage later would mean reallocating a buffer descriptor sets already point at
		buffer_ci.usage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
		                  vk::BufferUsageFlagBits::eShaderDeviceAddress |
		                  vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
		buffer_ci.sharingMode = vk::SharingMode::eExclusive;

		vma::AllocationCreateInfo alloc_ci {};
		alloc_ci.usage = vma::MemoryUsage::eAutoPreferDevice;

		m_posed_buffers[i].emplace(core.getAllocator().createBuffer(buffer_ci, alloc_ci));
		setDebugName(core, **m_posed_buffers[i], std::format("SkinningPass PosedVertices[{}]", i));
	}

	TOAST_INFO(
	    "Render",
	    "SkinningPass ready: {} posed vertices max ({:.1f} MiB per frame in flight)",
	    k_max_posed_vertices,
	    (static_cast<double>(k_max_posed_vertices) * sizeof(Vertex)) / (1024.0 * 1024.0)
	);
}

auto SkinningPass::getPosedVertexBuffer(uint32_t frame_index) const -> vk::Buffer {
	if (frame_index >= m_posed_buffers.size() || !m_posed_buffers[frame_index].has_value()) {
		return {};
	}
	return **m_posed_buffers[frame_index];
}

auto SkinningPass::ensureMeshResources(const VulkanMesh* mesh) -> MeshResources* {
	auto [it, inserted] = m_mesh_resources.try_emplace(mesh);
	if (!inserted) {
		return &it->second;
	}

	MeshResources& res = it->second;

	const auto& layouts = m_shader_layout.getDescriptorSetLayouts();
	if (layouts.empty()) {
		return &res;
	}

	const auto& device = m_core->getDevice();
	const vk::DescriptorSetLayout set_layout = *layouts[0];
	const vk::DescriptorPool pool = VulkanRenderer::instance->getDescriptorPoolHandle();

	const vk::Buffer vertices = mesh->getVertexBuffer();
	const vk::Buffer skin = mesh->getSkinVertexBuffer();
	if (!vertices || !skin) {
		return &res;
	}

	res.sets.reserve(VulkanRenderer::k_frames_in_flight);
	for (uint32_t i = 0; i < VulkanRenderer::k_frames_in_flight; ++i) {
		const vk::DescriptorSetAllocateInfo alloc_info(pool, 1, &set_layout);
		auto allocated = device.allocateDescriptorSets(alloc_info);
		res.sets.push_back(std::move(allocated[0]));

		const vk::Buffer joints = VulkanRenderer::instance->getJointMatrixBuffer(i);
		const vk::Buffer posed = getPosedVertexBuffer(i);
		if (!joints || !posed) {
			continue;
		}

		// Whole-buffer ranges: the per-instance slices are addressed by push constant, not by descriptor
		// offset, so one set serves every instance of this mesh. Bindings 0..3, matching skinning.slang
		constexpr auto k_storage = vk::DescriptorType::eStorageBuffer;
		DescriptorWriter writer;
		writer.buffer(*res.sets[i], 0, k_storage, vertices)
		    .buffer(*res.sets[i], 1, k_storage, skin)
		    .buffer(*res.sets[i], 2, k_storage, posed)
		    .buffer(*res.sets[i], 3, k_storage, joints, sizeof(glm::mat4) * VulkanRenderer::k_max_joint_matrices);
		writer.flush(device);
	}

	return &res;
}

void SkinningPass::record(vk::CommandBuffer cmd, uint32_t frame_index) {
	m_posed_instances = 0;

	if (!m_pipeline.isReady() || frame_index >= m_posed_buffers.size()) {
		return;
	}

	const auto* frame = VulkanRenderer::instance->renderingFrame();
	if (frame == nullptr) {
		return;
	}

	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline.getPipeline());

	for (const auto& proxy : frame->mesh_instances) {
		if (proxy.mesh == nullptr || !proxy.mesh->isReady() || !proxy.mesh->isSkinned() || proxy.joint_count == 0) {
			continue;
		}
		// Assigned in tick() alongside the instance slots; k_max_posed_vertices is what caps it, and a proxy
		// past the cap is marked with this sentinel rather than posed into someone else's slice
		if (proxy.posed_vertex_offset == VulkanRenderer::MeshInstanceProxy::k_no_posed_vertices) {
			continue;
		}

		auto* res = ensureMeshResources(proxy.mesh);
		if (res == nullptr || res->sets.size() <= frame_index) {
			continue;
		}

		cmd.bindDescriptorSets(
		    vk::PipelineBindPoint::eCompute,
		    *m_shader_layout.getPipelineLayout(),
		    0,
		    std::array<vk::DescriptorSet, 1> {*res->sets[frame_index]},
		    {}
		);

		const PushConstants push {
		  .vertex_count = proxy.mesh->getVertexCount(),
		  .output_base = proxy.posed_vertex_offset,
		  .joint_offset = proxy.joint_offset,
		};
		cmd.pushConstants(*m_shader_layout.getPipelineLayout(), vk::ShaderStageFlagBits::eAll, 0, sizeof(PushConstants), &push);

		cmd.dispatch((proxy.mesh->getVertexCount() + k_group_size - 1) / k_group_size, 1, 1);
		++m_posed_instances;
	}

	if (m_posed_instances == 0) {
		return;
	}

	// The posed vertices are read as a vertex stream by every raster pass in this command buffer, and by the
	// acceleration-structure build recorded alongside them. eVertexInput is the one that is easy to miss: the
	// fetch happens before the vertex shader runs, so waiting at eVertexShader would be too late
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
	barrier.srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite;
	barrier.dstStageMask = vk::PipelineStageFlagBits2::eVertexInput | vk::PipelineStageFlagBits2::eVertexShader |
	                       vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR;
	barrier.dstAccessMask = vk::AccessFlagBits2::eVertexAttributeRead | vk::AccessFlagBits2::eShaderStorageRead |
	                        vk::AccessFlagBits2::eAccelerationStructureReadKHR;

	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers = &barrier;
	cmd.pipelineBarrier2(dependency);
}

}
