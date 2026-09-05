#include "scene_descriptor_set.hpp"

#include "descriptor_writer.hpp"
#include "passes/cluster_lighting_pass.hpp"
#include "passes/environment_pass.hpp"
#include "passes/reflection_probe_pass.hpp"
#include "passes/shadow_pass.hpp"
#include "ray_tracing_scene.hpp"
#include "vulkan_core.hpp"
#include "vulkan_debug.hpp"
#include "vulkan_renderer.hpp"

#include <algorithm>
#include <format>
#include <toast/log.hpp>

namespace renderer {

void SceneDescriptorSets::create(
    const VulkanCore& core, const ShaderReflection& reflection, vk::DescriptorSetLayout layout, std::string_view owner
) {
	m_core = &core;
	m_owner = owner;

	const auto& device = core.getDevice();
	const vk::DescriptorPool pool = VulkanRenderer::instance->getDescriptorPoolHandle();
	const auto* cluster_lighting = VulkanRenderer::instance->getClusterLightingPass();

	m_sets.clear();
	m_sets.reserve(VulkanRenderer::k_frames_in_flight);
	m_scene_binding.reset();
	m_bound_tlas.assign(VulkanRenderer::k_frames_in_flight, vk::AccelerationStructureKHR {});

	for (uint32_t i = 0; i < VulkanRenderer::k_frames_in_flight; ++i) {
		const vk::DescriptorSetAllocateInfo alloc_info(pool, 1, &layout);
		auto allocated = device.allocateDescriptorSets(alloc_info);
		m_sets.push_back(std::move(allocated[0]));
		setDebugName(core, *m_sets[i], std::format("{} SceneSet[{}]", m_owner, i));

		const auto* frame_res = VulkanRenderer::instance->getFrameUBORes(i);
		if (!frame_res->gpu_buffer.has_value()) {
			TOAST_CRITICAL("Render", "Frame UBO buffer missing for frame {}", i);
			continue;
		}

		DescriptorWriter writer;

		for (const auto& binding : reflection.bindings) {
			if (binding.set != 0) {
				continue;
			}

			// EnvironmentPass's cubemaps. Created once at startup and never resized, so like the shadow maps
			// they are bound once here rather than per frame
			const bool is_irradiance = binding.name == "gIrradianceMap";
			if (is_irradiance || binding.name == "gPrefilteredEnv") {
				const auto* environment = VulkanRenderer::instance->getEnvironmentPass();
				vk::ImageView view;
				if (environment != nullptr) {
					view = is_irradiance ? environment->getIrradianceView() : environment->getPrefilteredView();
				}

				// Falling back here while environment_params says "ready" is the one combination that renders a
				// scene black with nothing else to show for it: the shader samples, and gets the 1x1 black cube
				if (!view && i == 0) {
					TOAST_WARN(
					    "Render", "'{}' bound the black fallback cubemap to {} - no environment lighting here", m_owner, binding.name
					);
				}

				// A cube-typed descriptor has to be fed a cube view even when there is no environment; the
				// flag in FrameUBO::environment_params is what actually keeps the shader off it
				writer.image(
				    *m_sets[i],
				    binding.binding,
				    vk::DescriptorType::eCombinedImageSampler,
				    environment != nullptr ? environment->getSampler() : VulkanRenderer::instance->getDefaultSampler(),
				    view ? view : VulkanRenderer::instance->getDefaultCubeView(),
				    vk::ImageLayout::eShaderReadOnlyOptimal
				);
				continue;
			}

			// ReflectionProbePass's per-probe captures, as a descriptor array
			const bool is_probe_irradiance = binding.name == "gProbeIrradiance";
			if (binding.name == "gProbeMaps" || is_probe_irradiance) {
				const auto* probes = VulkanRenderer::instance->getReflectionProbePass();
				const uint32_t count = std::max(binding.count, 1U);

				std::vector<vk::DescriptorImageInfo> probe_infos;
				probe_infos.reserve(count);
				for (uint32_t probe = 0; probe < count; ++probe) {
					vk::ImageView view;
					if (probes != nullptr) {
						view = is_probe_irradiance ? probes->getProbeIrradianceView(probe) : probes->getProbeView(probe);
					}

					// Every slot must be written even when nothing has been baked into it; an unbaked probe is
					// flagged in the frame UBO rather than left dangling here
					probe_infos.emplace_back(
					    probes != nullptr ? probes->getSampler() : VulkanRenderer::instance->getDefaultSampler(),
					    view ? view : VulkanRenderer::instance->getDefaultCubeView(),
					    vk::ImageLayout::eShaderReadOnlyOptimal
					);
				}

				writer.imageArray(*m_sets[i], binding.binding, vk::DescriptorType::eCombinedImageSampler, probe_infos);
				continue;
			}

			// ShadowPass's per-frame layered depth maps
			const bool is_cascade_map = binding.name == "gShadowMap";
			if (is_cascade_map || binding.name == "gPunctualShadowMaps") {
				const auto* shadow_pass = VulkanRenderer::instance->getShadowPass();
				vk::ImageView view;
				if (shadow_pass != nullptr) {
					view = is_cascade_map ? shadow_pass->getCascadeMapView(i) : shadow_pass->getPunctualMapView(i);
				}

				// No shadow pass (standalone runtime): a 1x1 depth image cleared to 1.0 reads as unshadowed.
				// View and sampler must come from the same fallback - a shadow descriptor may only pair a
				// comparison sampler with a depth image
				const bool has_map = static_cast<bool>(view);
				writer.image(
				    *m_sets[i],
				    binding.binding,
				    vk::DescriptorType::eCombinedImageSampler,
				    has_map && shadow_pass != nullptr ? shadow_pass->getShadowSampler()
				                                      : VulkanRenderer::instance->getDefaultShadowSampler(),
				    has_map ? view : VulkanRenderer::instance->getDefaultShadowMapView(),
				    vk::ImageLayout::eShaderReadOnlyOptimal
				);
				continue;
			}

			// Written per frame by updateTlas() rather than here, since the handle changes. On a device without
			// ray query it is never written at all, which is legal only because the layout marks it partially
			// bound and FrameUBO::traced_shadow_params keeps the shader off it
			if (binding.name == "gScene") {
				m_scene_binding = binding.binding;
				continue;
			}

			vk::Buffer buffer;
			vk::DeviceSize size = VK_WHOLE_SIZE;
			vk::DescriptorType descriptor_type = vk::DescriptorType::eUniformBuffer;

			if (binding.name == "gCamera") {
				buffer = **frame_res->gpu_buffer;
				size = sizeof(VulkanRenderer::FrameUBO);
			} else if (cluster_lighting != nullptr && binding.name == "gClusterParams") {
				buffer = cluster_lighting->getClusterParamsBuffer(i);
			} else if (cluster_lighting != nullptr && binding.name == "gLights") {
				buffer = cluster_lighting->getLightsBuffer(i);
				descriptor_type = vk::DescriptorType::eStorageBuffer;
			} else if (cluster_lighting != nullptr && binding.name == "gClusterLightGrid") {
				buffer = cluster_lighting->getClusterLightGridBuffer(i);
				descriptor_type = vk::DescriptorType::eStorageBuffer;
			} else if (cluster_lighting != nullptr && binding.name == "gLightIndexList") {
				buffer = cluster_lighting->getLightIndexListBuffer(i);
				descriptor_type = vk::DescriptorType::eStorageBuffer;
			} else if (binding.name == "gJointMatrices") {
				buffer = VulkanRenderer::instance->getJointMatrixBuffer(i);
				size = sizeof(glm::mat4) * VulkanRenderer::k_max_joint_matrices;
				descriptor_type = vk::DescriptorType::eStorageBuffer;
			} else if (binding.name == "gInstances") {
				buffer = VulkanRenderer::instance->getInstanceBuffer(i);
				size = sizeof(VulkanRenderer::InstanceData) * VulkanRenderer::k_max_instances;
				descriptor_type = vk::DescriptorType::eStorageBuffer;
			} else if (binding.name == "gIrradianceSH") {
				// One buffer for every frame in flight, unlike the cluster resources above: it is written only
				// by a bake, never per frame, so there is nothing for a second copy to protect against
				if (auto* probes = VulkanRenderer::instance->getReflectionProbePassMutable(); probes != nullptr) {
					buffer = probes->getShBuffer();
				}
				descriptor_type = vk::DescriptorType::eStorageBuffer;
			} else {
				TOAST_WARN("Render", "'{}': unrecognized scene-set binding '{}', leaving unbound", m_owner, binding.name);
				continue;
			}

			// A null buffer is skipped inside the writer, which is what an unavailable pass resource means here
			writer.buffer(*m_sets[i], binding.binding, descriptor_type, buffer, size);
		}

		writer.flush(device);
	}
}

void SceneDescriptorSets::updateTlas(uint32_t frame_index) {
	if (m_core == nullptr || !m_scene_binding.has_value() || frame_index >= m_bound_tlas.size()) {
		return;
	}

	const auto* scene = VulkanRenderer::instance->getRayTracingScene();
	const vk::AccelerationStructureKHR tlas = scene != nullptr ? scene->getAccelerationStructure(frame_index) : nullptr;

	// A null handle means this frame has nothing traceable. Leave the last write in place rather than clearing
	// the descriptor - a descriptor cannot be written with a null structure, and the shader is already off it
	// via traced_shadow_params
	if (!tlas || tlas == m_bound_tlas[frame_index]) {
		return;
	}

	DescriptorWriter writer;
	writer.accelerationStructure(*m_sets[frame_index], *m_scene_binding, tlas);
	writer.flush(m_core->getDevice());

	m_bound_tlas[frame_index] = tlas;
}

}
