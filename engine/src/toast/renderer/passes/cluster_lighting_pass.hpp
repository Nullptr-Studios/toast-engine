/// @file cluster_lighting_pass.hpp
/// @author dario
/// @date 18/07/2026

#pragma once
#include "../clustered_lighting_constants.hpp"
#include "../compute_pass_base.hpp"
#include "../shader_layout.hpp"
#include "../vulkan_common.hpp"
#include "../vulkan_pipeline.hpp"

#include <glm/glm.hpp>
#include <span>
#include <vector>

namespace renderer {
class VulkanCore;

/**
 * @brief Builds a view-space cluster grid and culls PointLight/Spotlight sources into it every frame
 *
 * Two back-to-back dispatches sharing one layout: clusterBuildMain writes per-cluster AABBs, lightCullMain
 * sphere-tests each against every light. One buffer set per frame in flight, because frame N+1's dispatch
 * can start before frame N's fragment shader has finished reading, thats BAD
 */
class ClusterLightingPass : public IComputePass {
public:
	explicit ClusterLightingPass(const renderer::VulkanCore& core);

	void update(uint32_t frame_index, float dt) override;

	void dispatch(vk::CommandBuffer cmd, uint32_t frame_index) override;

	[[nodiscard]]
	auto getClusterParamsBuffer(uint32_t frame_index) const -> vk::Buffer;

	[[nodiscard]]
	auto getLightsBuffer(uint32_t frame_index) const -> vk::Buffer;

	[[nodiscard]]
	auto getClusterLightGridBuffer(uint32_t frame_index) const -> vk::Buffer;

	[[nodiscard]]
	auto getLightIndexListBuffer(uint32_t frame_index) const -> vk::Buffer;

	/// @brief CPU-readable copy of ClusterLightGrid (one uint32_t light count per cluster, k_cluster_count
	/// entries), for debug visualization only. A few frames stale (copied on the render thread after each
	/// dispatch(), read back here on the same thread the next time this frame_index cycles around) - fine
	/// for a debug overlay, not something the real shading path uses
	[[nodiscard]]
	auto getClusterLightGridCounts(uint32_t frame_index) const -> std::span<const uint32_t>;

private:
	/// @brief Mirrors cluster_lighting.slang's ClusterParams UBO layout exactly
	struct ClusterParamsGpu {
		glm::mat4 inverse_projection;
		glm::uvec4 cluster_dims;           // x,y,z dims, w = max_lights_per_cluster
		glm::vec4 screen_size_near_far;    // x,y screen dims, z near, w far
		uint32_t light_count = 0;
		glm::vec3 _pad0 {0.0f};
	};

	/// @brief Compute-internal only, never exposed to MeshPass - mirrors cluster_lighting.slang's ClusterAABB
	struct ClusterAabbGpu {
		glm::vec4 min_point;
		glm::vec4 max_point;
	};

	/// @brief One frame-in-flight's worth of every buffer this pass owns
	struct FrameBuffers {
		FrameResources cluster_params;
		FrameResources lights;
		FrameResources cluster_aabb;
		FrameResources cluster_light_grid;
		FrameResources light_index_list;

		/// @brief Host-visible copy of cluster_light_grid, filled by a device->host vkCmdCopyBuffer at the
		/// end of dispatch() - debug-visualization-only, see getClusterLightGridCounts()
		FrameResources cluster_light_grid_readback;
	};

	void createResources(const renderer::VulkanCore& core);

	[[nodiscard]]
	static auto createBuffer(const renderer::VulkanCore& core, vk::DeviceSize size, vk::BufferUsageFlags usage, bool host_visible)
	    -> vma::raii::Buffer;

	VulkanPipeline m_build_clusters_pipeline;
	VulkanPipeline m_cull_lights_pipeline;

	ShaderLayout m_shader_layout;
	std::vector<vk::raii::DescriptorSet> m_descriptor_sets;

	std::vector<FrameBuffers> m_frame_buffers;
};

}
