/// @file VulkanRenderer.hpp
/// @author dario
/// @date 16/05/2026.

#pragma once

#include "compute_pass_base.hpp"
#include "output_target_base.hpp"
#include "render_pass_base.hpp"
#include "vulkan_core.hpp"
#include "vulkan_mesh.hpp"
#include "vulkan_pipeline.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <semaphore>
#include <thread>
#include <toast/events/event.inl>
#include <toast/events/listener.hpp>
#include <toast/world/gizmo_layout.hpp>
#include <type_traits>
#include <utility>
#include <vector>

namespace assets {
class Material;
}

namespace toast {
class Camera;
class MeshNode;
class Light;
}

namespace renderer {

/**
 * @brief Coordinates rendering by managing frame submissions, render passes, and GPU synchronization
 *
 * Runs on a separate render thread and handles frame timing, depth resources,
 * and mesh upload queues
 */
class VulkanRenderer {
public:
	[[nodiscard]]
	static auto selectDepthFormat(const VulkanCore& core) -> vk::Format;

	static constexpr uint32_t k_frames_in_flight = 3;

	static constexpr uint8_t k_render_frames = 3;    // Number of frames queued for rendering

	struct FrameContext {
		vk::raii::CommandBuffer command_buffer = nullptr;
		vk::raii::CommandBuffer transfer_command_buffer = nullptr;
		vk::raii::CommandBuffer compute_command_buffer = nullptr;
		vk::raii::Semaphore image_available = nullptr;
		vk::raii::Semaphore transfer_finished = nullptr;
		vk::raii::Semaphore compute_to_graphics = nullptr;    ///< signaled by the compute submit, waited on by the graphics submit
		vk::raii::Fence in_flight = nullptr;
		vk::raii::Fence compute_in_flight = nullptr;          ///< defense in depth, separate from in_flight per the review
		uint32_t last_image_index = 0;
		bool has_submitted = false;
	};

	static constexpr uint32_t k_max_directional_lights = 4;

	/// @brief One AmbientLight/DirectionalLight is unbounded/unculled, so it rides along in FrameUBO
	/// instead of going through the clustered PointLight/Spotlight system
	struct DirectionalLightData {
		glm::vec4 direction;          // world-space, w unused
		glm::vec4 color_intensity;    // rgb color, a intensity
	};

	struct FrameUBO {
		glm::mat4 view;
		glm::mat4 projection;
		glm::mat4 view_projection;

		glm::vec3 camera_position;

		float time;

		// Global (unclustered) lighting - AmbientLight pre-summed on the CPU into one value (no position/
		// direction to cull by), DirectionalLight as a small fixed array. PointLight/Spotlight instead go
		// through RenderFrame::lights + the clustered system (inert until Stage 3b)
		glm::vec4 ambient_color_intensity {0.0f};    // rgb color, a intensity; zero if there's no AmbientLight

		// x = directional_light_count, yzw unused padding. A scalar directly followed by a vec3 packs
		// differently under std140 (16-byte-aligned, so it pads 12 bytes first) vs HLSL cbuffer rules (packs
		// tight into the same 16-byte slot) - collapsing to one uvec4 sidesteps the ambiguity entirely instead
		// of relying on whichever convention Slang's Vulkan backend happens to use for ConstantBuffer<T>
		glm::uvec4 directional_light_count_pad {0};

		std::array<DirectionalLightData, k_max_directional_lights> directional_lights {};
	};

	/// @brief One PointLight/Spotlight, resolved to POD on the main thread same as MeshInstanceProxy;
	/// inert until Stage 3b's clustered light culling reads RenderFrame::lights
	struct GpuLight {
		glm::vec4 world_pos_range;    // xyz world-space position, w = attenuation range (sphere radius, falloff distance)
		glm::vec4 view_pos_type;      // xyz view-space position (culling only), w = type (0=point,1=spot)
		glm::vec4 color_intensity;    // rgb color, a intensity
		glm::vec4 direction_pad;      // xyz world-space direction (spot only)
		glm::vec4 cone_angles;        // x = cos(outer), y = cos(inner)
	};

	struct MeshInstanceProxy {
		VulkanMesh* mesh = nullptr;
		assets::Material* material = nullptr;
		glm::mat4 model = glm::mat4(1.0f);
	};

	/// @brief One vertex of an immediate-mode debug line; two consecutive vertices make one line segment
	struct DebugVertex {
		glm::vec<3, float, glm::packed_highp> position;
		glm::vec<4, float, glm::packed_highp> color;
	};

	static_assert(std::is_standard_layout_v<DebugVertex>, "DebugVertex must be standard layout");

	/// @brief Editor selection's active gizmo
	struct TransformGizmoDraw {
		bool visible = false;
		toast::GizmoTool tool = toast::GizmoTool::select;
		glm::mat4 model {1.0f};
		toast::GizmoHandle hover = toast::GizmoHandle::none;
		toast::GizmoHandle active = toast::GizmoHandle::none;
		float drag_scale_factor = 1.0f;    ///< scale tool only: live multiplicative factor for the active handle
	};

	                                     /// @brief Main-thread-only input to TransformGizmoDraw's, set via setGizmoState()
	struct GizmoState {
		bool visible = false;
		toast::GizmoTool tool = toast::GizmoTool::select;
		glm::vec3 origin {0.0f};
		glm::quat orientation {1.0f, 0.0f, 0.0f, 0.0f};
		toast::GizmoHandle hover = toast::GizmoHandle::none;
		toast::GizmoHandle active = toast::GizmoHandle::none;
		float drag_scale_factor = 1.0f;
	};

	/// @brief One key transition, in the same SDL keycode encoding as event::WindowKey::key (a printable
	/// ASCII/unicode codepoint, or (1<<30)|scancode for non-printable keys). DebugPass maps this to an
	/// ImGuiKey on the render thread - keeping ImGui types entirely out of this widely-included header
	struct ImGuiKeyEvent {
		int32_t key = 0;
		bool down = false;
	};

	/// @brief Per-frame ImGui input, resolved main-thread-side from WindowMousePosition/WindowMouseButton/
	/// WindowMouseScroll/WindowKey/WindowChar - ImGui isn't thread-safe, so the render thread (where
	/// DebugPass actually calls ImGui::NewFrame()) never touches raw input events directly
	struct ImGuiInputSnapshot {
		glm::vec2 mouse_pos {-1.0f, -1.0f};    // -1,-1 = outside window, matches ImGui's own convention
		std::array<bool, 3> mouse_down {};     // 0=left, 1=right, 2=middle - matches ImGui's own indexing
		float mouse_wheel_x = 0.0f;
		float mouse_wheel_y = 0.0f;
		std::vector<ImGuiKeyEvent> key_events;
		std::vector<uint32_t> char_events;    // typed unicode codepoints
	};

	struct RenderFrame {
		FrameUBO frame_data;

		std::vector<MeshInstanceProxy> mesh_instances;

		// PointLight/Spotlight entries; inert until Stage 3b's clustered light culling consumes it
		std::vector<GpuLight> lights;

		// Resolved main-thread-side so ClusterLightingPass::update() never touches Camera/output-target
		// state directly from the render thread - same POD-snapshot pattern as everything else here
		glm::vec2 viewport_extent {0.0f};
		float camera_near = 0.01f;
		float camera_far = 5000.0f;

		ImGuiInputSnapshot imgui_input;

		// Viewport shading mode from the toolbar's "Mode" dropdown - 0 Lit, 1 ClusterHeatmap. Threaded
		// through MeshPass's push constants rather than a UBO field, so no cross-language packing to worry about
		uint32_t render_mode = 0;

		// Immediate-mode debug draw data queued via debugDrawLine()/debugDrawBox()/debugDrawSphere()/
		// debugDrawAxes() dnd consumed by DebugPass
		std::vector<DebugVertex> debug_line_vertices;    // consecutive pairs; each pair is one line segment
		std::vector<glm::mat4> debug_gizmo_instances;    // one axis-triad gizmo draw per entry

		TransformGizmoDraw transform_gizmo;
	};

	VulkanRenderer(const VulkanCore& core, std::unique_ptr<IOutputTarget> output_target) noexcept;

	~VulkanRenderer();

	VulkanRenderer(const VulkanRenderer&) = delete;
	auto operator=(const VulkanRenderer&) -> VulkanRenderer& = delete;
	VulkanRenderer(VulkanRenderer&&) = delete;
	auto operator=(VulkanRenderer&&) -> VulkanRenderer& = delete;

	void start() noexcept;

	[[nodiscard]]
	auto beginFrameBuild() noexcept -> RenderFrame& {
		return m_render_frames[m_write_index];
	}

	[[nodiscard]]
	auto getFreeFramesSemaphore() noexcept -> std::counting_semaphore<k_render_frames>& {
		return m_free_frames;
	}

	void submitFrame() noexcept;

	/**
	 * @brief Builds the next RenderFrame from the active camera and registered mesh proxies, then submits it
	 *
	 * Called once per simulation tick; does nothing if there's no free render slot available
	 *
	 * @param time Elapsed time in seconds, forwarded into the frame's FrameUBO
	 */
	void tick(float time) noexcept;

	/// @brief Registers @p node so its mesh is drawn each frame; no-op if already registered
	void registerMeshNodeProxy(toast::MeshNode* node);

	/// @brief Unregisters @p node so it stops being drawn
	void unregisterMeshNodeProxy(toast::MeshNode* node);

	/// @brief Registers @p node so it's collected into the per-frame light list each frame; no-op if already registered
	void registerLightNodeProxy(toast::Light* node);

	/// @brief Unregisters @p node so it stops contributing to lighting
	void unregisterLightNodeProxy(toast::Light* node);

	/**
	 * @brief Caps how often the render thread draws & presents a frame
	 *
	 * Applies uniformly: a brand new frame submitted faster than the cap is still paced to it, and an
	 * unchanged cached frame is redrawn no faster than the cap either
	 *
	 * @param max_fps Target draw rate in Hz. Pass 0 or a negative value to run fully uncapped
	 */
	void setFrameRateLimit(double max_fps) noexcept { m_frame_rate_limit_hz.store(max_fps, std::memory_order_relaxed); }

	[[nodiscard]]
	auto frameRateLimit() const noexcept -> double {
		return m_frame_rate_limit_hz.load(std::memory_order_relaxed);
	}

	void stop();

	void addRenderPass(std::unique_ptr<IRenderPass> pass);

	void addComputePass(std::unique_ptr<IComputePass> pass);

	void applyResize(vk::Extent2D extent);

	void queueResourceUpload(std::unique_ptr<PendingResourceUpload> upload);

	[[nodiscard]]
	auto getFrameUBORes(uint32_t current_frame) const -> const FrameResources* {
		return &m_frame_ubo_res[current_frame];
	}

	// Expose raw descriptor pool handle so render passes can allocate their own descriptor sets
	[[nodiscard]]
	auto getDescriptorPoolHandle() const noexcept -> vk::DescriptorPool {
		return *m_descriptor_pool;
	}

	[[nodiscard]]
	void setActiveCamera(toast::Camera* camera);

	[[nodiscard]]
	auto getCore() -> const VulkanCore& {
		return *m_core;
	}

	[[nodiscard]]
	auto getActiveCamera() -> toast::Camera* {
		return m_camera;
	}

	/// @brief Main-thread-only, resolved into RenderFrame::transform_gizmo at the top of the next tick()
	void setGizmoState(const GizmoState& state) noexcept { m_gizmo_state = state; }

	[[nodiscard]]
	auto renderingFrame() const -> const RenderFrame* {
		return m_rendering_frame;
	}

	[[nodiscard]]
	auto getRenderDocAPI() const noexcept -> const RENDERDOC_API_1_6_0* {
		return m_core->getRenderDocAPI();
	}

	[[nodiscard]]
	auto getOutputTarget() const noexcept -> const IOutputTarget& {
		return *m_output_target;
	}

	static VulkanRenderer* instance;

private:
	void drawFrame(RenderFrame& frame_data);

	void mainRenderThread();

	std::atomic_bool m_running {false};

	std::thread m_render_thread;

	std::array<RenderFrame, k_render_frames> m_render_frames;
	std::atomic<uint32_t> m_write_index = 0;
	std::atomic<uint32_t> m_read_index = 0;

	std::mutex m_queue_mutex;

	std::condition_variable m_frame_cv;

	std::queue<uint32_t> m_ready_frames;
	RenderFrame m_cached_frame;
	bool m_has_cached_frame = false;
	const RenderFrame* m_rendering_frame = nullptr;

	std::counting_semaphore<k_render_frames> m_free_frames {k_render_frames};

	struct DepthResources {
		std::optional<vma::raii::Image> image;
		std::optional<vk::raii::ImageView> view;
	};

	void createGraphicsCommandPool();
	void createTransferCommandPool();
	void createComputeCommandPool();
	void createFrameContexts();
	void createPerImageSync();
	void createDepthResources();
	void createDescriptorPool();

	void recordFrame(FrameContext& frame, uint32_t image_index) noexcept;

	// Resource uploading
	struct BatchedUploadGroup {
		std::vector<std::unique_ptr<PendingResourceUpload>> jobs;
		vk::raii::Fence completion_fence = nullptr;
	};

	std::vector<std::unique_ptr<PendingResourceUpload>> m_upload_staging;
	std::queue<BatchedUploadGroup> m_pending_uploads;
	void processPendingUploads();
	void flushResourceUploads();
	std::mutex m_upload_mutex;

	// queueResourceUpload() offloads PendingResourceUpload::build() to the thread pool, and that job
	// captures m_core by raw pointer. Tracks how many such jobs are still in flight so stop() can
	// wait for them before returning
	std::atomic<int> m_pending_upload_builds {0};

	const VulkanCore* m_core = nullptr;

	std::unique_ptr<IOutputTarget> m_output_target;
	std::vector<std::unique_ptr<IRenderPass>> m_render_passes;
	std::vector<std::unique_ptr<IComputePass>> m_compute_passes;
	vk::Format m_depth_format = vk::Format::eUndefined;
	DepthResources m_depth_resources;
	vk::ImageLayout m_depth_layout = vk::ImageLayout::eUndefined;

	vk::raii::CommandPool m_command_pool = nullptr;

	vk::raii::CommandPool m_transfer_command_pool = nullptr;

	vk::raii::CommandPool m_compute_command_pool = nullptr;

	vk::raii::DescriptorPool m_descriptor_pool = nullptr;

	std::vector<FrameContext> m_frames;
	std::vector<vk::raii::Semaphore> m_render_finished_per_image;
	std::vector<vk::Fence> m_images_in_flight;
	std::vector<vk::ImageLayout> m_output_image_layouts;
	uint32_t m_current_frame = 0;

	/// Active camera for the renderer, Can be nullptr if no camera is set
	toast::Camera* m_camera = nullptr;

	/// Main-thread-only, written by setGizmoState(), read back inside tick() on the same thread
	GizmoState m_gizmo_state;

	/// Set by event::CaptureFrame (F12 in the editor viewport), consumed once by the render thread's next
	/// iteration to wrap that one drawFrame() call in whichever capture tool is attached - RenderDoc's
	/// StartFrameCapture/EndFrameCapture, or Nsight Graphics Capture/GPU Trace's Start/Stop pair
	event::Listener m_capture_listener;
	std::atomic_bool m_capture_frame_requested {false};

	/// Main-thread-only, accumulated from WindowMousePosition/WindowMouseButton/WindowMouseScroll/WindowKey/
	/// WindowChar and resolved into RenderFrame::imgui_input inside tick() on the same thread - mirrors
	/// m_gizmo_state's pattern exactly
	event::Listener m_imgui_input_listener;
	glm::vec2 m_imgui_mouse_pos {-1.0f, -1.0f};
	std::array<bool, 3> m_imgui_mouse_down {};
	float m_imgui_wheel_x_accum = 0.0f;
	float m_imgui_wheel_y_accum = 0.0f;
	std::vector<ImGuiKeyEvent> m_imgui_key_events_accum;
	std::vector<uint32_t> m_imgui_char_events_accum;

	/// Main-thread-only, set by event::SetRenderMode (the viewport toolbar's "Mode" dropdown), read back
	/// inside tick() on the same thread
	event::Listener m_render_mode_listener;
	uint32_t m_render_mode = 0;

	std::mutex m_mesh_proxy_mutex;
	std::vector<toast::MeshNode*> m_mesh_proxy_nodes;

	std::mutex m_light_proxy_mutex;
	std::vector<toast::Light*> m_light_proxy_nodes;

	// FrameUBO and related resources
	std::vector<FrameUBO> m_frame_ubos;
	std::vector<FrameResources> m_frame_ubo_res;

	void createFrameResources();
	void updateFrameResources(uint32_t frame_index, RenderFrame& frame_data);

	void applyResizeInternal(vk::Extent2D extent);

	static constexpr uint64_t k_no_pending_resize = 0;
	std::atomic<uint64_t> m_pending_resize_packed {k_no_pending_resize};

	std::atomic<double> m_frame_rate_limit_hz {0.0};
};

inline void start() {
	VulkanRenderer::instance->start();
}

inline void stop() {
	VulkanRenderer::instance->stop();
}

inline auto beginFrameBuild() -> VulkanRenderer::RenderFrame& {
	return VulkanRenderer::instance->beginFrameBuild();
}

inline void submitFrame() {
	VulkanRenderer::instance->submitFrame();
}

inline auto getActiveCamera() -> toast::Camera* {
	return VulkanRenderer::instance->getActiveCamera();
}

inline void setActiveCamera(toast::Camera* camera) {
	VulkanRenderer::instance->setActiveCamera(camera);
}

inline void registerMeshNodeProxy(toast::MeshNode* node) {
	VulkanRenderer::instance->registerMeshNodeProxy(node);
}

inline void unregisterMeshNodeProxy(toast::MeshNode* node) {
	VulkanRenderer::instance->unregisterMeshNodeProxy(node);
}

inline void registerLightNodeProxy(toast::Light* node) {
	VulkanRenderer::instance->registerLightNodeProxy(node);
}

inline void unregisterLightNodeProxy(toast::Light* node) {
	VulkanRenderer::instance->unregisterLightNodeProxy(node);
}

inline void queueResourceUpload(std::unique_ptr<PendingResourceUpload> upload) {
	VulkanRenderer::instance->queueResourceUpload(std::move(upload));
}

inline void applyResize(vk::Extent2D extent) {
	VulkanRenderer::instance->applyResize(extent);
}

inline auto getOutputTarget() -> const IOutputTarget& {
	return VulkanRenderer::instance->getOutputTarget();
}

inline auto getCore() -> const VulkanCore& {
	return VulkanRenderer::instance->getCore();
}

inline auto getRenderDocAPI() -> const RENDERDOC_API_1_6_0* {
	return VulkanRenderer::instance->getRenderDocAPI();
}

//@WARN NOT THREAD SAFE
inline auto renderingFrame() -> const VulkanRenderer::RenderFrame* {
	return VulkanRenderer::instance->renderingFrame();
}

/// DEBUG LINES

/**
 * @brief Queues a debug line segmentfor the frame currently being built
 * @note Call between beginFrameBuild() and submitFrame()
 */
inline void debugDrawLine(glm::vec3 a, glm::vec3 b, glm::vec4 color = {1.0f, 1.0f, 1.0f, 1.0f}) {
	auto& frame = VulkanRenderer::instance->beginFrameBuild();
	frame.debug_line_vertices.push_back({a, color});
	frame.debug_line_vertices.push_back({b, color});
}

/// @brief Queues a wireframe axis-aligned box for this frame
inline void debugDrawBox(glm::vec3 min, glm::vec3 max, glm::vec4 color = {1.0f, 1.0f, 1.0f, 1.0f}) {
	const std::array<glm::vec3, 8> corners {
	  glm::vec3 {min.x, min.y, min.z},
	  glm::vec3 {max.x, min.y, min.z},
	  glm::vec3 {max.x, max.y, min.z},
	  glm::vec3 {min.x, max.y, min.z},
	  glm::vec3 {min.x, min.y, max.z},
	  glm::vec3 {max.x, min.y, max.z},
	  glm::vec3 {max.x, max.y, max.z},
	  glm::vec3 {min.x, max.y, max.z},
	};
	static constexpr std::array<std::pair<int, int>, 12> edges {
	  {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}}
	};
	for (const auto& [a, b] : edges) {
		debugDrawLine(corners[a], corners[b], color);
	}
}

/// @brief Queues a wireframe sphere for this frame
inline void debugDrawSphere(glm::vec3 center, float radius, glm::vec4 color = {1.0f, 1.0f, 1.0f, 1.0f}, int segments = 24) {
	for (int axis = 0; axis < 3; ++axis) {
		glm::vec3 prev {};
		for (int i = 0; i <= segments; ++i) {
			const float t = (static_cast<float>(i) / static_cast<float>(segments)) * glm::two_pi<float>();
			glm::vec3 p {};
			switch (axis) {
				case 0: p = center + glm::vec3(0.0f, std::cos(t), std::sin(t)) * radius; break;
				case 1: p = center + glm::vec3(std::cos(t), 0.0f, std::sin(t)) * radius; break;
				default: p = center + glm::vec3(std::cos(t), std::sin(t), 0.0f) * radius; break;
			}
			if (i > 0) {
				debugDrawLine(prev, p, color);
			}
			prev = p;
		}
	}
}

/// @brief Queues an axis-triad gizmo at @p transform
inline void debugDrawAxes(const glm::mat4& transform) {
	VulkanRenderer::instance->beginFrameBuild().debug_gizmo_instances.push_back(transform);
}

/// @brief Queues a simple arrow (shaft + a small V-shaped head) pointing from @p from to @p to - used for
/// DirectionalLight, which has no meaningful radius/range to draw a bounded shape for
inline void debugDrawArrow(glm::vec3 from, glm::vec3 to, glm::vec4 color = {1.0f, 1.0f, 1.0f, 1.0f}, float head_size = 0.2f) {
	debugDrawLine(from, to, color);

	const glm::vec3 dir = to - from;
	const float len = glm::length(dir);
	if (len < 0.0001f) {
		return;
	}
	const glm::vec3 axis = dir / len;
	const glm::vec3 up = std::abs(axis.y) < 0.99f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
	const glm::vec3 side = glm::normalize(glm::cross(up, axis));

	const glm::vec3 back = to - axis * head_size;
	debugDrawLine(to, back + side * head_size * 0.5f, color);
	debugDrawLine(to, back - side * head_size * 0.5f, color);
}

/// @brief Queues a wireframe cone (base ring + a few spokes back to the apex) - used for Spotlight's cone,
/// @p apex is the light position, @p direction the light's forward vector, @p length its attenuation range,
/// @p half_angle_degrees its outer cone angle
void debugDrawCone(
    glm::vec3 apex, glm::vec3 direction, float length, float half_angle_degrees, glm::vec4 color = {1.0f, 1.0f, 1.0f, 1.0f},
    int segments = 24
);

/**
 * @brief Queues a wireframe frustum for @p camera
 *
 * Built from the camera own fov/near/far and view matrix rather than by un-projecting NDC corners
 */
void debugDrawFrustum(const toast::Camera& camera, float aspect, glm::vec4 color = {1.0f, 1.0f, 0.0f, 1.0f});

}    // namespace renderer
