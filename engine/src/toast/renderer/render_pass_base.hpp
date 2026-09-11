/// @file IRenderPass.hpp
/// @author dario
/// @date 07/06/2026

#pragma once

#include "vulkan_common.hpp"

#include <atomic>
#include <string_view>

/// @brief Which of the renderer's two rendering scopes a pass records into
///
/// The scene is tonemapped once at the end, which splits passes in two: those exposed and tonemapped with
/// it, and those authored in display space that must not be
enum class RenderStage : uint8_t {
	/// Into the HDR scene target, before tonemapping. Scene geometry, the grid, world-space UI panels
	world,
	/// Into the final output image, after tonemapping. UI and debug overlays, whose colours are the literal
	/// values that should reach the screen
	overlay,
};

/**
 * @brief Interface for custom rendering passes
 */
class IRenderPass {
public:
	virtual ~IRenderPass() = default;

	/// @returns which rendering scope this pass belongs to; see RenderStage
	[[nodiscard]]
	virtual auto stage() const -> RenderStage {
		return RenderStage::world;
	}

	virtual void update(uint32_t frame_index, float dt) { }

	/// @brief Records work that must run before the renderer's main scope opens
	///
	/// Outside any rendering scope, so a pass can render into its own targets and transition them
	virtual void recordPre(vk::CommandBuffer cmd, uint32_t frame_index, uint32_t image_index) { }

	virtual void record(vk::CommandBuffer cmd, uint32_t frame_index, uint32_t image_index) = 0;

	/// @returns name shown in the editor's pass visibility popup
	[[nodiscard]]
	virtual auto name() const -> std::string_view {
		return "Pass";
	}

	/// Disabled passes are skipped by the renderer's record loop
	void setEnabled(bool enabled) noexcept { m_enabled.store(enabled, std::memory_order_relaxed); }

	[[nodiscard]]
	auto isEnabled() const noexcept -> bool {
		return m_enabled.load(std::memory_order_relaxed);
	}

private:
	std::atomic_bool m_enabled {true};
};
