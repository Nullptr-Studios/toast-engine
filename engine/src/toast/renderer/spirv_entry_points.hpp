/**
 * @file spirv_entry_points.hpp
 * @author dario
 * @date 14/08/2026
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <toast/export.hpp>
#include <vector>

namespace renderer::spirv {

/// SPIR-V execution models, from the spec's OpEntryPoint operand
enum class ExecutionModel : uint32_t {
	vertex = 0,
	fragment = 4,
	compute = 5,
};

/**
 * @brief Names every entry point @p spirv declares for @p model, from its OpEntryPoint records
 *
 * Read from the module rather than from ShaderReflection: the string Vulkan matches against lives in the
 * SPIR-V, and reflection is a separate extraction that can disagree with it. Returns empty for anything that
 * is not SPIR-V, so a caller that cannot read the names falls back to what it was asked for
 */
[[nodiscard]]
auto TOAST_API entryPointNames(std::span<const std::byte> spirv, ExecutionModel model) -> std::vector<std::string>;

/**
 * @brief Resolves the entry point name to hand a VkPipelineShaderStageCreateInfo
 *
 * Slang renames a module's *only* entry point to "main", so a shader losing its second entry point breaks
 * every pass asking for the declared name - a pipeline that fails to create, and an editor that dies on
 * launch. Rather than hardcode "main" per call site, the request is matched against what the module declares
 *
 * @returns @p requested when it exists or nothing can be read; the single declared name when the request is
 *          absent and exactly one candidate exists; otherwise @p requested, so a typo still fails loudly
 */
[[nodiscard]]
auto TOAST_API resolveEntryPoint(
    std::span<const std::byte> spirv, ExecutionModel model, const std::string& requested, std::string_view debug_name
) -> std::string;

}
