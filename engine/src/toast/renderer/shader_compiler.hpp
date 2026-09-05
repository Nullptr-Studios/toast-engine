/// @file ShaderCompiler.hpp
/// @author dario
/// @date 17/05/2026

#pragma once

#include "shader_reflection.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <toast/uid.hpp>
#include <vector>

namespace renderer {

struct CompiledShaderCode {
	std::vector<std::byte> spirv;
	ShaderReflection reflection;
	std::vector<std::string> dependencies;    // Virtual URIs the module depends on
};

class ShaderCompiler {
public:
	/**
	 * @brief Compiles a Slang module from in-memory source to SPIR-V
	 * @param uid UID of the shader asset
	 * @param source Slang source code
	 * @param source_uri Virtual URI of the source
	 */
	static auto compile(toast::UID uid, std::string_view source, std::string_view source_uri) -> CompiledShaderCode;

	/**
	 * @brief Declares which optional device features shaders may compile against
	 *
	 * A SPIR-V module declaring a capability the device lacks is invalid and pipeline creation fails rather
	 * than degrading, so a shader that traces rays has to be compiled differently, not gated at runtime. The
	 * flags become preprocessor defines (TOAST_RAY_QUERY) and are folded into the shader cache key, so a
	 * machine that gains or loses the feature recompiles rather than loading a module built for the other
	 *
	 * @note Call before any compilation, i.e. before ShaderCache::compileAllAtStartup(). VulkanCore does this
	 * once the device is up
	 */
	static void setRayQueryAvailable(bool available);

	[[nodiscard]]
	static auto isRayQueryAvailable() -> bool;

	/// @returns bits identifying the current feature set, mixed into every cache entry's hash
	[[nodiscard]]
	static auto featureHash() -> uint64_t;
};

}
