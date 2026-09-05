/// @file shader_reflection.hpp
/// @author Xein
/// @date 17/07/2026

#pragma once

#include <cstdint>
#include <nlohmann/json.hpp>
#include <optional>
#include <slang.h>
#include <string>
#include <toast/export.hpp>
#include <vector>

namespace renderer {

/// Scalar/vector/matrix type of a uniform block member
enum class ShaderMemberType : uint8_t {
	unknown,
	bool_t,
	int_t,
	uint_t,
	float_t,
	vec2,
	vec3,
	vec4,
	mat3,
	mat4,
};

/// Vulkan descriptor kind of a shader binding
enum class ShaderBindingKind : uint8_t {
	uniform_buffer,
	storage_buffer,
	combined_image_sampler,
	sampled_image,
	sampler,
	storage_image,
	acceleration_structure,
};

struct ShaderInspectorMeta {
	bool reflected = false;    ///< only [Reflect] parameters are material-editable
	std::optional<float> range_min;
	std::optional<float> range_max;
	bool is_color = false;
	std::string display_name;
	std::string group;
	std::string subgroup;
	std::string unit;
	std::string default_fallback;    ///< "white" (default), "black", or "flat_normal" - Sampler2D fallback when unbound
	/// [Linear] - the texture carries data (normals, metalness, roughness, occlusion) rather than colour, so it
	/// must not be sRGB-encoded. Declared rather than inferred: roughness and occlusion are indistinguishable
	/// from albedo by fallback alone, and decoding them through sRGB corrupts the values silently
	bool linear_data = false;
};

struct ShaderBlockMember {
	std::string name;
	ShaderMemberType type = ShaderMemberType::unknown;
	uint32_t offset = 0;
	uint32_t size = 0;
	uint32_t element_count = 0;     ///< >0 when the member is an array
	uint32_t element_stride = 0;    ///< byte stride between array elements
	std::string engine_semantic;
	ShaderInspectorMeta inspector;
};

struct ShaderBinding {
	uint32_t set = 0;
	uint32_t binding = 0;
	std::string name;
	ShaderBindingKind kind = ShaderBindingKind::uniform_buffer;
	uint32_t count = 1;
	uint32_t size = 0;                         ///< byte size for uniform buffers
	std::vector<ShaderBlockMember> members;    ///< uniform buffer contents, in declaration order
	std::string engine_semantic;               ///< "frame" for engine-reserved set 0 bindings
	ShaderInspectorMeta inspector;
};

struct ShaderPushConstants {
	std::string name;
	uint32_t size = 0;
	std::vector<ShaderBlockMember> members;
};

struct ShaderEntryPoint {
	std::string name;
	std::string stage;    ///< "vertex" | "fragment" | "compute"
};

/**
 * @struct ShaderReflection
 * @brief Plain-data mirror of a compiled shader's layout, serializable to JSON
 *
 * Holds everything ShaderLayout needs to build pipeline layouts and everything the
 * editor needs to generate material schemas, without keeping any Slang objects alive
 */
struct ShaderReflection {
	std::vector<ShaderEntryPoint> entry_points;
	std::vector<ShaderBinding> bindings;      ///< declaration order
	std::vector<ShaderPushConstants> push_constants;
	std::vector<std::string> layout_order;    ///< global parameter names in declaration order

	// Exported so tests/renderer/02-reflection-round-trip.cpp can exercise the pair directly. That round trip
	// is the disk shader cache's correctness boundary: a kind that does not survive it produces a descriptor
	// set layout with the wrong type, which nothing in the engine reports and the driver rejects much later
	[[nodiscard]]
	auto TOAST_API toJson() const -> nlohmann::json;

	static auto TOAST_API fromJson(const nlohmann::json& json) -> std::optional<ShaderReflection>;
};

/// Walks a Slang program layout into a plain ShaderReflection
auto extractReflection(slang::ProgramLayout* layout) -> ShaderReflection;

/**
 * @brief Fills @p reflection's entry_points from the entry points a module actually defines
 *
 * Has to come from the IModule rather than the ProgramLayout: per Slang's docs an entry point declared with
 * [shader("...")] is not part of a module's linkage, so ProgramLayout::getEntryPointCount() on a module-only
 * composite always reports 0 - which silently left entry_points empty even though the emitted SPIR-V carried
 * every entry point. Anything selecting a pipeline variant by entry-point name depends on this
 */
void extractModuleEntryPoints(slang::IModule* module, ShaderReflection& reflection);

auto toString(ShaderMemberType type) -> std::string_view;
auto toString(ShaderBindingKind kind) -> std::string_view;
auto memberTypeFromString(std::string_view str) -> ShaderMemberType;
auto bindingKindFromString(std::string_view str) -> ShaderBindingKind;

}
