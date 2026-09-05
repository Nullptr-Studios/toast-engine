/// @file vertex.hpp
/// @author dario
/// @date 07/06/2026

#pragma once

#include <array>
#include <cstddef>
#include <glm/glm.hpp>

namespace renderer {

/// @brief Vertex with position, normals, UVs, tangents, and colors for mesh rendering
struct Vertex {
	glm::vec<3, float, glm::packed_highp> position;
	glm::vec<3, float, glm::packed_highp> normal;
	glm::vec<2, float, glm::packed_highp> uv;
	glm::vec<4, float, glm::packed_highp> tangent;
	glm::vec<3, float, glm::packed_highp> color;
};

/**
 * @brief Per-vertex skinning influences, kept in a second vertex buffer binding
 *
 * Deliberately not folded into Vertex: only skinned meshes have these, and inlining them would cost every
 * static mesh ~40% more vertex memory for data its shader never reads. Skinned meshes bind this as binding 1
 * alongside the shared Vertex stream, and the pipeline only declares it when the shader consumes joints
 *
 * glTF permits joint indices as unsigned byte or short; both widen into uint16 here. Weights are stored as
 * floats already normalised to sum to 1 by the importer
 */
struct SkinVertex {
	glm::vec<4, uint16_t, glm::packed_highp> joints {0, 0, 0, 0};
	glm::vec<4, float, glm::packed_highp> weights {0.0f, 0.0f, 0.0f, 0.0f};
};

// skinning.slang reads both streams through a ByteAddressBuffer with these strides and offsets hard-coded,
// because it cannot describe them any other way: as vertex attributes they are packed, with every offset
// spelled out in a VkVertexInputAttributeDescription, while a Slang struct view of the same fields falls
// under std430 and pads every float3 to 16 bytes - 80 and 32 rather than 60 and 24. A field reordered or
// widened here silently shifts what the compute pass reads, so the numbers are pinned on this side too
static_assert(sizeof(Vertex) == 60, "skinning.slang's kVertexStride");
static_assert(offsetof(Vertex, position) == 0, "skinning.slang's kVertexPositionOffset");
static_assert(offsetof(Vertex, normal) == 12, "skinning.slang's kVertexNormalOffset");
static_assert(offsetof(Vertex, uv) == 24, "skinning.slang's kVertexUvOffset");
static_assert(offsetof(Vertex, tangent) == 32, "skinning.slang's kVertexTangentOffset");
static_assert(offsetof(Vertex, color) == 48, "skinning.slang's kVertexColorOffset");

static_assert(sizeof(SkinVertex) == 24, "skinning.slang's kSkinStride");
static_assert(offsetof(SkinVertex, joints) == 0, "skinning.slang's kSkinJointsOffset");
static_assert(offsetof(SkinVertex, weights) == 8, "skinning.slang's kSkinWeightsOffset");

}
