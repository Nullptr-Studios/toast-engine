/**
 * @file brick.hpp
 * @author dario
 * @date 08/09/2026
 *
 * @brief Occupancy bit operations over a 8x8x8 brick
 *
 * The foundation of the whole voxel system: primary-visibility traversal, narrowphase contact generation,
 * surface classification, connectivity and flood fill all reduce to the operations in this file. Every one of
 * them is a handful of 64-bit words rather than a loop over 512 voxels, which is the entire reason the brick
 * dimension is 8 - see `voxel_constants.hpp` and `docs/voxel_plan.md` §2.2.
 *
 * **Bit layout. A brick is eight `uint64_t`, one per z-slice. Within a slice, the bit for `(x, y)` is at
 * index `y * 8 + x`. So a whole 8x8 z-slice is one word, a y-row is one byte, and x is the bit within that
 * byte. Nothing here is valid if that layout changes.
 */

#pragma once
#include "voxel_constants.hpp"

#include <array>
#include <bit>
#include <cassert>
#include <cstdint>

namespace toast::voxel {

/**
 * @brief state of one brick's 512 voxels
 *
 * Index by z-slice, the bit for (x, y) within a slice is y * 8 + x
 */
struct BrickOccupancy {
	std::array<uint64_t, k_brick_dim> slices {};

	[[nodiscard]]
	constexpr auto operator[](uint32_t z) noexcept -> uint64_t& {
		assert(z < k_brick_dim);
		return slices[z];
	}

	[[nodiscard]]
	constexpr auto operator[](uint32_t z) const noexcept -> const uint64_t& {
		assert(z < k_brick_dim);
		return slices[z];
	}

	[[nodiscard]]
	constexpr auto operator==(const BrickOccupancy&) const noexcept -> bool = default;
};

/// @brief An entirely empty brick
inline constexpr BrickOccupancy k_empty_brick {};

/// @brief An entirely solid brick
inline constexpr BrickOccupancy k_full_brick {
  {~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull}
};

/// @brief Bits of a z-slice whose voxel has x == 0
inline constexpr uint64_t k_column_x_min = 0x0101010101010101ull;

/// @brief Bits of a z-slice whose voxel has x == 7
inline constexpr uint64_t k_column_x_max = 0x8080808080808080ull;

/// @brief Bits of a z-slice whose voxel has y == 0
inline constexpr uint64_t k_row_y_min = 0x00000000000000FFull;

/// @brief Bits of a z-slice whose voxel has y == 7
inline constexpr uint64_t k_row_y_max = 0xFF00000000000000ull;


// Addressing

[[nodiscard]]
constexpr auto localIndex(const uint32_t x, const uint32_t y, const uint32_t z) noexcept -> uint32_t {
	assert(x < k_brick_dim && y < k_brick_dim && z < k_brick_dim);
	return x + (y * k_brick_dim) + (z * k_brick_dim * k_brick_dim);
}

/// @brief Brick-local coordinate
struct BrickCoord {
	uint32_t x = 0;
	uint32_t y = 0;
	uint32_t z = 0;

	[[nodiscard]]
	constexpr auto operator==(const BrickCoord&) const noexcept -> bool = default;
};

/// @brief Inverse of localIndex
[[nodiscard]]
constexpr auto localFromIndex(uint32_t index) noexcept -> BrickCoord {
	assert(index < k_brick_voxel_count);
	return BrickCoord {index & 7u, (index >> 3u) & 7u, index >> 6u};
}

/// @brief Bit position of (x, y) within its zslice
[[nodiscard]]
constexpr auto sliceBit(uint32_t x, uint32_t y) noexcept -> uint32_t {
	assert(x < k_brick_dim && y < k_brick_dim);
	return y * k_brick_dim + x;
}

// Single-voxel access

[[nodiscard]]
constexpr auto isSolid(const BrickOccupancy& brick, uint32_t x, uint32_t y, uint32_t z) noexcept -> bool {
	assert(z < k_brick_dim);
	return ((brick[z] >> sliceBit(x, y)) & 1ull) != 0ull;
}

[[nodiscard]]
constexpr auto isSolid(const BrickOccupancy& brick, uint32_t local_index) noexcept -> bool {
	const BrickCoord c = localFromIndex(local_index);
	return isSolid(brick, c.x, c.y, c.z);
}

constexpr void setSolid(BrickOccupancy& brick, uint32_t x, uint32_t y, uint32_t z, bool solid) noexcept {
	assert(z < k_brick_dim);
	const uint64_t bit = 1ull << sliceBit(x, y);
	if (solid) {
		brick[z] |= bit;
	} else {
		brick[z] &= ~bit;
	}
}

constexpr void setSolid(BrickOccupancy& brick, uint32_t local_index, bool solid) noexcept {
	const BrickCoord c = localFromIndex(local_index);
	setSolid(brick, c.x, c.y, c.z, solid);
}

// Whole-brick

[[nodiscard]]
constexpr auto isEmpty(const BrickOccupancy& brick) noexcept -> bool {
	for (uint64_t slice : brick.slices) {
		if (slice != 0ull) {
			return false;
		}
	}
	return true;
}

[[nodiscard]]
constexpr auto isFull(const BrickOccupancy& brick) noexcept -> bool {
	for (uint64_t slice : brick.slices) {
		if (slice != ~0ull) {
			return false;
		}
	}
	return true;
}

/// @brief Number of solid voxels
[[nodiscard]]
constexpr auto popCount(const BrickOccupancy& brick) noexcept -> uint32_t {
	uint32_t total = 0;
	for (uint64_t slice : brick.slices) {
		total += static_cast<uint32_t>(std::popcount(slice));
	}
	return total;
}

[[nodiscard]]
constexpr auto operator&(const BrickOccupancy& a, const BrickOccupancy& b) noexcept -> BrickOccupancy {
	BrickOccupancy out {};
	for (uint32_t z = 0; z < k_brick_dim; ++z) {
		out[z] = a[z] & b[z];
	}
	return out;
}

[[nodiscard]]
constexpr auto operator|(const BrickOccupancy& a, const BrickOccupancy& b) noexcept -> BrickOccupancy {
	BrickOccupancy out {};
	for (uint32_t z = 0; z < k_brick_dim; ++z) {
		out[z] = a[z] | b[z];
	}
	return out;
}

[[nodiscard]]
constexpr auto operator~(const BrickOccupancy& a) noexcept -> BrickOccupancy {
	BrickOccupancy out {};
	for (uint32_t z = 0; z < k_brick_dim; ++z) {
		out[z] = ~a[z];
	}
	return out;
}


/// @brief Occupancy of each voxel -X neighbour
[[nodiscard]]
constexpr auto neighboursNegX(const BrickOccupancy& brick, const BrickOccupancy* adjacent = nullptr) noexcept -> BrickOccupancy {
	BrickOccupancy out {};
	for (uint32_t z = 0; z < k_brick_dim; ++z) {
		out[z] = (brick[z] & ~k_column_x_max) << 1;
		if (adjacent != nullptr) {
			out[z] |= ((*adjacent)[z] >> 7) & k_column_x_min;
		}
	}
	return out;
}

/// @brief Occupancy of each voxel +X neighbour
[[nodiscard]]
constexpr auto neighboursPosX(const BrickOccupancy& brick, const BrickOccupancy* adjacent = nullptr) noexcept -> BrickOccupancy {
	BrickOccupancy out {};
	for (uint32_t z = 0; z < k_brick_dim; ++z) {
		out[z] = (brick[z] & ~k_column_x_min) >> 1;
		if (adjacent != nullptr) {
			out[z] |= ((*adjacent)[z] << 7) & k_column_x_max;
		}
	}
	return out;
}

/// @brief Occupancy of each voxel -Y neighbour
[[nodiscard]]
constexpr auto neighboursNegY(const BrickOccupancy& brick, const BrickOccupancy* adjacent = nullptr) noexcept -> BrickOccupancy {
	BrickOccupancy out {};
	for (uint32_t z = 0; z < k_brick_dim; ++z) {
		out[z] = brick[z] << k_brick_dim;
		if (adjacent != nullptr) {
			out[z] |= (*adjacent)[z] >> 56;
		}
	}
	return out;
}

/// @brief Occupancy of each voxel +Y neighbour
[[nodiscard]]
constexpr auto neighboursPosY(const BrickOccupancy& brick, const BrickOccupancy* adjacent = nullptr) noexcept -> BrickOccupancy {
	BrickOccupancy out {};
	for (uint32_t z = 0; z < k_brick_dim; ++z) {
		out[z] = brick[z] >> k_brick_dim;
		if (adjacent != nullptr) {
			out[z] |= (*adjacent)[z] << 56;
		}
	}
	return out;
}

/// @brief Occupancy of each voxel -Z neighbour
[[nodiscard]]
constexpr auto neighboursNegZ(const BrickOccupancy& brick, const BrickOccupancy* adjacent = nullptr) noexcept -> BrickOccupancy {
	BrickOccupancy out {};
	out[0] = adjacent != nullptr ? (*adjacent)[k_brick_dim - 1] : 0ull;
	for (uint32_t z = 1; z < k_brick_dim; ++z) {
		out[z] = brick[z - 1];
	}
	return out;
}

/// @brief Occupancy of each voxel +Z neighbour
[[nodiscard]]
constexpr auto neighboursPosZ(const BrickOccupancy& brick, const BrickOccupancy* adjacent = nullptr) noexcept -> BrickOccupancy {
	BrickOccupancy out {};
	for (uint32_t z = 0; z + 1 < k_brick_dim; ++z) {
		out[z] = brick[z + 1];
	}
	out[k_brick_dim - 1] = adjacent != nullptr ? (*adjacent)[0] : 0ull;
	return out;
}

/**
 * @brief The six bricks abutting one brick's faces, any of which may be absent
 */
struct BrickNeighbourhood {
	const BrickOccupancy* neg_x = nullptr;
	const BrickOccupancy* pos_x = nullptr;
	const BrickOccupancy* neg_y = nullptr;
	const BrickOccupancy* pos_y = nullptr;
	const BrickOccupancy* neg_z = nullptr;
	const BrickOccupancy* pos_z = nullptr;
};

/**
 * @brief Solid voxels all six of whose neighbours are also solid
 *
 * These generate no contacts and carry no surface classification
 */
[[nodiscard]]
constexpr auto interior(const BrickOccupancy& brick, const BrickNeighbourhood& neighbours = {}) noexcept -> BrickOccupancy {
	return brick & neighboursNegX(brick, neighbours.neg_x) & neighboursPosX(brick, neighbours.pos_x) &
	       neighboursNegY(brick, neighbours.neg_y) & neighboursPosY(brick, neighbours.pos_y) &
	       neighboursNegZ(brick, neighbours.neg_z) & neighboursPosZ(brick, neighbours.pos_z);
}

/// @brief Solid voxels with at least one empty face neighbour
[[nodiscard]]
constexpr auto surfaceShell(const BrickOccupancy& brick, const BrickNeighbourhood& neighbours = {}) noexcept -> BrickOccupancy {
	return brick & ~interior(brick, neighbours);
}

/**
 * @brief Grows a set by one voxel in all six directions
 */
[[nodiscard]]
constexpr auto dilate(const BrickOccupancy& brick) noexcept -> BrickOccupancy {
	return brick | neighboursNegX(brick) | neighboursPosX(brick) | neighboursNegY(brick) | neighboursPosY(brick) |
	       neighboursNegZ(brick) | neighboursPosZ(brick);
}

// Face masks

/// @brief x == 0 plane, bit z * 8 + y
[[nodiscard]]
constexpr auto faceNegX(const BrickOccupancy& brick) noexcept -> uint64_t {
	uint64_t out = 0;
	for (uint32_t z = 0; z < k_brick_dim; ++z) {
		for (uint32_t y = 0; y < k_brick_dim; ++y) {
			if (((brick[z] >> sliceBit(0, y)) & 1ull) != 0ull) {
				out |= 1ull << (z * k_brick_dim + y);
			}
		}
	}
	return out;
}

/// @brief x == 7 plane, bit z * 8 + y
[[nodiscard]]
constexpr auto facePosX(const BrickOccupancy& brick) noexcept -> uint64_t {
	uint64_t out = 0;
	for (uint32_t z = 0; z < k_brick_dim; ++z) {
		for (uint32_t y = 0; y < k_brick_dim; ++y) {
			if (((brick[z] >> sliceBit(k_brick_dim - 1, y)) & 1ull) != 0ull) {
				out |= 1ull << (z * k_brick_dim + y);
			}
		}
	}
	return out;
}

/// @brief y == 0 plane, bit z * 8 + x
[[nodiscard]]
constexpr auto faceNegY(const BrickOccupancy& brick) noexcept -> uint64_t {
	uint64_t out = 0;
	for (uint32_t z = 0; z < k_brick_dim; ++z) {
		out |= (brick[z] & k_row_y_min) << (z * k_brick_dim);
	}
	return out;
}

/// @brief y == 7 plane, bit z * 8 + x
[[nodiscard]]
constexpr auto facePosY(const BrickOccupancy& brick) noexcept -> uint64_t {
	uint64_t out = 0;
	for (uint32_t z = 0; z < k_brick_dim; ++z) {
		out |= (brick[z] >> 56) << (z * k_brick_dim);
	}
	return out;
}

/// @brief z == 0 plane, bit y * 8 + x
[[nodiscard]]
constexpr auto faceNegZ(const BrickOccupancy& brick) noexcept -> uint64_t {
	return brick[0];
}

/// @brief z == 7 plane, bit y * 8 + x
[[nodiscard]]
constexpr auto facePosZ(const BrickOccupancy& brick) noexcept -> uint64_t {
	return brick[k_brick_dim - 1];
}

/// @brief All six face planes of one brick in cache
struct BrickFaces {
	uint64_t neg_x = 0;
	uint64_t pos_x = 0;
	uint64_t neg_y = 0;
	uint64_t pos_y = 0;
	uint64_t neg_z = 0;
	uint64_t pos_z = 0;

	[[nodiscard]]
	constexpr auto operator==(const BrickFaces&) const noexcept -> bool = default;
};

[[nodiscard]]
constexpr auto computeFaces(const BrickOccupancy& brick) noexcept -> BrickFaces {
	return BrickFaces {faceNegX(brick), facePosX(brick), faceNegY(brick), facePosY(brick), faceNegZ(brick), facePosZ(brick)};
}

/**
 * @brief Whether two abutting faces share at least one solid voxel
 */
[[nodiscard]]
constexpr auto facesConnect(uint64_t face_a, uint64_t face_b) noexcept -> bool {
	return (face_a & face_b) != 0ull;
}

}
