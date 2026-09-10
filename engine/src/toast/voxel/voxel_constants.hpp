/**
 * @file voxel_constants.hpp
 * @author dario
 * @date 08/09/2026
 *
 * @brief Every fixed size in the voxel system
 */

#pragma once

#include <cstdint>

namespace toast::voxel {

/// @brief length of one voxel
///
inline constexpr float k_voxel_size = 0.1f;

/// @brief Voxels per brick edge
///
inline constexpr uint32_t k_brick_dim = 8;

/// @brief Voxels in one brick
inline constexpr uint32_t k_brick_voxel_count = k_brick_dim * k_brick_dim * k_brick_dim;

/// @brief Edge length of one brick
inline constexpr float k_brick_size = static_cast<float>(k_brick_dim) * k_voxel_size;

/// @brief Bricks per region edge
///
/// 32 bricks is 256 voxels
inline constexpr uint32_t k_region_dim_bricks = 32;

/// @brief Voxels per region edge
inline constexpr uint32_t k_region_dim_voxels = k_region_dim_bricks * k_brick_dim;

/// @brief Edge length of one region
inline constexpr float k_region_size = static_cast<float>(k_region_dim_bricks) * k_brick_size;

/// @brief Entries in one palette BYTE
inline constexpr uint32_t k_palette_size = 256;

/// @brief Reserved in every palette: not a material, and what a cleared voxel holds
inline constexpr uint8_t k_empty_palette_index = 0;

/**
 * @brief What one indirection-grid entry refers to
 */
enum class BrickTag : uint32_t {
	/// Nothing here
	empty = 0,
	/// Fully solid, one material throughout; the payload is a palette index, not a brick id
	uniform = 1,
	/// A real brick in the pool, owned by the source asset. Read freely; copy before writing
	shared = 2,
	/// A real brick in the pool, owned by this volume. Writable in place
	owned = 3,
};

/// @brief Bits of a BrickEntry given over to the tag
inline constexpr uint32_t k_brick_tag_bits = 2;

/// @brief Mask of a BrickEntry payload
inline constexpr uint32_t k_brick_payload_mask = (1u << (32 - k_brick_tag_bits)) - 1u;

/**
 * @brief One entry of a volume indirection grid
 */
struct BrickEntry {
	uint32_t value = 0;

	[[nodiscard]]
	static constexpr auto make(BrickTag tag, uint32_t payload) noexcept -> BrickEntry {
		return BrickEntry {(payload & k_brick_payload_mask) << k_brick_tag_bits | static_cast<uint32_t>(tag)};
	}

	[[nodiscard]]
	constexpr auto tag() const noexcept -> BrickTag {
		return static_cast<BrickTag>(value & ((1u << k_brick_tag_bits) - 1u));
	}

	/// @returns a palette index when `tag()` is `uniform`, a brick pool id when it is `shared` or `owned`
	[[nodiscard]]
	constexpr auto payload() const noexcept -> uint32_t {
		return value >> k_brick_tag_bits;
	}

	/// @returns true when this entry refers to a brick in the pool, whether shared or owned
	[[nodiscard]]
	constexpr auto isPooled() const noexcept -> bool {
		const auto t = tag();
		return t == BrickTag::shared || t == BrickTag::owned;
	}

	[[nodiscard]]
	constexpr auto operator==(const BrickEntry&) const noexcept -> bool = default;
};

// The whole of brick.hpp rests on a z-slice being exactly one 64-bit word
static_assert(k_brick_dim * k_brick_dim == 64, "a brick z-slice must be exactly 64 bits");
static_assert(k_brick_voxel_count == 512, "a brick must be 512 voxels");
static_assert(sizeof(BrickEntry) == 4, "an indirection entry must be four bytes");
static_assert(BrickEntry {}.tag() == BrickTag::empty, "a zeroed indirection grid must read as empty");

}
