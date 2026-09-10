/**
 * @file brick_pool.hpp
 * @author dario
 * @date 08/09/2026
 *
 * @brief Storage for every brick in the world addressed by id
 */

#pragma once
#include "brick.hpp"
#include "voxel_constants.hpp"

#include <bit>
#include <cstdint>
#include <span>
#include <toast/export.hpp>
#include <vector>

namespace toast::voxel {

/// @brief Bytes of palette index per brick, one per voxel
inline constexpr size_t k_brick_material_bytes = static_cast<size_t>(k_brick_voxel_count);

/**
 * @brief Returned by BrickPool::allocate when the pool is full
 */
inline constexpr uint32_t k_invalid_brick = k_brick_payload_mask;

/**
 * @brief The material and occupancy of every brick
 *
 * Not thread-safe allocation, freeing and writes are single-threaded. Should apply all destruction
 * at the end of a physics step
 */
class TOAST_API BrickPool {
public:
	/**
	 * @brief Reserves storage for @p capacity bricks up front
	 */
	explicit BrickPool(uint32_t capacity);

	/**
	 * @brief Takes a brick from the pool, cleared to empty
	 *
	 * @returns a brick id, or k_invalid_brick when pool is exhausted
	 */
	[[nodiscard]]
	auto allocate() -> uint32_t;

	/// @brief Returns a brick to the pool. The id must have come from `allocate` and not been freed since
	void free(uint32_t id);

	/// @brief One palette index per voxel, indexed by `localIndex`
	[[nodiscard]]
	auto material(uint32_t id) -> std::span<uint8_t, k_brick_material_bytes>;

	[[nodiscard]]
	auto material(uint32_t id) const -> std::span<const uint8_t, k_brick_material_bytes>;

	[[nodiscard]]
	auto occupancy(uint32_t id) -> BrickOccupancy&;

	[[nodiscard]]
	auto occupancy(uint32_t id) const -> const BrickOccupancy&;

	[[nodiscard]]
	auto capacity() const noexcept -> uint32_t {
		return m_capacity;
	}

	/// @brief Bricks currently handed out
	[[nodiscard]]
	auto allocatedCount() const noexcept -> uint32_t {
		return m_next_unused - static_cast<uint32_t>(m_free_list.size());
	}

	[[nodiscard]]
	auto freeCount() const noexcept -> uint32_t {
		return m_capacity - allocatedCount();
	}

	[[nodiscard]]
	auto isValid(uint32_t id) const noexcept -> bool {
		return id < m_next_unused;
	}

private:
	void clearBrick(uint32_t id);

	uint32_t m_capacity = 0;

	/// Bump pointer over never allocated ids
	uint32_t m_next_unused = 0;

	std::vector<uint32_t> m_free_list;

	std::vector<uint8_t> m_material;
	std::vector<BrickOccupancy> m_occupancy;
};

/// justcause4
static_assert(std::endian::native == std::endian::little, "the brick material layer assumes a little-endian host");

}
