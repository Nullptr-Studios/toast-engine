/**
 * @file volume.hpp
 * @author dario
 * @date 08/09/2026
 *
 * @brief A brickmap: an indirection grid over pooled bricks, and the unit physics simulates
 *
 * A volume is the collision shape *and* the render primitive. It is not a scene node and not an asset - a
 * `.tvox` is the immutable asset, a `VoxelNode` is the scene node that owns one of these, and debris carries
 * one with no node at all (`docs/voxel_plan.md` §4.3).
 *
 * **Copy-on-write is at brick granularity**, which is the difference between ten crates from one asset
 * costing one copy of a 128 KB indirection grid each and costing 18 MB of bricks each. An instance shares
 * every brick with its source until it is written to, and then copies only the bricks that were actually hit
 * (§4.8).
 *
 * See `docs/voxel_plan.md` §2 and §4
 */

#pragma once
#include "brick.hpp"
#include "brick_pool.hpp"
#include "voxel_constants.hpp"

#include <cstdint>
#include <glm/glm.hpp>
#include <toast/export.hpp>
#include <vector>

namespace toast::voxel {

class TOAST_API Volume {
public:
	/**
	 * @brief What one voxel write changed
	 *
	 * The shape is driven by what destruction needs (§13.9): the previous material so mass can be subtracted
	 * without a second lookup, and the two occupancy transitions because those - and only those - are when
	 * the acceleration structure has to be rebuilt. Carving voxels out of a brick that stays occupied never
	 * changes its AABB, which is the property that makes brick-granularity acceleration structures viable
	 */
	struct VoxelWrite {
		/// What was there before. `k_empty_palette_index` when the voxel was empty
		uint8_t previous_material = k_empty_palette_index;

		/// False when the write was a no-op, including a write outside the volume
		bool changed = false;

		/// The brick held nothing and now holds something
		bool brick_became_occupied = false;

		/// The brick held something and now holds nothing; its storage has been returned to the pool
		bool brick_became_empty = false;
	};

	/// @brief An empty volume of @p brick_dims bricks, drawing storage from @p pool
	Volume(BrickPool& pool, glm::uvec3 brick_dims);

	~Volume();

	Volume(const Volume&) = delete;
	auto operator=(const Volume&) -> Volume& = delete;
	Volume(Volume&& other) noexcept;
	auto operator=(Volume&& other) noexcept -> Volume&;

	/**
	 * @brief A writable instance sharing every brick with @p source until it is written to
	 *
	 * Copies the indirection grid and nothing else. Bricks the source owns become `shared` entries pointing at
	 * the same storage, so instantiating a model is proportional to its brick *count* rather than its brick
	 * *contents*.
	 *
	 * **@p source must outlive the instance and must not be written to.** That is exactly the asset case, and
	 * it is why there is no reference counting here: an immutable source needs none. An engine that lets a
	 * `.tvox` unload while instances of it are live would need it, and would know to add it
	 */
	[[nodiscard]]
	static auto instanceOf(const Volume& source) -> Volume;

	[[nodiscard]]
	auto brickDims() const noexcept -> glm::uvec3 {
		return m_brick_dims;
	}

	[[nodiscard]]
	auto voxelDims() const noexcept -> glm::uvec3 {
		return m_brick_dims * k_brick_dim;
	}

	[[nodiscard]]
	auto brickCount() const noexcept -> uint32_t {
		return static_cast<uint32_t>(m_entries.size());
	}

	[[nodiscard]]
	auto containsBrick(glm::ivec3 brick) const noexcept -> bool;

	[[nodiscard]]
	auto containsVoxel(glm::ivec3 voxel) const noexcept -> bool;

	/// @returns the indirection entry, or an empty entry for a brick outside the volume
	[[nodiscard]]
	auto entryAt(glm::ivec3 brick) const noexcept -> BrickEntry;

	/// @returns the palette index at @p voxel, or `k_empty_palette_index` outside the volume
	[[nodiscard]]
	auto materialAt(glm::ivec3 voxel) const noexcept -> uint8_t;

	[[nodiscard]]
	auto isSolidAt(glm::ivec3 voxel) const noexcept -> bool;

	/**
	 * @brief Writes one voxel, materialising and releasing bricks as needed
	 *
	 * A write outside the volume is silently a no-op rather than an assertion, because carving clips against
	 * a volume's bounds constantly - a blast sphere near an edge would otherwise need the caller to clip
	 * first, at every call site
	 */
	auto setVoxel(glm::ivec3 voxel, uint8_t material) -> VoxelWrite;

	/**
	 * @brief Makes a whole brick solid with one material, storing it as a `uniform` entry
	 *
	 * Four bytes instead of 576. A building's interior is mostly solid, so this is not a micro-optimisation -
	 * it is most of why a large static world fits in memory (§2.3)
	 */
	void setBrickUniform(glm::ivec3 brick, uint8_t material);

	/**
	 * @brief Collapses a pooled brick back to a `uniform` entry when every voxel shares one material
	 *
	 * Deliberately explicit rather than checked on every write: detecting it costs a scan of 512 bytes, and
	 * the caller that benefits is the bake (§4.5) stamping solid regions, not the carve loop
	 *
	 * @returns true when the brick was collapsed
	 */
	auto tryCollapseUniform(glm::ivec3 brick) -> bool;

	/**
	 * @brief The occupancy of one brick, or null when it holds nothing
	 *
	 * A `uniform` brick has no pool storage to point at, so this returns `&k_full_brick` - which is what the
	 * constant exists for. That is what lets `interior()` and `surfaceShell()` see across a seam into a solid
	 * neighbour without the volume having to synthesise a brick for them.
	 *
	 * @warning **Compare what it points at, never the pointer itself.** `k_full_brick` is an `inline constexpr`
	 *          in a header, and the engine is a shared library - so `toast_engine` and anything linking against
	 *          it each hold their own copy of it at their own address. Dereferencing the returned pointer is
	 *          correct across that boundary; `result == &k_full_brick` is not, and fails only in a shared-library
	 *          build, which is the one that ships
	 */
	[[nodiscard]]
	auto occupancyPointer(glm::ivec3 brick) const noexcept -> const BrickOccupancy*;

	/// @brief The six bricks abutting @p brick, for `interior()` and `surfaceShell()`
	[[nodiscard]]
	auto neighbourhoodOf(glm::ivec3 brick) const noexcept -> BrickNeighbourhood;

	[[nodiscard]]
	auto solidVoxelCount() const -> uint32_t;

	/// @brief Bricks this volume owns outright, and will return to the pool when it is destroyed
	[[nodiscard]]
	auto ownedBrickCount() const -> uint32_t;

	/// @brief Bricks still shared with the source this volume was instantiated from
	[[nodiscard]]
	auto sharedBrickCount() const -> uint32_t;

	[[nodiscard]]
	auto pool() const noexcept -> BrickPool* {
		return m_pool;
	}

private:
	[[nodiscard]]
	auto entryIndex(glm::ivec3 brick) const noexcept -> uint32_t;

	/// @brief Ensures the entry at @p entry_index owns writable storage, and returns its brick id
	auto makeWritable(uint32_t entry_index) -> uint32_t;

	void releaseOwned();

	BrickPool* m_pool = nullptr;
	glm::uvec3 m_brick_dims {0};

	/// One entry per brick slot, x fastest then y then z. Zero-initialised, which reads as entirely empty
	std::vector<BrickEntry> m_entries;
};

}
