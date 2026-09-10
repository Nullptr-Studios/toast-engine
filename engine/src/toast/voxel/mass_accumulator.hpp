/**
 * @file mass_accumulator.hpp
 * @author dario
 * @date 08/09/2026
 *
 * @brief incremental mass properties for a body made of voxels
 *
 * Centre of mass and the inertia tensor are sums over solid voxels
 *
 * The part that makes it more than "incremental" is that the sums are kept as **integers**. Voxels sit on a
 * lattice and palette densities are whole numbers of kg/m³, so every add and every remove is exact and a body
 * carved ten thousand times has accumulated no error at all. In floating point the same scheme drifts, and
 * the drift presents as an object that slowly begins to spin about the wrong axis - a symptom that appears
 * long after the code responsible and looks like a solver bug.
 *
 * Ten sums are enough for the whole tensor: the zeroth moment, three first moments and six second moments.
 * Everything metric - kilograms, metres, kg·m² - is derived once in `resolve()`, so the conversion to float
 * happens at the end rather than being accumulated.
 *
 * See `docs/voxel_plan.md` §2.4 and §13.1
 */

#pragma once
#include "voxel_constants.hpp"

#include <cassert>
#include <cstdint>
#include <glm/glm.hpp>

namespace toast::voxel {

/**
 * @brief Headroom check for the second moments
 *
 * The dangerous term is `sum(density * coordinate²)`. For a fully solid cube of edge `E` voxels at the
 * highest density the palette can express (65535 kg/m³, denser than osmium), that sum is about
 * `65535 * E⁵ / 3`, which reaches `int64`'s range at roughly `E = 670` - a 67 m solid body at an absurd
 * density, and far beyond any dynamic body this engine will produce. The static world never queries mass
 * properties at all (`docs/voxel_plan.md` §13.10), so it is not bounded by this.
 *
 * Asserted rather than assumed, with a factor of two spare, because silent `int64` overflow would corrupt an
 * inertia tensor in a way that reads as a physics bug
 */
inline constexpr int64_t k_moment_safe_limit = int64_t {1} << 62;

/**
 * @brief The ten integer moments of a voxel body, in **lattice coordinates**
 *
 * Coordinates are integer voxel indices, not metres. Keeping them integral is what keeps the sums exact; the
 * half-voxel offset that places a voxel's mass at its *centre* rather than its corner is uniform across every
 * voxel, so it is applied analytically in `resolve()` instead of being carried per sample.
 *
 * "Density" throughout is the palette's `uint16` kg/m³. Mass never appears here in kilograms - `mass` is a
 * plain sum of densities, and one multiplication by the voxel volume turns it into one
 */
struct MassMoments {
	/// Sum of densities. Zero exactly when the body has no solid voxels
	int64_t mass = 0;

	/// Sum of `density * coordinate`
	int64_t m_x = 0;
	int64_t m_y = 0;
	int64_t m_z = 0;

	/// Sum of `density * coordinate²`
	int64_t m_xx = 0;
	int64_t m_yy = 0;
	int64_t m_zz = 0;

	/// Sum of `density * coordinate_a * coordinate_b`
	int64_t m_xy = 0;
	int64_t m_xz = 0;
	int64_t m_yz = 0;

	[[nodiscard]]
	constexpr auto operator==(const MassMoments&) const noexcept -> bool = default;

	[[nodiscard]]
	constexpr auto isEmpty() const noexcept -> bool {
		return mass == 0;
	}

	/// @brief Accumulates one voxel at lattice coordinate @p x, @p y, @p z with the given @p density
	constexpr void add(int32_t x, int32_t y, int32_t z, uint32_t density) noexcept {
		accumulate(x, y, z, static_cast<int64_t>(density));
	}

	/**
	 * @brief Removes one voxel, exactly undoing the `add` that placed it
	 *
	 * The density must be the one the voxel was added with - which the caller has, because it is the palette
	 * entry it is about to clear
	 */
	constexpr void remove(int32_t x, int32_t y, int32_t z, uint32_t density) noexcept {
		accumulate(x, y, z, -static_cast<int64_t>(density));
		assert(mass >= 0);    // removing more than was added is a bookkeeping bug, not a valid state
	}

	constexpr auto operator+=(const MassMoments& other) noexcept -> MassMoments& {
		mass += other.mass;
		m_x += other.m_x;
		m_y += other.m_y;
		m_z += other.m_z;
		m_xx += other.m_xx;
		m_yy += other.m_yy;
		m_zz += other.m_zz;
		m_xy += other.m_xy;
		m_xz += other.m_xz;
		m_yz += other.m_yz;
		checkHeadroom();
		return *this;
	}

	[[nodiscard]]
	constexpr auto operator+(const MassMoments& other) const noexcept -> MassMoments {
		MassMoments out = *this;
		out += other;
		return out;
	}

	/**
	 * @brief The same moments expressed in a coordinate frame translated by @p ox, @p oy, @p oz
	 *
	 * The parallel-axis theorem, and the reason moments are stored **per brick** in brick-local coordinates:
	 * a brick's own sums never exceed a few billion whatever the body's size, and a body's - or a fragment's,
	 * after a split - are the sum of its bricks' moments each shifted by that brick's offset.
	 *
	 * Brick offsets are integer multiples of the brick dimension, so the shift stays in integers and stays
	 * exact. That is what makes splitting a body **O(bricks) rather than O(voxels)** (`docs/voxel_plan.md`
	 * §13.10)
	 */
	[[nodiscard]]
	constexpr auto shifted(int32_t ox, int32_t oy, int32_t oz) const noexcept -> MassMoments {
		const int64_t dx = ox;
		const int64_t dy = oy;
		const int64_t dz = oz;

		MassMoments out;
		out.mass = mass;
		out.m_x = m_x + dx * mass;
		out.m_y = m_y + dy * mass;
		out.m_z = m_z + dz * mass;
		out.m_xx = m_xx + 2 * dx * m_x + dx * dx * mass;
		out.m_yy = m_yy + 2 * dy * m_y + dy * dy * mass;
		out.m_zz = m_zz + 2 * dz * m_z + dz * dz * mass;
		out.m_xy = m_xy + dx * m_y + dy * m_x + dx * dy * mass;
		out.m_xz = m_xz + dx * m_z + dz * m_x + dx * dz * mass;
		out.m_yz = m_yz + dy * m_z + dz * m_y + dy * dz * mass;
		out.checkHeadroom();
		return out;
	}

private:
	constexpr void accumulate(int32_t x, int32_t y, int32_t z, int64_t signed_density) noexcept {
		const int64_t lx = x;
		const int64_t ly = y;
		const int64_t lz = z;

		mass += signed_density;
		m_x += signed_density * lx;
		m_y += signed_density * ly;
		m_z += signed_density * lz;
		m_xx += signed_density * lx * lx;
		m_yy += signed_density * ly * ly;
		m_zz += signed_density * lz * lz;
		m_xy += signed_density * lx * ly;
		m_xz += signed_density * lx * lz;
		m_yz += signed_density * ly * lz;
		checkHeadroom();
	}

	constexpr void checkHeadroom() const noexcept {
		assert(m_xx < k_moment_safe_limit && m_yy < k_moment_safe_limit && m_zz < k_moment_safe_limit);
	}
};

/// @brief Mass properties in physical units, derived from a `MassMoments`
struct MassProperties {
	/// Total mass, kilograms
	float mass = 0.0f;

	/// Centre of mass in volume-local metres, measured from the volume's lattice origin
	glm::vec3 center_of_mass {0.0f};

	/// Inertia tensor about the centre of mass, kg·m². Symmetric
	glm::mat3 inertia {0.0f};
};

/**
 * @brief Converts integer moments into kilograms, metres and kg·m²
 *
 * Three things happen here that the integer sums deliberately do not carry:
 *
 * - **The half-voxel offset.** A voxel's mass sits at its centre, and the lattice coordinate names its
 *   corner. The offset is the same for every voxel, so it factors out of the sums.
 * - **Each voxel's own inertia.** Summing point masses at voxel centres gives the tensor of a lattice of
 *   points, not of a body made of cubes. A cube of edge `s` has `m·s²/6` about each of its own axes, and
 *   because that term does not depend on where the voxel is, it is one addition to the diagonal for the
 *   whole body. Without it the tensor is wrong for any body only a few voxels thick.
 * - **The shift to the centre of mass**, since the sums are taken about the lattice origin.
 *
 * Computed in double and narrowed once at the end, so the float result carries one rounding rather than a
 * body's worth of them
 */
[[nodiscard]]
inline auto resolve(const MassMoments& moments, float voxel_size = k_voxel_size) -> MassProperties {
	MassProperties out;
	if (moments.mass <= 0) {
		return out;
	}

	const double s = voxel_size;
	const double s3 = s * s * s;
	const double s5 = s3 * s * s;

	const double density_sum = static_cast<double>(moments.mass);
	const double total_mass = density_sum * s3;

	// Second moments about the lattice origin, in lattice units, with the half-voxel offset folded in:
	//   sum(d * (l_a + 1/2) * (l_b + 1/2)) = m_ab + (m_a + m_b)/2 + density_sum/4
	const auto product = [&](int64_t m_ab, int64_t m_a, int64_t m_b) {
		return static_cast<double>(m_ab) + 0.5 * (static_cast<double>(m_a) + static_cast<double>(m_b)) + 0.25 * density_sum;
	};

	const double p_xx = product(moments.m_xx, moments.m_x, moments.m_x);
	const double p_yy = product(moments.m_yy, moments.m_y, moments.m_y);
	const double p_zz = product(moments.m_zz, moments.m_z, moments.m_z);
	const double p_xy = product(moments.m_xy, moments.m_x, moments.m_y);
	const double p_xz = product(moments.m_xz, moments.m_x, moments.m_z);
	const double p_yz = product(moments.m_yz, moments.m_y, moments.m_z);

	const double com_x = (static_cast<double>(moments.m_x) / density_sum + 0.5) * s;
	const double com_y = (static_cast<double>(moments.m_y) / density_sum + 0.5) * s;
	const double com_z = (static_cast<double>(moments.m_z) / density_sum + 0.5) * s;

	// Each voxel's inertia about its own centre: a cube of edge s contributes m * s² / 6 to every diagonal
	const double self_term = total_mass * s * s / 6.0;

	// About the lattice origin
	double i_xx = s5 * (p_yy + p_zz) + self_term;
	double i_yy = s5 * (p_xx + p_zz) + self_term;
	double i_zz = s5 * (p_xx + p_yy) + self_term;
	double i_xy = -s5 * p_xy;
	double i_xz = -s5 * p_xz;
	double i_yz = -s5 * p_yz;

	// Shifted to the centre of mass. The self term is position-independent and correctly rides along
	i_xx -= total_mass * (com_y * com_y + com_z * com_z);
	i_yy -= total_mass * (com_x * com_x + com_z * com_z);
	i_zz -= total_mass * (com_x * com_x + com_y * com_y);
	i_xy += total_mass * com_x * com_y;
	i_xz += total_mass * com_x * com_z;
	i_yz += total_mass * com_y * com_z;

	out.mass = static_cast<float>(total_mass);
	out.center_of_mass = glm::vec3(static_cast<float>(com_x), static_cast<float>(com_y), static_cast<float>(com_z));
	out.inertia = glm::mat3(
	    static_cast<float>(i_xx),
	    static_cast<float>(i_xy),
	    static_cast<float>(i_xz),
	    static_cast<float>(i_xy),
	    static_cast<float>(i_yy),
	    static_cast<float>(i_yz),
	    static_cast<float>(i_xz),
	    static_cast<float>(i_yz),
	    static_cast<float>(i_zz)
	);
	return out;
}

}
