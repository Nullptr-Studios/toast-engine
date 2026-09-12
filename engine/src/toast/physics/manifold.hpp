/**
 * @file manifold.hpp
 * @author Xein
 * @date 10 Sep 2026
 *
 * @brief TODO: Brief description of the file's purpose
 */

#pragma once
#include "collision.hpp"

#include <array>
#include <cstdint>
#include <glm/glm.hpp>

namespace physics {

struct ContactPoint {
	glm::vec3 position;
	float penetration;
	ContactFeatureID feature_a;
	ContactFeatureID feature_b;
};

struct Manifold {
	BroadPhasePair pair;
	glm::vec3 normal;
	std::array<ContactPoint, 4> contacts;
	uint8_t contact_count;
};

}
