/**
 * @file constraint.hpp
 * @author Xein
 * @date 11 Sep 2026
 * @brief TODO: Brief description of the file's purpose
 */

#pragma once

#include "body.hpp"

#include <glm/vec3.hpp>

namespace physics {

struct NormalConstraint {
	BodyID body_a;
	BodyID body_b;
	glm::vec3 normal;
	float inverse_mass_a;
	float inverse_mass_b;
	float effective_mass;
};

}
