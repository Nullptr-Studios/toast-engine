/**
 * @file shape.hpp
 * @author Xein
 * @date 10 Sep 2026
 * @brief Shape identifiers, geometry, and physics-owned runtime records
 */

#pragma once

#include "body.hpp"

#include <compare>
#include <cstdint>
#include <glm/glm.hpp>
#include <limits>

namespace physics {

struct ShapeID {
	uint32_t slot = std::numeric_limits<uint32_t>::max();
	uint32_t generation = 0;
	auto operator<=>(const ShapeID&) const = default;
};

struct ContactFeatureID {
	uint64_t value = 0;
	auto operator<=>(const ContactFeatureID&) const = default;
};

enum class ShapeType : uint8_t {
	sphere
};

struct SphereShape {
	glm::vec3 local_center = {};
	float radius = 0.5f;
};

struct BoxShape {
	glm::vec3 local_center = {};
	glm::vec3 size = {1.0f, 1.0f, 1.0f};
};

struct CapsuleShape {
	glm::vec3 local_center = {};
	float radius = 0.5f;
	float height = 1.0f;
};

struct Shape {
	BodyID owner;
	ShapeType type = ShapeType::sphere;

	union {
		SphereShape sphere;
		BoxShape box;
		CapsuleShape capsule;
	};
};

struct ShapeSlot {
	Shape shape;
	uint32_t generation = 1;
	bool occupied = false;
};

inline constexpr ContactFeatureID sphere_surface_feature {0};

}
