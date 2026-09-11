/**
 * @file DynamicRigidbody.hpp
 * @author Xein
 * @date 09 Sep 2026
 *
 * @brief TODO: Brief description of the file's purpose
 */

#pragma once
#include "rigidbody.hpp"

namespace physics {

class [[ToastNode, Icon("RigidBody")]] TOAST_API DynamicRigidbody : public physics::Rigidbody {
public:
	DynamicRigidbody() : Rigidbody(BodyType::dynamic_body) { }

	[[Reflect, Unit("kg"), ReadOnly]]
	float mass = 1.0f;
	// physics material
	[[Reflect, ReadOnly]]
	float gravity_scale = 1.0f;
	
	[[Reflect, Group("Mass Distribution"), Unit("m"), ReadOnly]]
	glm::vec3 center_of_mass = {};
	[[Reflect, Group("Mass Distribution"), Unit("kg•m²"), ReadOnly]]
	glm::vec3 inertia = {};
	
	[[Reflect, Group("Velocities"), Unit("m/s"), ReadOnly]]
	glm::vec3 linear_velocity = {};
	[[Reflect, Group("Velocities"), Unit("m/s"), ReadOnly]]
	glm::vec3 angular_velocity = {};
	
	[[Reflect, Group("Constant Forces"), Unit("N"), ReadOnly]]
	glm::vec3 constant_force = {};
	[[Reflect, Group("Constant Forces"), Unit("N•m"), ReadOnly]]
	glm::vec3 constant_torque = {};
private:
};

}
