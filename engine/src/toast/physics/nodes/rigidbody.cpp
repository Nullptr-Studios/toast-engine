#include "rigidbody.hpp"

#include <toast/physics/simulator.hpp>

namespace physics {

void Rigidbody::begin() {
	if (not m_registration_requested && participatesIn(toast::NodeOwnerParticipation::gameplay_tick)) {
		m_registration_requested = true;
		Simulator::registerRigidbody(*this);
	}
}

void Rigidbody::end() {
	if (m_registration_requested) {
		m_registration_requested = false;
		Simulator::unregisterRigidbody(*this);
	}
}

auto Rigidbody::descriptor() const -> BodyDescriptor {
	syncTransform();
	BodyDescriptor result;
	result.type = m_body_type;
	result.position = world_position;
	result.rotation = world_rotation;
	return result;
}

void Rigidbody::applyPhysicsTransform(const glm::vec3& position, const glm::quat& rotation) {
	world_position = position;
	world_rotation = rotation;
	syncTransform();
}

}
