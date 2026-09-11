#include "rigidbody.hpp"

#include "collider.hpp"
#include "sphere_collider.hpp"

#include <algorithm>
#include <cmath>
#include <toast/physics/simulator.hpp>

namespace physics {

void Rigidbody::updateInspectorMessages() {
	static const toast::NodeMessage message {
		.severity = toast::NodeMessage::warning,
		.id = 2,
		.text = "Rigidbodies require a collider",
	};

	const bool has_collider = std::ranges::any_of(children(), [](const auto& child) {
		const auto sphere = child.template as<SphereCollider>();
		return sphere.exists() && !sphere->disabled && std::isfinite(sphere->radius) && sphere->radius > 0.0f;
	});
	if (has_collider) {
		removeInspectorMessage(message);
	} else {
		addInspectorMessage(message);
	}
}

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

auto Rigidbody::sphereShapes() const -> std::vector<SphereShape> {
	std::vector<SphereShape> shapes;

	for (const auto& child : children()) {
		const auto sphere = child.as<SphereCollider>();
		if (not sphere.exists() || sphere->disabled) {
			continue;
		}

		shapes.emplace_back(SphereShape {
			.local_center = sphere->position,
			.radius = sphere->radius,
		});
	}

	return shapes;
}

void Rigidbody::applyPhysicsTransform(const glm::vec3& position, const glm::quat& rotation) {
	world_position = position;
	world_rotation = rotation;
	syncTransform();
}

}
