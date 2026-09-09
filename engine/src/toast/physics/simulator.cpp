#include "simulator.hpp"

#include "accumulator.hpp"
#include "nodes/rigidbody.hpp"

#include <cmath>

namespace physics {

Simulator::Simulator() {
	TOAST_ASSERT(not instance, "Physics", "Simulator can only be created once");
	TOAST_INFO("Physics", "Simulator created");
	instance = this;
}

Simulator::~Simulator() {
	instance = nullptr;
}

void Simulator::registerRigidbody(Rigidbody& node) {
	TOAST_ASSERT(instance, "Physics", "Simulator instance is null; cannot register rigidbody");
	if (instance->valid(node.m_body)) {
		return;
	}
	const BodyID body = instance->createBody(node.descriptor());
	node.assignBody(body);
	if (instance->valid(body)) {
		instance->m_body_bindings.push_back({.body = body, .node = node.box().as<Rigidbody>()});
	}
}

void Simulator::unregisterRigidbody(Rigidbody& node) {
	if (!instance) {
		return;
	}
	const BodyID body = node.m_body;
	instance->destroyBody(body);
	std::erase_if(instance->m_body_bindings, [body](const BodyBinding& binding) { return binding.body == body; });
	node.assignBody({});
}

void Simulator::tick() {
	step(static_cast<float>(Accumulator::fixed_delta));

	// push poses after simulation settles
	for (auto binding = m_body_bindings.begin(); binding != m_body_bindings.end();) {
		TOAST_INFO("Physics", "Simulator::tick");
		Body* body = tryGetBody(binding->body);
		if (not binding->node.exists() or not body) {
			binding = m_body_bindings.erase(binding);
			continue;
		}

		if (body->type == BodyType::dynamic_body) {
			binding->node->applyPhysicsTransform(body->position, body->rotation);
		}
		++binding;
	}
}

void Simulator::step(float dt) {
	if (not std::isfinite(dt) or dt <= 0.0f) {
		return;
	}

	for (auto& slot : m_bodies) {
		if (not slot.occupied) {
			continue;
		}

		Body& body = slot.body;
		// preserve the previous pose for interpolation
		body.previous_position = body.position;
		body.previous_rotation = body.rotation;

		if (body.type != BodyType::dynamic_body) {
			continue;
		}

		body.linear_velocity += gravity * body.gravity_scale * dt;
		body.position += body.linear_velocity * dt;
	}
}

void Simulator::callTick() {
	TOAST_ASSERT(instance, "Physics", "Simulator instance is null; cannot tick");
	instance->tick();
}

auto Simulator::createBody(const BodyDescriptor& descriptor) -> BodyID {
	if (descriptor.type == BodyType::dynamic_body &&
	    (not std::isfinite(descriptor.mass) or descriptor.mass <= 0.0f)) {
		TOAST_ERROR("Physics", "Dynamic body mass must be finite and greater than zero");
		return {};
	}

	if (not std::isfinite(descriptor.gravity_scale)) {
		TOAST_ERROR("Physics", "Body gravity scale must be finite");
		return {};
	}

	const float rotation_length_squared = glm::dot(descriptor.rotation, descriptor.rotation);
	if (not std::isfinite(rotation_length_squared) or rotation_length_squared <= 0.0f) {
		TOAST_ERROR("Physics", "Body orientation must be a finite, non-zero quaternion");
		return {};
	}

	const glm::quat rotation = glm::normalize(descriptor.rotation);
	const float inverse_mass =
	    descriptor.type == BodyType::dynamic_body ? 1.0f / descriptor.mass : 0.0f;

	Body body {
	  .type = descriptor.type,
	  .position = descriptor.position,
	  .rotation = rotation,
	  .previous_position = descriptor.position,
	  .previous_rotation = rotation,
	  .linear_velocity = descriptor.linear_velocity,
	  .angular_velocity = descriptor.angular_velocity,
	  .inverse_mass = inverse_mass,
	  .gravity_scale = descriptor.gravity_scale
	};

	if (not m_free_body_slots.empty()) {
		// try reusing dead slots to avoid allocating
		const uint32_t index = m_free_body_slots.back();
		m_free_body_slots.pop_back();

		BodySlot& slot = m_bodies[index];
		slot.body = body;
		slot.occupied = true;
		return {.slot = index, .generation = slot.generation};
	}

	// best effort :(
	// just push back and regrow if needed
	const uint32_t index = static_cast<uint32_t>(m_bodies.size());
	m_bodies.emplace_back(BodySlot {.body = body, .generation = 1, .occupied = true});
	return {.slot = index, .generation = 1};
}

void Simulator::destroyBody(BodyID body) {
	if (not valid(body)) {
		return;
	}

	BodySlot& slot = m_bodies[body.slot];
	slot.occupied = false;
	// invalidate every stale handle
	++slot.generation;
	if (slot.generation == 0) {
		++slot.generation;
	}
	m_free_body_slots.emplace_back(body.slot);
}

auto Simulator::valid(BodyID body) const -> bool {
	return body.slot < m_bodies.size() && m_bodies[body.slot].occupied &&
	       m_bodies[body.slot].generation == body.generation;
}

auto Simulator::state(BodyID body) const -> std::optional<BodyState> {
	const Body* value = tryGetBody(body);
	if (not value) {
		return std::nullopt;
	}

	return BodyState {
	  .type = value->type,
	  .position = value->position,
	  .rotation = value->rotation,
	  .previous_position = value->previous_position,
	  .previous_rotation = value->previous_rotation,
	  .linear_velocity = value->linear_velocity,
	  .angular_velocity = value->angular_velocity
	};
}

auto Simulator::setTransform(BodyID body, const glm::vec3& position, const glm::quat& rotation) -> bool {
	Body* value = tryGetBody(body);
	const float rotation_length_squared = glm::dot(rotation, rotation);
	if (not value or not std::isfinite(rotation_length_squared) or rotation_length_squared <= 0.0f) {
		return false;
	}

	const glm::quat normalized_rotation = glm::normalize(rotation);
	value->position = position;
	value->rotation = normalized_rotation;
	value->previous_position = position;
	value->previous_rotation = normalized_rotation;
	return true;
}

auto Simulator::setLinearVelocity(BodyID body, const glm::vec3& velocity) -> bool {
	Body* value = tryGetBody(body);
	if (not value) {
		return false;
	}

	value->linear_velocity = velocity;
	return true;
}

auto Simulator::tryGetBody(BodyID body) -> Body* {
	return valid(body) ? &m_bodies[body.slot].body : nullptr;
}

auto Simulator::tryGetBody(BodyID body) const -> const Body* {
	return valid(body) ? &m_bodies[body.slot].body : nullptr;
}

}
