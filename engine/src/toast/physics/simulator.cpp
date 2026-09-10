#include "simulator.hpp"

#include "accumulator.hpp"
#include "nodes/rigidbody.hpp"

#include <cmath>
#include <tracy/Tracy.hpp>

namespace physics {

Simulator::Simulator() {
	TOAST_ASSERT(not instance, "Physics", "Simulator can only be created once");
	TOAST_INFO("Physics", "Simulator created");
	instance = this;

	// TODO: only singlethreaded right now
	m_manifold_queues.emplace_back();
}

Simulator::~Simulator() {
	TOAST_INFO("Physics", "Simulator destroyed");
	instance = nullptr;
}

void Simulator::registerRigidbody(Rigidbody& node) {
	TOAST_ASSERT(instance, "Physics", "Simulator instance is null; cannot register rigidbody");
	if (instance->valid(node.m_body)) {
		return;
	}

	const std::vector<SphereShape> spheres = node.sphereShapes();
	const BodyID body = instance->createBody(node.descriptor());
	node.assignBody(body);
	if (instance->valid(body)) {
		instance->m_node_bindings.push_back({.body = body, .node = node.box().as<Rigidbody>()});

		size_t registered_shape_count = 0;
		for (const SphereShape& sphere : spheres) {
			const ShapeID shape = instance->createSphere(body, sphere);
			if (not instance->valid(shape)) {
				TOAST_WARN("Physics", "Sphere collider on rigidbody '{}' was not registered", node.name());
				continue;
			}
			++registered_shape_count;
		}

		if (registered_shape_count == 0) {
			TOAST_WARN("Physics", "Rigidbody '{}' registered without an enabled sphere collider", node.name());
		} else {
			TOAST_TRACE(
			    "Physics",
			    "Registered rigidbody '{}' with {} sphere shape(s)",
			    node.name(),
			    registered_shape_count
			);
		}
	}
}

void Simulator::unregisterRigidbody(Rigidbody& node) {
	if (!instance) {
		return;
	}
	const BodyID body = node.m_body;
	if (instance->valid(body)) {
		TOAST_TRACE("Physics", "Unregistering rigidbody '{}'", node.name());
	}
	instance->destroyBody(body);
	std::erase_if(instance->m_node_bindings, [body](const NodeBinding& binding) { return binding.body == body; });
	node.assignBody({});
}

void Simulator::tick() {
	ZoneScoped;

	integrate(static_cast<float>(Accumulator::fixed_delta));

	// clear every manifold queue before starting the record step
	clearManifoldQueues();

	// record manifolds into the queue
	const auto candidates = broadPhase();
	narrowPhase(candidates);

	// now sort the manifolds
	mergeManifoldQueues();
	sortManifolds();

	// resolve
	const auto constraints = prepareConstraints(m_manifolds);
	solveConstraints(constraints);

	// push poses after simulation settles
	publishTransforms();
}

void Simulator::publishTransforms() {
	ZoneScoped;
	ZoneValue(static_cast<uint64_t>(m_node_bindings.size()));

	for (auto binding = m_node_bindings.begin(); binding != m_node_bindings.end();) {
		if (not publishTransform(*binding)) {
			binding = m_node_bindings.erase(binding);
			continue;
		}
		++binding;
	}
}

auto Simulator::publishTransform(NodeBinding& binding) -> bool {
	ZoneScoped;
	ZoneValue(static_cast<uint64_t>(binding.body.slot));

	Body* body = tryGetBody(binding.body);
	if (not binding.node.exists() || not body) {
		return false;
	}

	if (body->type == BodyType::dynamic_body) {
		binding.node->applyPhysicsTransform(body->position, body->rotation);
	}
	return true;
}

void Simulator::integrate(float dt) {
	ZoneScoped;
	ZoneValue(static_cast<uint64_t>(m_bodies.size()));

	if (not std::isfinite(dt) or dt <= 0.0f) {
		return;
	}

	for (size_t index = 0; index < m_bodies.size(); ++index) {
		BodySlot& slot = m_bodies[index];
		if (not slot.occupied) {
			continue;
		}

		integrateBody(
		    BodyID {.slot = static_cast<uint32_t>(index), .generation = slot.generation},
		    slot.body,
		    gravity,
		    dt
		);
	}
}

void Simulator::integrateBody(BodyID id, Body& body, const glm::vec3& gravity, float dt) {
	ZoneScoped;
	ZoneValue(static_cast<uint64_t>(id.slot));

	// preserve the previous pose for interpolation
	body.previous_position = body.position;
	body.previous_rotation = body.rotation;

	if (body.type != BodyType::dynamic_body) {
		return;
	}

	body.linear_velocity += gravity * body.gravity_scale * dt;
	body.position += body.linear_velocity * dt;
}

void Simulator::callTick() {
	TOAST_ASSERT(instance, "Physics", "Simulator instance is null; cannot tick");
	instance->tick();
}

auto Simulator::createBody(const BodyDescriptor& descriptor) -> BodyID {
	if (descriptor.type == BodyType::dynamic_body && (not std::isfinite(descriptor.mass) or descriptor.mass <= 0.0f)) {
		TOAST_WARN("Physics", "Rejected dynamic body with non-finite or non-positive mass");
		return {};
	}

	if (not std::isfinite(descriptor.gravity_scale)) {
		TOAST_WARN("Physics", "Rejected body with non-finite gravity scale");
		return {};
	}

	const float rotation_length_squared = glm::dot(descriptor.rotation, descriptor.rotation);
	if (not std::isfinite(rotation_length_squared) or rotation_length_squared <= 0.0f) {
		TOAST_WARN("Physics", "Rejected body with a non-finite or zero-length orientation");
		return {};
	}

	const glm::quat rotation = glm::normalize(descriptor.rotation);
	const float inverse_mass = descriptor.type == BodyType::dynamic_body ? 1.0f / descriptor.mass : 0.0f;

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

	for (uint32_t index = 0; index < m_shapes.size(); ++index) {
		ShapeSlot& shape_slot = m_shapes[index];
		if (shape_slot.occupied && shape_slot.shape.owner == body) {
			destroyShape({.slot = index, .generation = shape_slot.generation});
		}
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

auto Simulator::createSphere(BodyID owner, const SphereShape& sphere) -> ShapeID {
	if (not valid(owner)) {
		TOAST_WARN("Physics", "Rejected sphere registration for an invalid body");
		return {};
	}

	if (not std::isfinite(sphere.radius) or sphere.radius <= 0.0f || not std::isfinite(sphere.local_center.x) or
	    not std::isfinite(sphere.local_center.y) or not std::isfinite(sphere.local_center.z)) {
		TOAST_WARN("Physics", "Rejected sphere with invalid radius or local center");
		return {};
	}

	Shape shape {
	  .owner = owner,
	  .type = ShapeType::sphere,
	  .sphere = sphere,
	};

	if (not m_free_shape_slots.empty()) {
		const uint32_t index = m_free_shape_slots.back();
		m_free_shape_slots.pop_back();

		ShapeSlot& slot = m_shapes[index];
		slot.shape = shape;
		slot.occupied = true;
		return {.slot = index, .generation = slot.generation};
	}

	const uint32_t index = static_cast<uint32_t>(m_shapes.size());
	m_shapes.emplace_back(ShapeSlot {.shape = shape, .generation = 1, .occupied = true});
	return {.slot = index, .generation = 1};
}

void Simulator::destroyShape(ShapeID shape) {
	if (not valid(shape)) {
		return;
	}

	ShapeSlot& slot = m_shapes[shape.slot];
	slot.occupied = false;
	++slot.generation;
	if (slot.generation == 0) {
		++slot.generation;
	}
	m_free_shape_slots.emplace_back(shape.slot);
}

auto Simulator::valid(ShapeID shape) const -> bool {
	return shape.slot < m_shapes.size() && m_shapes[shape.slot].occupied && m_shapes[shape.slot].generation == shape.generation;
}

auto Simulator::tryGetShape(ShapeID shape) -> Shape* {
	return valid(shape) ? &m_shapes[shape.slot].shape : nullptr;
}

auto Simulator::tryGetShape(ShapeID shape) const -> const Shape* {
	return valid(shape) ? &m_shapes[shape.slot].shape : nullptr;
}

auto Simulator::valid(BodyID body) const -> bool {
	return body.slot < m_bodies.size() && m_bodies[body.slot].occupied && m_bodies[body.slot].generation == body.generation;
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

auto Simulator::broadPhase() const -> std::vector<BroadPhasePair> {
	ZoneScoped;

	std::vector<BroadPhasePair> pairs;

	for (size_t i = 0; i < m_shapes.size(); ++i) {
		if (not m_shapes[i].occupied) {
			continue;
		}

		for (size_t j = i + 1; j < m_shapes.size(); ++j) {
			if (auto pair = broadPhasePair(i, j)) {
				pairs.emplace_back(*pair);
			}
		}
	}

	ZoneValue(static_cast<uint64_t>(pairs.size()));
	return pairs;
}

auto Simulator::broadPhasePair(size_t shape_a_index, size_t shape_b_index) const -> std::optional<BroadPhasePair> {
	ZoneScoped;
	ZoneValue((static_cast<uint64_t>(shape_a_index) << 32) | static_cast<uint64_t>(shape_b_index));

	const ShapeSlot& shape_a = m_shapes[shape_a_index];
	const ShapeSlot& shape_b = m_shapes[shape_b_index];
	if (not shape_b.occupied) {
		return std::nullopt;
	}

	// skip if they have the same owner
	if (shape_b.shape.owner == shape_a.shape.owner) {
		return std::nullopt;
	}

	const Body* body_a = tryGetBody(shape_a.shape.owner);
	const Body* body_b = tryGetBody(shape_b.shape.owner);

	// skip if body is invalid
	if (not body_a || not body_b) {
		return std::nullopt;
	}

	// skip if they are both static/kinematic
	if (body_a->inverse_mass == 0.0f && body_b->inverse_mass == 0.0f) {
		return std::nullopt;
	}

	// clang-format off
	return canonicalPair(
		BodyShapeKey {
			.body = shape_a.shape.owner,
			.shape = {
				.slot = static_cast<uint32_t>(shape_a_index),
				.generation = shape_a.generation
			}
		},
		BodyShapeKey {
			.body = shape_b.shape.owner,
			.shape = {
				.slot = static_cast<uint32_t>(shape_b_index),
				.generation = shape_b.generation
			}
		}
	);
	// clang-format on
}

void Simulator::clearManifoldQueues() {
	ZoneScoped;
	ZoneValue(static_cast<uint64_t>(m_manifold_queues.size()));

	std::ranges::for_each(m_manifold_queues, [](ManifoldQueue& queue) { queue.clear(); });
}

void Simulator::narrowPhase(const std::vector<BroadPhasePair>& candidates) {
	ZoneScoped;
	ZoneValue(static_cast<uint64_t>(candidates.size()));

	std::ranges::for_each(candidates, [&](auto pair) { collide(pair); });
}

void Simulator::mergeManifoldQueues() {
	ZoneScoped;

	m_manifolds.clear();

	for (const ManifoldQueue& queue : m_manifold_queues) {
		for (const Manifold& manifold : queue) {
			mergeManifold(manifold);
		}
	}

	ZoneValue(static_cast<uint64_t>(m_manifolds.size()));
}

void Simulator::mergeManifold(const Manifold& manifold) {
	ZoneScoped;
	ZoneValue(
	    (static_cast<uint64_t>(manifold.pair.a.shape.slot) << 32) |
	    static_cast<uint64_t>(manifold.pair.b.shape.slot)
	);

	m_manifolds.emplace_back(manifold);
}

void Simulator::sortManifolds() {
	ZoneScoped;
	ZoneValue(static_cast<uint64_t>(m_manifolds.size()));

	std::ranges::sort(m_manifolds, [](const Manifold& lhs, const Manifold& rhs) {
		return lhs.pair < rhs.pair;
	});
}

auto Simulator::prepareConstraints(const std::vector<Manifold>& manifolds) const -> std::vector<NormalConstraint> {
	ZoneScoped;
	std::vector<NormalConstraint> constraints;
	constraints.reserve(manifolds.size());
	size_t rejected_manifold_count = 0;

	for (const Manifold& manifold : manifolds) {
		if (auto constraint = prepareConstraint(manifold)) {
			constraints.emplace_back(*constraint);
		} else {
			++rejected_manifold_count;
		}
	}

	if (rejected_manifold_count > 0) {
		TOAST_WARN("Physics", "Rejected {} invalid manifold(s) while preparing constraints", rejected_manifold_count);
	}

	ZoneValue(static_cast<uint64_t>(constraints.size()));
	return constraints;
}

auto Simulator::prepareConstraint(const Manifold& manifold) const -> std::optional<NormalConstraint> {
	ZoneScoped;
	ZoneValue(
	    (static_cast<uint64_t>(manifold.pair.a.body.slot) << 32) |
	    static_cast<uint64_t>(manifold.pair.b.body.slot)
	);

	const Body* body_a = tryGetBody(manifold.pair.a.body);
	const Body* body_b = tryGetBody(manifold.pair.b.body);
	if (not body_a || not body_b || manifold.contact_count == 0) {
		return std::nullopt;
	}

	const ContactPoint& contact = manifold.contacts[0];
	const bool normal_is_finite = std::isfinite(manifold.normal.x) && std::isfinite(manifold.normal.y) &&
	                              std::isfinite(manifold.normal.z);
	if (not normal_is_finite || not std::isfinite(contact.penetration) || contact.penetration < 0.0f) {
		return std::nullopt;
	}

	const float inverse_mass_sum = body_a->inverse_mass + body_b->inverse_mass;
	if (not std::isfinite(inverse_mass_sum) || inverse_mass_sum <= 0.0f) {
		return std::nullopt;
	}

	return NormalConstraint {
	  .body_a = manifold.pair.a.body,
	  .body_b = manifold.pair.b.body,
	  .normal = manifold.normal,
	  .inverse_mass_a = body_a->inverse_mass,
	  .inverse_mass_b = body_b->inverse_mass,
	  .effective_mass = 1.0f / inverse_mass_sum,
	};
}

void Simulator::solveConstraints(const std::vector<NormalConstraint>& constraints) {
	ZoneScoped;
	ZoneValue(static_cast<uint64_t>(constraints.size()));

	size_t invalid_constraint_count = 0;
	for (const NormalConstraint& constraint : constraints) {
		if (not solveConstraint(constraint)) {
			++invalid_constraint_count;
		}
	}

	if (invalid_constraint_count > 0) {
		TOAST_WARN("Physics", "Skipped {} invalid solver constraint(s)", invalid_constraint_count);
	}
}

auto Simulator::solveConstraint(const NormalConstraint& constraint) -> bool {
	ZoneScoped;
	ZoneValue((static_cast<uint64_t>(constraint.body_a.slot) << 32) | static_cast<uint64_t>(constraint.body_b.slot));

	Body* body_a = tryGetBody(constraint.body_a);
	Body* body_b = tryGetBody(constraint.body_b);
	if (not body_a || not body_b) {
		return false;
	}

	const glm::vec3 relative_velocity = body_b->linear_velocity - body_a->linear_velocity;
	const float relative_normal_speed = glm::dot(relative_velocity, constraint.normal);
	if (not std::isfinite(relative_normal_speed)) {
		return false;
	}
	if (relative_normal_speed >= 0.0f) {
		return true;
	}

	const float impulse_magnitude = -relative_normal_speed * constraint.effective_mass;
	const glm::vec3 impulse = constraint.normal * impulse_magnitude;
	body_a->linear_velocity -= impulse * constraint.inverse_mass_a;
	body_b->linear_velocity += impulse * constraint.inverse_mass_b;
	return true;
}

void Simulator::collide(BroadPhasePair pair) {
	ZoneScoped;
	ZoneValue((static_cast<uint64_t>(pair.a.shape.slot) << 32) | static_cast<uint64_t>(pair.b.shape.slot));

	const Shape* shape_a = tryGetShape(pair.a.shape);
	const Shape* shape_b = tryGetShape(pair.b.shape);
	const Body* body_a = tryGetBody(pair.a.body);
	const Body* body_b = tryGetBody(pair.b.body);

#ifdef DEBUG
	// this is checked on broad phase so we don't need to check it back here
	TOAST_ASSERT(shape_a && shape_b && body_a && body_b, "Physics", "Broad-phase pair became invalid");
	TOAST_ASSERT(
			shape_a->owner == pair.a.body && shape_b->owner == pair.b.body,
			"Physics",
			"Broad-phase shape ownership mismatch"
	);
#endif

	// manifolds will be generated by the collideX functions
	// just call the correct one here
	std::optional<Manifold> manifold;
	switch (shape_a->type) {
		case ShapeType::sphere:
			switch (shape_b->type) {
				case ShapeType::sphere:
					manifold = collideSpheres(pair, *shape_a, *body_a, *shape_b, *body_b);
			}
	}

	if (manifold.has_value()) {
		m_manifold_queues[0].emplace_back(*manifold);
	}
}

auto Simulator::collideSpheres(
	BroadPhasePair pair,
	const Shape& shape_a,
	const Body& body_a,
	const Shape& shape_b,
	const Body& body_b
) -> std::optional<Manifold> {
	ZoneScoped;
	ZoneValue((static_cast<uint64_t>(pair.a.shape.slot) << 32) | static_cast<uint64_t>(pair.b.shape.slot));

	const glm::vec3 center_a = body_a.position + body_a.rotation * shape_a.sphere.local_center;
	const glm::vec3 center_b = body_b.position + body_b.rotation * shape_b.sphere.local_center;
	const glm::vec3 delta = center_b - center_a;
	const float radius_sum = shape_a.sphere.radius + shape_b.sphere.radius;
	const float distance_squared = glm::dot(delta, delta);

	if (not std::isfinite(distance_squared) || distance_squared > radius_sum * radius_sum) {
		return std::nullopt;
	}

	const float distance = std::sqrt(distance_squared);
	const glm::vec3 normal = distance_squared > 1.0e-9f ? delta / distance : glm::vec3 {1.0f, 0.0f, 0.0f};
	const glm::vec3 point_a = center_a + normal * shape_a.sphere.radius;
	const glm::vec3 point_b = center_b - normal * shape_b.sphere.radius;

	return Manifold {
	  .pair = pair,
	  .normal = normal,
	  .contacts = {
	    ContactPoint {
	      .position = (point_a + point_b) * 0.5f,
	      .penetration = radius_sum - distance,
	      .feature_a = sphere_surface_feature,
	      .feature_b = sphere_surface_feature,
	    }
	  },
	  .contact_count = 1
	};
}

}
