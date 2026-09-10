/**
 * @file simulator.hpp
 * @author Xein
 * @date 09 Sep 2026
 * @brief This class simulates all of the physics of the project
 */

#pragma once

#include "body.hpp"
#include "collision.hpp"
#include "shape.hpp"

#include <deque>
#include <optional>
#include <toast/export.hpp>
#include <toast/log.hpp>
#include <toast/world/box.hpp>
#include <vector>

namespace physics {

class Rigidbody;

class TOAST_API Simulator {
public:
	Simulator();
	~Simulator();

	void tick();
	void integrate(float dt);

	[[nodiscard]]
	auto createBody(const BodyDescriptor& descriptor) -> BodyID;
	void destroyBody(BodyID body);

	[[nodiscard]]
	auto valid(BodyID body) const -> bool;
	[[nodiscard]]
	auto state(BodyID body) const -> std::optional<BodyState>;

	auto setTransform(BodyID body, const glm::vec3& position, const glm::quat& rotation) -> bool;
	auto setLinearVelocity(BodyID body, const glm::vec3& velocity) -> bool;

	static void callTick();
	static void registerRigidbody(Rigidbody& node);
	static void unregisterRigidbody(Rigidbody& node);

private:
	struct NodeBinding {
		BodyID body;
		toast::Box<Rigidbody> node;
	};

	[[nodiscard]]
	auto createSphere(BodyID owner, const SphereShape& sphere) -> ShapeID;
	void destroyShape(ShapeID shape);
	[[nodiscard]]
	auto valid(ShapeID shape) const -> bool;
	[[nodiscard]]
	auto tryGetShape(ShapeID shape) -> Shape*;
	[[nodiscard]]
	auto tryGetShape(ShapeID shape) const -> const Shape*;

	[[nodiscard]]
	auto tryGetBody(BodyID body) -> Body*;
	[[nodiscard]]
	auto tryGetBody(BodyID body) const -> const Body*;

	[[nodiscard]]
	auto broadPhase() const -> std::vector<BroadPhasePair> {
		std::vector<BroadPhasePair> pairs;

		for (size_t i = 0; i < m_shapes.size(); ++i) {
			if (not m_shapes[i].occupied) {
				continue;
			}

			for (size_t j = i + 1; j < m_shapes.size(); ++j) {
				if (not m_shapes[j].occupied) {
					continue;
				}

				// skip if they have the same owner
				if (m_shapes[j].shape.owner == m_shapes[i].shape.owner) {
					continue;
				}

				const auto* body1 = tryGetBody(m_shapes[i].shape.owner);
				const auto* body2 = tryGetBody(m_shapes[j].shape.owner);

				// skip if body is invalid
				if (not body1 || not body2) {
					continue;
				}

				// skip if they are both static/kinematic
				if (body1->inverse_mass == 0 && body2->inverse_mass == 0) {
					continue;
				}

				// clang-format off
				pairs.emplace_back(canonicalPair(
					BodyShapeKey {
						.body = m_shapes[i].shape.owner,
						.shape = {
							.slot = static_cast<uint32_t>(i),
							.generation = m_shapes[i].generation
						}
					},
					BodyShapeKey {
						.body = m_shapes[j].shape.owner,
						.shape = {
							.slot = static_cast<uint32_t>(j),
							.generation = m_shapes[j].generation
						}
					}
				));
				// clang-format on
			}
		}

		return pairs;
	}

	inline static Simulator* instance = nullptr;

	std::vector<BodySlot> m_bodies;
	std::deque<uint32_t> m_free_body_slots;
	std::vector<NodeBinding> m_node_bindings;

	std::vector<ShapeSlot> m_shapes;
	std::deque<uint32_t> m_free_shape_slots;

	glm::vec3 gravity = {0.0f, 0.0f, -9.8f};
};

}
