/**
 * @file simulator.hpp
 * @author Xein
 * @date 09 Sep 2026
 * @brief This class simulates all of the physics of the project
 */

#pragma once

#include "body.hpp"

#include <deque>
#include <optional>
#include <vector>

#include <toast/export.hpp>
#include <toast/log.hpp>
#include <toast/world/box.hpp>

namespace physics {

class Rigidbody;

class TOAST_API Simulator {
public:
	Simulator();
	~Simulator();

	void tick();
	void step(float dt);

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
	struct Body {
		BodyType type = BodyType::dynamic_body;
		glm::vec3 position = {};
		glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
		glm::vec3 previous_position = {};
		glm::quat previous_rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
		glm::vec3 linear_velocity = {};
		glm::vec3 angular_velocity = {};
		float inverse_mass = 1.0f;
		float gravity_scale = 1.0f;
	};

	struct BodySlot {
		Body body;
		uint32_t generation = 1;
		bool occupied = false;
	};

	struct BodyBinding {
		BodyID body;
		toast::Box<Rigidbody> node;
	};

	[[nodiscard]]
	auto tryGetBody(BodyID body) -> Body*;
	[[nodiscard]]
	auto tryGetBody(BodyID body) const -> const Body*;

	inline static Simulator* instance = nullptr;
	std::vector<BodySlot> m_bodies;
	std::deque<uint32_t> m_free_body_slots;
	std::vector<BodyBinding> m_body_bindings;
	glm::vec3 gravity = {0.0f, 0.0f, -9.8f};
};

}
