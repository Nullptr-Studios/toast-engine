/**
 * @file Rigidbody.hpp
 * @author Xein
 * @date 09 Sep 2026
 *
 * @brief TODO: Brief description of the file's purpose
 */

#pragma once
#include <toast/physics/body.hpp>
#include <toast/world/node_3d.hpp>

namespace physics {

class Simulator;

class [[ToastNode, Hidden, Interface, Icon("PhysicsBody"), Color("Green")]] TOAST_API Rigidbody : public toast::Node3D {
	friend class Simulator;

public:

protected:
	explicit Rigidbody(BodyType type) : m_body_type(type) { }

private:
	void begin();
	void end();

	[[nodiscard]]
	auto descriptor() const -> BodyDescriptor;

	void assignBody(BodyID body) noexcept { m_body = body; }

	void applyPhysicsTransform(const glm::vec3& position, const glm::quat& rotation);

	BodyType m_body_type;
	BodyID m_body;
	bool m_registration_requested = false;
};

}
