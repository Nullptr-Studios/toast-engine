/**
 * @file Rigidbody.hpp
 * @author Xein
 * @date 09 Sep 2026
 *
 * @brief TODO: Brief description of the file's purpose
 */

#pragma once
#include <toast/physics/body.hpp>
#include <toast/physics/shape.hpp>
#include <toast/world/node_3d.hpp>

#include <vector>

namespace physics {

class Simulator;

class [[ToastNode, Hidden, Interface, Icon("PhysicsBody"), Color("Green")]] TOAST_API Rigidbody : public toast::Node3D {
	friend class Simulator;

protected:
	explicit Rigidbody(BodyType type) : m_body_type(type) { }

	[[Reflect, Name("Lock x position"), ReadOnly]]
	bool lock_pos_x = false;
	[[Reflect, Name("Lock y position"), ReadOnly]]
	bool lock_pos_y = false;
	[[Reflect, Name("Lock z position"), ReadOnly]]
	bool lock_pos_z = false;
	[[Reflect, Name("Lock x rotation"), ReadOnly]]
	bool lock_rot_x = false;
	[[Reflect, Name("Lock y rotation"), ReadOnly]]
	bool lock_rot_y = false;
	[[Reflect, Name("Lock z rotation"), ReadOnly]]
	bool lock_rot_z = false;

private:
	void updateInspectorMessages() override;
	void begin();
	void end();

	[[nodiscard]]
	auto descriptor() const -> BodyDescriptor;
	[[nodiscard]]
	auto sphereShapes() const -> std::vector<SphereShape>;

	void assignBody(BodyID body) noexcept { m_body = body; }

	void applyPhysicsTransform(const glm::vec3& position, const glm::quat& rotation);

	BodyType m_body_type;
	BodyID m_body;
	bool m_registration_requested = false;
};

}
