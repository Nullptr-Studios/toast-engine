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

private:
};

}
