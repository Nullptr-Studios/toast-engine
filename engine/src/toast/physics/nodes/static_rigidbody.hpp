/**
 * @file StaticRigidbody.hpp
 * @author Xein
 * @date 09 Sep 2026
 *
 * @brief TODO: Brief description of the file's purpose
 */

#pragma once
#include "rigidbody.hpp"

namespace physics {

class [[ToastNode]] TOAST_API StaticRigidbody : public physics::Rigidbody {
public:
	StaticRigidbody() : Rigidbody(BodyType::static_body) { }
private:
};

}
