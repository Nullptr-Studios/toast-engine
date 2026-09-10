/**
 * @file SphereCollider.hpp
 * @author Xein
 * @date 09 Sep 2026
 *
 * @brief TODO: Brief description of the file's purpose
 */

#pragma once
#include "collider.hpp"

namespace physics {

class [[ToastNode, Icon("SphereMesh")]] TOAST_API SphereCollider : public physics::Collider {
public:
	[[Reflect, Unit("m")]]
	float radius = 0.5f;

	[[Reflect, Unit("m")]]
	float margin = 0.05f;
};

}
