/**
 * @file Collider.hpp
 * @author Xein
 * @date 09 Sep 2026
 *
 * @brief TODO: Brief description of the file's purpose
 */

#pragma once
#include <toast/world/node_3d.hpp>

namespace physics {

class [[ToastNode, Hidden, Interface, Icon("Container"), Color("Green")]] TOAST_API Collider : public toast::Node3D {
public:
	[[Reflect]]
	bool disabled = false;

	[[Reflect, Color]]
	glm::vec4 debug_color = glm::vec4(0.0f, 1.0f, 0.251f, 0.5f);    // Matches editor green

	[[Reflect]]
	bool debug_fill = true;

private:
};

}
