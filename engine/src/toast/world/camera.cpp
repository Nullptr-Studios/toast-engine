/// @file camera.cpp
/// @author dario
/// @date 7/4/2026

#include "camera.hpp"

namespace toast {
void Camera::setActiveCamera() {
	if (m_owner) {
		m_owner->activateCamera(*this);
	}
}

void Camera::begin() {
	setActiveCamera();
}

void Camera::end() {
	if (m_owner) {
		m_owner->deactivateCamera(*this);
	}
}

void Camera::onEnable() {
	setActiveCamera();
}

void Camera::onDisable() {
	if (m_owner) {
		m_owner->deactivateCamera(*this);
	}
}

auto Camera::getView() const -> glm::mat4 {
	syncTransform();
	return glm::lookAt(world_position, world_position + forward(), up());
}

auto Camera::getProjection(float aspect) const -> glm::mat4 {
	// _ZO, not plain glm::perspective. GLM defaults to OpenGL's [-1,1] depth range, and Vulkan's clip volume
	// is [0,1] - so the default maps the near plane to -1 and lets the hardware clip everything in front of
	// ndc z = 0, which for a 1cm near plane is roughly the first 2cm of the view. It also throws away half the
	// depth buffer's precision, since only the [0,1] half of the emitted range is ever stored
	//
	// Every other projection in the engine already says _ZO explicitly - the shadow cascades, the punctual
	// shadow faces and the probe captures. This was the one that did not, which made the camera the odd one
	// out rather than the rule. screenPointToRay() below already assumed this convention: it unprojects
	// z = 0 and calls the result the near point, which is only true here
	glm::mat4 proj = glm::perspectiveRH_ZO(glm::radians(fov), aspect, near_plane, far_plane);

	// Vulkan's framebuffer y runs down the screen where GL's runs up, so the projection is flipped once here
	// rather than every shader having to remember. Note this is the *only* place it happens - a cube face
	// capture deliberately does not flip, because its texel rows run along the face's own up vector
	proj[1][1] *= -1.0f;

	return proj;
}

auto Camera::screenPointToRay(glm::vec2 screen_px, glm::vec2 viewport_size) const noexcept -> Ray {
	syncTransform();

	if (viewport_size.x <= 0.0f || viewport_size.y <= 0.0f) {
		return {world_position, forward()};
	}

	const float aspect = viewport_size.x / viewport_size.y;

	// screen (y-down) -> NDC ([-1,1])
	const glm::vec2 ndc {
	  ((2.0f * screen_px.x) / viewport_size.x) - 1.0f,
	  ((2.0f * screen_px.y) / viewport_size.y) - 1.0f,
	};

	const glm::mat4 inv_view_proj = glm::inverse(getProjection(aspect) * getView());

	glm::vec4 near_point = inv_view_proj * glm::vec4(ndc.x, ndc.y, 0.0f, 1.0f);
	glm::vec4 far_point = inv_view_proj * glm::vec4(ndc.x, ndc.y, 1.0f, 1.0f);
	near_point /= near_point.w;
	far_point /= far_point.w;

	return {world_position, glm::normalize(glm::vec3(far_point) - glm::vec3(near_point))};
}
}
