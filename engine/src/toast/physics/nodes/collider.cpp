#include "collider.hpp"

#include "box_collider.hpp"
#include "capsule_collider.hpp"
#include "sphere_collider.hpp"
#include "rigidbody.hpp"
#include <toast/renderer/vulkan_renderer.hpp>

namespace physics {
void Collider::init() {
	m_debug_visible = enabled();
	if (renderer::VulkanRenderer::instance) {
		renderer::VulkanRenderer::instance->registerDebugDraw(this, [](toast::Node3D& node) {
			static_cast<Collider&>(node).drawDebug();
		});
	}
}
void Collider::destroy() {
	if (renderer::VulkanRenderer::instance) renderer::VulkanRenderer::instance->unregisterDebugDraw(this);
}
void Collider::onEnable() {
	m_debug_visible = true;
}
void Collider::onDisable() {
	m_debug_visible = false;
}
void Collider::drawDebug() {
	ZoneScoped;
	if (!m_debug_visible) return;
	const glm::vec4 color = disabled ? glm::vec4(0.5f, 0.5f, 0.5f, debug_color.a) : debug_color;
	syncTransform();
	auto transform = glm::translate(glm::mat4(1.0f), world_position) * glm::mat4_cast(world_rotation);
	if (const auto sphere = box().as<SphereCollider>(); sphere.exists()) {
		glm::vec3 center = world_position;
		if (const auto body = parent().as<Rigidbody>(); body.exists()) {
			body->syncTransform();
			center = body->world_position + body->world_rotation * position;
		}
		if (!std::isfinite(sphere->radius) || sphere->radius <= 0.0f) return;
		renderer::debugDrawSphere(center, sphere->radius, color);
		if (debug_fill) {
			auto fill_color = color;
			fill_color.a *= 0.2f;
			renderer::debugDrawSolidSphere(center, sphere->radius, fill_color);
		}
	} else if (const auto capsule = box().as<CapsuleCollider>(); capsule.exists()) {
		renderer::debugDrawCapsule(transform, capsule->radius, capsule->height, color, debug_fill);
	} else if (const auto cube = box().as<BoxCollider>(); cube.exists()) {
		renderer::debugDrawShapeBox(glm::scale(transform, cube->size), color, debug_fill);
	}
}
}
