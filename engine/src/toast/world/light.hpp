/**
 * @file light.hpp
 * @author Xein
 * @date 22 Jun 2026
 *
 * @brief Base light node; registers with the renderer so it can be collected into the per-frame light list
 */

#pragma once
#include "node_3d.hpp"

#include <toast/export.hpp>

namespace toast {

/// @brief Concrete light kind, tagged once by each subclass's constructor - not RTTI, not reflected
enum class LightType : uint8_t {
	none,
	point,
	directional,
	spot,
	ambient,
};

class [[ToastNode, Hidden, Icon("PointLight")]] TOAST_API Light : public Node3D {
public:
	Light() = default;

	[[nodiscard]]
	auto lightType() const -> LightType {
		return m_light_type;
	}

	[[nodiscard]]
	auto color() const -> const glm::vec3& {
		return m_light_color;
	}

	[[nodiscard]]
	auto intensity() const -> float {
		return m_intensity;
	}

protected:
	/// @brief Tags the concrete light kind; called once from each subclass's own constructor body
	void setLightType(LightType type) { m_light_type = type; }

private:
	void init();
	void end();
	void destroy();

	[[Reflect, Color]]
	glm::vec3 m_light_color = glm::vec3(1.0f, 1.0f, 1.0f);

	[[Reflect, Unit("lm")]]
	float m_intensity = 1.0f;

	// Deliberately not [[Reflect]] - applyFields() only ever touches reflected fields during prefab
	// deserialization, so this constructor-set value can never be clobbered
	LightType m_light_type = LightType::none;

	bool m_registered_proxy = false;
};
}
