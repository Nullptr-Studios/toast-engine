/**
 * @file cube_face_basis.hpp
 * @author dario
 * @date 05/08/2026
 *
 * @brief Per-face basis for rendering into a cubemap, shared by everything that fills one
 */

#pragma once

#include <cstdint>
#include <glm/glm.hpp>

namespace renderer {

/// @brief Right/up/forward of one cube face
struct CubeFaceBasis {
	glm::vec3 right;
	glm::vec3 up;
	glm::vec3 forward;
};

/**
 * @brief Basis of cube face @p face, in the +X,-X,+Y,-Y,+Z,-Z order Vulkan lays cube layers out in
 *
 * Consumed by directionForFace() in environment.slang. Shared rather than duplicated per pass: the
 * environment convolutions and the probe prefilter write cubemaps the same shader later samples, so any
 * disagreement about face orientation shows up as reflections subtly rotated in one and not the other
 */
[[nodiscard]]
inline auto cubeFaceBasis(uint32_t face) -> CubeFaceBasis {
	switch (face) {
		case 0:
			return {
			  {0.0f,  0.0f, -1.0f},
        {0.0f, -1.0f,  0.0f},
        {1.0f,  0.0f,  0.0f}
			};
		case 1:
			return {
			  { 0.0f,  0.0f, 1.0f},
        { 0.0f, -1.0f, 0.0f},
        {-1.0f,  0.0f, 0.0f}
			};
		case 2:
			return {
			  {1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
        {0.0f, 1.0f, 0.0f}
			};
		case 3:
			return {
			  {1.0f,  0.0f,  0.0f},
        {0.0f,  0.0f, -1.0f},
        {0.0f, -1.0f,  0.0f}
			};
		case 4:
			return {
			  {1.0f,  0.0f, 0.0f},
        {0.0f, -1.0f, 0.0f},
        {0.0f,  0.0f, 1.0f}
			};
		default:
			return {
			  {-1.0f,  0.0f,  0.0f},
        { 0.0f, -1.0f,  0.0f},
        { 0.0f,  0.0f, -1.0f}
			};
	}
}

}
