/**
 * @file renderer_settings.hpp
 * @author dario
 * @date 16/08/2026
 */

#pragma once
#include <toast/export.hpp>

namespace renderer {

class VulkanRenderer;

/**
 * @brief Settings that must be applied before any pass is built
 *
 * ShadowPass reads its resolution when it allocates. onChange() fires once at declaration, so declaring this
 * after the pass exists would apply the stored value to a pass that had already sized itself - every run,
 * meaning the setting could never take effect
 *
 * @note Call after Settings::load(), before the passes exist. Needs no renderer
 */
TOAST_API void registerRendererStartupSettings();

/**
 * @brief Declares the rest of the renderer settings and binds each to what it controls
 *
 * These declarations are the only description of the settings anywhere - the editor window is generated from
 * the registry, so adding a knob here is the whole change. onChange() fires at declaration, so the stored
 * value is applied on startup by the same path a later edit takes
 *
 * @note Call after the post-process passes exist. The quality toggles resolve a pass by name, and an
 * unmatched name is ignored, so declaring early would drop every stored toggle
 */
TOAST_API void registerRendererSettings(VulkanRenderer& renderer);

}
