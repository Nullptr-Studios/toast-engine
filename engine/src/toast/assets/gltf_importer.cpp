#include "gltf_importer.hpp"

#include "gltf_importer.h"    // ffi

#define GLM_ENABLE_EXPERIMENTAL

#include "asset_manager.hpp"
#include "mesh.hpp"
#include "prefab.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <functional>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <nlohmann/json.hpp>
#include <span>
#include <toast/uid.hpp>
#include <unordered_map>
#define TINYGLTF3_IMPLEMENTATION
#define TINYGLTF3_ENABLE_FS
#include <tiny_gltf_v3.h>
#include <toast/log.hpp>

using namespace tinygltf3;

namespace assets {

namespace {

/// @brief Decodes a standard (RFC 4648) base64 payload, e.g. the part of a data: URI after the comma
auto decodeBase64(std::string_view input) -> std::vector<uint8_t> {
	static constexpr std::string_view k_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

	std::array<int8_t, 256> lookup {};
	lookup.fill(-1);
	for (size_t i = 0; i < k_alphabet.size(); ++i) {
		lookup[static_cast<uint8_t>(k_alphabet[i])] = static_cast<int8_t>(i);
	}

	std::vector<uint8_t> out;
	out.reserve(input.size() / 4 * 3);

	int32_t val = 0;
	int32_t bits = -8;
	for (const char c : input) {
		const int8_t digit = lookup[static_cast<uint8_t>(c)];
		if (digit == -1) {
			continue;    // padding ('='), whitespace, or line breaks - just skip
		}
		val = (val << 6) + digit;
		bits += 6;
		if (bits >= 0) {
			out.push_back(static_cast<uint8_t>((val >> bits) & 0xFF));
			bits -= 8;
		}
	}
	return out;
}

/// @brief Percent-decodes a glTF URI (e.g. "%20" -> ' '), per the glTF spec's URI encoding requirement
auto percentDecode(std::string_view uri) -> std::string {
	std::string out;
	out.reserve(uri.size());
	for (size_t i = 0; i < uri.size(); ++i) {
		if (uri[i] == '%' && i + 2 < uri.size()) {
			out.push_back(static_cast<char>(std::stoi(std::string(uri.substr(i + 1, 2)), nullptr, 16)));
			i += 2;
		} else {
			out.push_back(uri[i]);
		}
	}
	return out;
}

/// @brief Sniffs the KTX2 file signature (the fixed 12-byte magic every .ktx2 file starts with, per the
/// KTX 2.0 spec) directly off the bytes, rather than trusting mimeType/extension - some pipelines already
/// pack textures as KTX2 inside the glTF (via KHR_texture_basisu or just a raw .ktx2 uri/bufferView with no
/// declared mimeType), and re-running them through toktx is both wasteful and pointless
auto isKtx2(std::span<const uint8_t> data) -> bool {
	static constexpr std::array<uint8_t, 12> k_ktx2_magic {0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};
	return data.size() >= k_ktx2_magic.size() && std::equal(k_ktx2_magic.begin(), k_ktx2_magic.end(), data.begin());
}

/// @brief Resolves an image's raw file bytes (this importer always parses with images_as_is, so these are
/// undecoded PNG/JPEG bytes, not pixels) from whichever of the three ways glTF can store image data is
/// actually present: an embedded GLB bufferView, a base64 data: URI, or a URI pointing at an external file
/// relative to the source .gltf/.glb. Returns an empty vector (and logs) if none of these produced data -
/// callers must not assume a non-empty result
auto loadImageBytes(const tg3_model& model, const tg3_image& img, const std::filesystem::path& base_dir, size_t index)
    -> std::vector<uint8_t> {
	if (img.buffer_view != -1) {
		const auto& bv = model.buffer_views[img.buffer_view];
		const auto& buf = model.buffers[bv.buffer];
		const uint8_t* raw = buf.data.data + bv.byte_offset;
		return {raw, raw + bv.byte_length};
	}

	const std::string uri(img.uri.data, img.uri.len);
	if (uri.empty()) {
		TOAST_ERROR("AssetManager", "Image {} has neither a bufferView nor a uri; skipping", index);
		return {};
	}

	if (uri.starts_with("data:")) {
		const auto comma = uri.find(',');
		if (comma == std::string::npos) {
			TOAST_ERROR("AssetManager", "Image {} has a malformed data URI; skipping", index);
			return {};
		}
		return decodeBase64(std::string_view(uri).substr(comma + 1));
	}

	const std::filesystem::path file_path = base_dir / percentDecode(uri);
	std::ifstream file(file_path, std::ios::binary | std::ios::ate);
	if (!file) {
		TOAST_ERROR("AssetManager", "Image {} references '{}', which could not be opened", index, file_path.string());
		return {};
	}

	const auto size = file.tellg();
	file.seekg(0, std::ios::beg);
	std::vector<uint8_t> data(static_cast<size_t>(size));
	file.read(reinterpret_cast<char*>(data.data()), size);
	return data;
}

/// @brief Looks up an integer field nested inside a named glTF extension object (e.g. KHR_texture_basisu's
/// "source"). tinygltf3 has no first-class struct field for this - extensions it doesn't specifically model
/// are only exposed as generic tg3_value trees off tg3_extras_ext::extensions
auto findExtensionInt(const tg3_extras_ext& ext, std::string_view ext_name, std::string_view key) -> int32_t {
	for (uint32_t i = 0; i < ext.extensions_count; ++i) {
		const auto& e = ext.extensions[i];
		if (std::string_view(e.name.data, e.name.len) != ext_name) {
			continue;
		}
		if (e.value.type != TG3_VALUE_OBJECT) {
			return -1;
		}
		for (uint32_t j = 0; j < e.value.object_count; ++j) {
			const auto& kv = e.value.object_data[j];
			if (kv.value.type == TG3_VALUE_INT && std::string_view(kv.key.data, kv.key.len) == key) {
				return static_cast<int32_t>(kv.value.int_val);
			}
		}
		return -1;
	}
	return -1;
}

}    // namespace

auto generateIntermediates(const std::filesystem::path& path) {
	const glm::mat4 gltf_y_up_to_engine_z_up = glm::rotate(glm::mat4(1.0F), glm::radians(90.0F), glm::vec3(1.0F, 0.0F, 0.0F));
	const glm::mat4 engine_z_up_to_gltf_y_up = glm::transpose(gltf_y_up_to_engine_z_up);

	// NOLINTBEGIN(modernize-return-braced-init-list)
	auto to_engine_space_vec3 = [&](const glm::vec3& v) -> glm::vec3 {
		return glm::vec3(gltf_y_up_to_engine_z_up * glm::vec4(v, 1.0F));
	};

	auto to_engine_space_dir3 = [&](const glm::vec3& v) -> glm::vec3 {
		return glm::vec3(gltf_y_up_to_engine_z_up * glm::vec4(v, 0.0F));
	};
	// NOLINTEND(modernize-return-braced-init-list)

	auto to_engine_space_mat4 = [&](const glm::mat4& m) -> glm::mat4 {
		return gltf_y_up_to_engine_z_up * m * engine_z_up_to_gltf_y_up;
	};

	tg3_parse_options options;
	tg3_error_stack errors;
	tg3_model model;

	tg3_parse_options_init(&options);
	options.images_as_is = 1;
	tg3_error_stack_init(&errors);

	std::string path_str = path.string();
	auto success = tg3_parse_file(&model, &errors, path_str.c_str(), path_str.size(), &options);

	if (success != TG3_OK) {
		for (size_t i = 0; i < errors.count; ++i) {
			TOAST_ERROR("AssetManager", "{}", errors.entries[i].message);
		}
		return;
	}

	// map each GLTF mesh index to the name of the first scene node that references it
	std::vector<std::string> mesh_node_name(model.meshes_count);
	for (size_t i = 0; i < model.nodes_count; i++) {
		const tg3_node& n = model.nodes[i];
		if (n.mesh != -1 && mesh_node_name[n.mesh].empty()) {
			std::string name(n.name.data, n.name.len);
			mesh_node_name[n.mesh] = name.empty() ? "node_" + std::to_string(i) : name;
		}
	}

	struct MeshFile {
		std::unique_ptr<Mesh> mesh;
		std::string file_name;
	};

	std::vector<MeshFile> mesh_files;
	std::vector<std::vector<int>> mesh_prim_to_file(model.meshes_count);
	int prim_counter = 0;

	// Disambiguates mesh base names so two distinct meshes can never collide on the same .tmesh filename
	std::unordered_map<std::string, int> mesh_name_counts;

	for (size_t i = 0; i < model.meshes_count; ++i) {
		const auto& m = model.meshes[i];
		std::string base_name = !mesh_node_name[i].empty() ? mesh_node_name[i] : std::string(m.name.data, m.name.len);
		if (base_name.empty()) {
			base_name = "mesh_" + std::to_string(i);
		}
		if (auto it = mesh_name_counts.find(base_name); it != mesh_name_counts.end()) {
			base_name = base_name + "_" + std::to_string(it->second);
			++it->second;
		} else {
			mesh_name_counts[base_name] = 1;
		}

		auto accessor_bytes = [&](int acc_idx) -> const uint8_t* {
			const auto& acc = model.accessors[acc_idx];
			const auto& bv = model.buffer_views[acc.buffer_view];
			const auto& buf = model.buffers[bv.buffer];
			return buf.data.data + bv.byte_offset + acc.byte_offset;
		};

		auto get_stride = [&](int acc_idx, size_t tight_size) {
			const auto& bv = model.buffer_views[model.accessors[acc_idx].buffer_view];
			return bv.byte_stride != 0 ? bv.byte_stride : tight_size;
		};

		mesh_prim_to_file[i].resize(m.primitives_count);

		for (uint32_t pi = 0; pi < m.primitives_count; ++pi) {
			const tg3_primitive& prim = m.primitives[pi];
			const bool is_triangles = prim.mode == -1 || prim.mode == 4;
			TOAST_ASSERT(is_triangles, "AssetManager", "Mesh primitive is not triangles");

			auto find_attr = [&](const char* attr_name) {
				for (uint32_t j = 0; j < prim.attributes_count; j++) {
					if (strcmp(prim.attributes[j].key.data, attr_name) == 0) {
						return prim.attributes[j].value;
					}
				}
				return -1;
			};

			int pos_idx = find_attr("POSITION");
			if (pos_idx == -1) {
				// TOAST_ASSERT is compiled out in Release and isn't guaranteed to halt in Debug either -
				// falling through would index model.accessors[-1] next, straight out-of-bounds
				TOAST_ERROR("AssetManager", "Mesh primitive {} of mesh {} has no POSITION attribute; aborting import", pi, i);
				return;
			}
			int norm_idx = find_attr("NORMAL");
			int uv_idx = find_attr("TEXCOORD_0");
			int tan_idx = find_attr("TANGENT");
			int col_idx = find_attr("COLOR_0");

			const uint32_t vertex_count = model.accessors[pos_idx].count;
			std::vector<renderer::Vertex> vertices(vertex_count);

			const uint8_t* pos_data = accessor_bytes(pos_idx);
			const uint8_t* norm_data = norm_idx != -1 ? accessor_bytes(norm_idx) : nullptr;
			const uint8_t* uv_data = uv_idx != -1 ? accessor_bytes(uv_idx) : nullptr;
			const uint8_t* tan_data = tan_idx != -1 ? accessor_bytes(tan_idx) : nullptr;
			const uint8_t* col_data = col_idx != -1 ? accessor_bytes(col_idx) : nullptr;

			for (uint32_t j = 0; j < vertex_count; j++) {
				auto& v = vertices[j];
				memcpy(&v.position, pos_data + (j * get_stride(pos_idx, sizeof(glm::vec3))), sizeof(glm::vec3));
				v.position = to_engine_space_vec3(v.position);
				if (norm_data) {
					memcpy(&v.normal, norm_data + (j * get_stride(norm_idx, sizeof(glm::vec3))), sizeof(glm::vec3));
					v.normal = to_engine_space_dir3(v.normal);
				}
				if (uv_data) {
					memcpy(&v.uv, uv_data + (j * get_stride(uv_idx, sizeof(glm::vec2))), sizeof(glm::vec2));
				}
				if (tan_data) {
					glm::vec4 t;
					memcpy(&t, tan_data + (j * get_stride(tan_idx, sizeof(glm::vec4))), sizeof(glm::vec4));
					const glm::vec3 tangent_xyz = to_engine_space_dir3(glm::vec3(t.x, t.y, t.z));
					v.tangent = glm::vec4(tangent_xyz, t.w);    // w is handedness
				}
				if (col_data) {
					memcpy(&v.color, col_data + (j * get_stride(col_idx, sizeof(glm::vec3))), sizeof(glm::vec3));
				}
			}

			if (prim.indices == -1) {
				// Non-indexed primitives are technically valid glTF, but this importer doesn't support them
				// (no triangulation-on-the-fly path) - fail cleanly instead of indexing model.accessors[-1]
				TOAST_ERROR("AssetManager", "Mesh primitive {} of mesh {} has no indices; aborting import", pi, i);
				return;
			}
			const auto& idx_acc = model.accessors[prim.indices];
			const uint8_t* idx_data = accessor_bytes(prim.indices);
			std::vector<assets::Mesh::Index> indices(idx_acc.count);
			for (uint32_t j = 0; j < idx_acc.count; j++) {
				if (idx_acc.component_type == 5125) {
					indices[j] = reinterpret_cast<const uint32_t*>(idx_data)[j];
				} else if (idx_acc.component_type == 5123) {
					indices[j] = reinterpret_cast<const uint16_t*>(idx_data)[j];
				} else {
					indices[j] = idx_data[j];
				}
			}

			std::string file_name = (m.primitives_count == 1) ? base_name : base_name + "_" + std::to_string(prim_counter);

			mesh_files.push_back(
			    {.mesh = std::make_unique<Mesh>(std::string_view(file_name), std::move(vertices), std::move(indices)),
			     .file_name = std::move(file_name)}
			);
			mesh_prim_to_file[i][pi] = static_cast<int>(mesh_files.size()) - 1;
			++prim_counter;
		}
	}
	TOAST_TRACE("AssetManager", "Imported {} meshes", mesh_files.size());

	// Textures
	struct TextureData {
		std::vector<uint8_t> data;
		std::string name;
		std::string format;
	};

	// Index-aligned with model.textures (resized, not reserved+push_back'd) - tex_uid() below indexes this
	// by the raw glTF texture index, so a failed/skipped image must still leave a (empty) slot behind
	// rather than shifting every later texture's index down by one
	std::vector<TextureData> textures(model.textures_count);

	// Disambiguates texture names so two distinct images never collide on disk - a silent overwrite would
	// leave whichever material referenced the earlier one pointing at the wrong image
	std::unordered_map<std::string, int> texture_name_counts;

	for (size_t i = 0; i < model.textures_count; i++) {
		const auto& texture = model.textures[i];

		// KHR_texture_basisu (used by exporters that ship pre-compressed KTX2 textures, sometimes marked
		// extensionsRequired) points at its image through a nested extension object instead of the plain
		// "source" property - tinygltf3 doesn't parse that extension into a dedicated field, so every such
		// texture would otherwise read source == -1 and get skipped even though the image is right there
		const int32_t source = texture.source != -1 ? texture.source : findExtensionInt(texture.ext, "KHR_texture_basisu", "source");
		if (source == -1) {
			// TOAST_ASSERT is compiled out in Release and isn't guaranteed to halt in Debug either - this
			// must be a real early-out, not just a diagnostic, since indexing model.images[-1] next would be
			// straight out-of-bounds
			TOAST_ERROR("AssetManager", "Texture {} has no image source; skipping", i);
			continue;
		}
		const auto& img = model.images[source];

		std::string name(img.name.data, img.name.len);
		if (name.empty()) {
			std::string uri(img.uri.data, img.uri.len);
			if (!uri.empty()) {
				name = std::filesystem::path(uri).stem().string();
			} else {
				name = "texture_" + std::to_string(i);
			}
		}
		if (auto it = texture_name_counts.find(name); it != texture_name_counts.end()) {
			name = name + "_" + std::to_string(it->second);
			++it->second;
		} else {
			texture_name_counts[name] = 1;
		}

		auto data = loadImageBytes(model, img, path.parent_path(), i);

		std::string format;
		if (isKtx2(data)) {
			// Already compressed - skip straight past mimeType/extension guessing entirely so this never
			// gets routed through a PNG/JPEG path (and, downstream, a wasted re-run through toktx)
			format = "image/ktx2";
		} else {
			format = std::string(img.mime_type.data, img.mime_type.len);
			if (format.empty()) {
				// External file references commonly omit mimeType - the extension is the only signal left
				const std::string uri(img.uri.data, img.uri.len);
				const std::string ext = std::filesystem::path(uri).extension().string();
				format = (ext == ".jpg" || ext == ".jpeg") ? "image/jpeg" : "image/png";
			}
		}

		textures[i] = {
		  .data = std::move(data),
		  .name = std::move(name),
		  .format = std::move(format),
		};
	}
	TOAST_TRACE("AssetManager", "Imported {} textures", textures.size());

	// Materials
	std::vector<toml::table> materials;
	materials.reserve(model.materials_count);

	// Resolved upfront - this is the single source of truth for a material's on-disk/reference identity,
	// used both for the .tmat filename below and every MeshNode's "material" reference in walk_node().
	// Previously walk_node() referenced the raw glTF material name directly, which silently diverged from
	// the name actually used to save the .tmat file whenever a material had no name (very common from
	// non-Blender exporters) or two materials shared a name - the scene node's material reference would
	// then never match any UID during the C# import patch step and end up unset
	std::unordered_map<std::string, int> material_name_counts;
	std::vector<std::string> material_file_names(model.materials_count);
	for (size_t i = 0; i < model.materials_count; i++) {
		std::string name(model.materials[i].name.data, model.materials[i].name.len);
		if (name.empty()) {
			name = "material_" + std::to_string(i);
		}
		if (auto it = material_name_counts.find(name); it != material_name_counts.end()) {
			name = name + "_" + std::to_string(it->second);
			++it->second;
		} else {
			material_name_counts[name] = 1;
		}
		material_file_names[i] = name;
	}

	for (size_t i = 0; i < model.materials_count; i++) {
		const auto& mat = model.materials[i];
		const auto& pbr = mat.pbr_metallic_roughness;

		auto tex_uid = [&](int32_t tex_idx) -> std::string {
			if (tex_idx == -1) {
				return "";
			}
			return textures[tex_idx].name;
		};

		toml::table material_table;
		material_table.insert("uid", "");
		material_table.insert("name", material_file_names[i]);
		material_table.insert("vertex", "NOT IMPLEMENTED");      // TODO: Replace with UID of default vertex shader
		material_table.insert("fragment", "NOT IMPLEMENTED");    // TODO: Replace with UID of default fragment shader

		if (pbr.base_color_texture.index != -1) {
			material_table.insert("albedo_map", tex_uid(pbr.base_color_texture.index));
		}
		if (mat.normal_texture.index != -1) {
			material_table.insert("normal_map", tex_uid(mat.normal_texture.index));
		}
		material_table.insert(
		    "color",
		    toml::array {pbr.base_color_factor[0], pbr.base_color_factor[1], pbr.base_color_factor[2], pbr.base_color_factor[3]}
		);

		// Extended PBR fields kept in params for future use
		toml::table params;
		if (pbr.metallic_roughness_texture.index != -1) {
			params.insert("metallic_roughness_map", tex_uid(pbr.metallic_roughness_texture.index));
		}
		if (mat.normal_texture.index != -1) {
			params.insert("normal_scale", mat.normal_texture.scale);
		}
		if (mat.occlusion_texture.index != -1) {
			params.insert("occlusion_map", tex_uid(mat.occlusion_texture.index));
			params.insert("occlusion_strength", mat.occlusion_texture.strength);
		}
		if (mat.emissive_texture.index != -1) {
			params.insert("emissive_map", tex_uid(mat.emissive_texture.index));
		}
		params.insert("metallic_factor", pbr.metallic_factor);
		params.insert("roughness_factor", pbr.roughness_factor);
		params.insert("emissive_factor", toml::array {mat.emissive_factor[0], mat.emissive_factor[1], mat.emissive_factor[2]});
		material_table.insert("params", std::move(params));

		toml::table render_state;
		render_state.insert("cull_mode", mat.double_sided != 0 ? "none" : "back");
		render_state.insert("blend_mode", std::string(mat.alpha_mode.data, mat.alpha_mode.len));
		render_state.insert("depth_write", true);
		render_state.insert("alpha_cutoff", mat.alpha_cutoff);
		material_table.insert("render_state", std::move(render_state));

		materials.push_back(std::move(material_table));
	}
	TOAST_TRACE("AssetManager", "Imported {} materials", materials.size());

	// Cameras
	struct CameraData {
		std::string name;
		std::string type;    // "perspective" or "orthographic"
		// perspective
		double aspect_ratio;
		double yfov;
		double znear;
		double zfar;
		// orthographic
		double xmag;
		double ymag;
	};

	std::vector<CameraData> cameras;
	cameras.reserve(model.cameras_count);

	for (size_t i = 0; i < model.cameras_count; i++) {
		const auto& cam = model.cameras[i];
		std::string type(cam.type.data, cam.type.len);
		cameras.push_back({
		  .name = std::string(cam.name.data, cam.name.len),
		  .type = type,
		  .aspect_ratio = cam.perspective.aspect_ratio,
		  .yfov = cam.perspective.yfov,
		  .znear = cam.perspective.znear,
		  .zfar = cam.perspective.zfar,
		  .xmag = cam.orthographic.xmag,
		  .ymag = cam.orthographic.ymag,
		});
	}
	TOAST_TRACE("AssetManager", "Imported {} cameras", cameras.size());

	// Lights
	struct LightData {
		std::string name;
		std::string type;    // "directional", "point", "spot"
		glm::vec3 color;
		double intensity;
		double range;
		double inner_cone_angle;
		double outer_cone_angle;
	};

	std::vector<LightData> lights;
	lights.reserve(model.lights_count);

	for (size_t i = 0; i < model.lights_count; i++) {
		const auto& light = model.lights[i];
		lights.push_back({
		  .name = std::string(light.name.data, light.name.len),
		  .type = std::string(light.type.data, light.type.len),
		  .color = {light.color[0], light.color[1], light.color[2]},
		  .intensity = light.intensity,
		  .range = light.range,
		  .inner_cone_angle = light.spot.inner_cone_angle,
		  .outer_cone_angle = light.spot.outer_cone_angle,
		});
	}
	TOAST_TRACE("AssetManager", "Imported {} lights", lights.size());

	// Scenes
	std::vector<nlohmann::json> scenes;
	scenes.reserve(model.scenes_count);

	auto node_transform = [&](const tg3_node& node) -> nlohmann::json {
		nlohmann::json transform;
		glm::mat4 local_transform = glm::mat4(1.0F);

		if (node.has_matrix) {
			memcpy(&local_transform, node.matrix, sizeof(glm::mat4));
		} else {
			const glm::vec3 t(node.translation[0], node.translation[1], node.translation[2]);
			const glm::quat r(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2]);
			const glm::vec3 s(node.scale[0], node.scale[1], node.scale[2]);

			local_transform = glm::translate(glm::mat4(1.0F), t) * glm::mat4_cast(r) * glm::scale(glm::mat4(1.0F), s);
		}

		const glm::mat4 converted = to_engine_space_mat4(local_transform);
		glm::vec3 t, s, skew;
		glm::vec4 persp;
		glm::quat r;
		glm::decompose(converted, s, r, t, skew, persp);

		transform["pos"] = {t.x, t.y, t.z};
		transform["rot"] = {r.x, r.y, r.z, r.w};
		transform["scl"] = {s.x, s.y, s.z};
		return transform;
	};

	std::function<nlohmann::json(int32_t)> walk_node = [&](int32_t node_idx) -> nlohmann::json {
		const tg3_node& node = model.nodes[node_idx];
		nlohmann::json n;
		n["name"] = std::string(node.name.data, node.name.len);
		n["transform"] = node_transform(node);

		if (node.mesh != -1) {
			const auto& gltf_mesh = model.meshes[node.mesh];
			const auto& prim_files = mesh_prim_to_file[node.mesh];

			if (gltf_mesh.primitives_count == 1) {
				n["type"] = "toast::MeshNode";
				n["params"]["mesh"] = mesh_files[prim_files[0]].file_name;
				const auto& prim = gltf_mesh.primitives[0];
				if (prim.material != -1) {    // TODO: Update field names when MeshNode fields are created
					n["params"]["material"] = material_file_names[prim.material];
				}
			} else {
				n["type"] = "toast::Node3D";
				n["children"] = nlohmann::json::array();
				for (uint32_t pi = 0; pi < gltf_mesh.primitives_count; ++pi) {
					const auto& prim = gltf_mesh.primitives[pi];
					const std::string& fname = mesh_files[prim_files[pi]].file_name;
					nlohmann::json child;
					child["name"] = fname;
					child["type"] = "toast::MeshNode";
					child["transform"]["pos"] = {0.0f, 0.0f, 0.0f};
					child["transform"]["rot"] = {0.0f, 0.0f, 0.0f, 1.0f};
					child["transform"]["scl"] = {1.0f, 1.0f, 1.0f};
					child["params"]["mesh"] = fname;
					if (prim.material != -1) {    // TODO: Update field names when MeshNode fields are created
						child["params"]["material"] = material_file_names[prim.material];
					}
					n["children"].push_back(std::move(child));
				}
			}
		} else if (node.camera != -1) {
			const auto& cam = cameras[node.camera];
			n["type"] = "toast::Camera";
			n["params"]["projection"] = cam.type;    // TODO: change this once the camera has been made
			if (cam.type == "perspective") {
				n["params"]["fov"] = cam.yfov;
				n["params"]["near"] = cam.znear;
				n["params"]["far"] = cam.zfar;
				n["params"]["aspect"] = cam.aspect_ratio;
			} else {
				n["params"]["xmag"] = cam.xmag;
				n["params"]["ymag"] = cam.ymag;
				n["params"]["near"] = cam.znear;
				n["params"]["far"] = cam.zfar;
			}
		} else if (node.light != -1) {
			const auto& light = lights[node.light];
			if (light.type == "directional") {
				n["type"] = "toast::DirectionalLight";
			} else if (light.type == "point") {
				n["type"] = "toast::PointLight";
			} else if (light.type == "spot") {
				n["type"] = "toast::Spotlight";
			}

			n["params"]["color"] = {light.color[0], light.color[1], light.color[2]};
			n["params"]["intensity"] = light.intensity;
			if (light.range > 0.0) {
				n["params"]["range"] = light.range;
			}
			if (light.type == "spot") {
				n["params"]["inner_cone_angle"] = light.inner_cone_angle;
				n["params"]["outer_cone_angle"] = light.outer_cone_angle;
			}
		} else {
			n["type"] = "toast::Node3D";
		}

		if (node.children_count > 0) {
			if (!n.contains("children")) {
				n["children"] = nlohmann::json::array();
			}
			for (uint32_t i = 0; i < node.children_count; i++) {
				n["children"].push_back(walk_node(node.children[i]));
			}
		}

		return n;
	};

	for (size_t i = 0; i < model.scenes_count; i++) {
		const auto& scene = model.scenes[i];
		nlohmann::json scene_json;
		std::string name = std::string(scene.name.data, scene.name.len);
		if (name.empty()) {
			name = path.filename().stem().string();
		}
		scene_json["name"] = name;
		scene_json["children"] = nlohmann::json::array();

		for (uint32_t j = 0; j < scene.nodes_count; j++) {
			scene_json["children"].push_back(walk_node(scene.nodes[j]));
		}

		scenes.push_back(std::move(scene_json));
	}
	TOAST_TRACE("AssetManager", "Imported {} nodes", scenes.size());

	// Save files in cache://<name_without_extension>/
	std::string base_name = path.stem().string();
	std::filesystem::path cache_dir = AssetManager::get().getCachePath() / base_name;
	std::filesystem::create_directories(cache_dir);

	// Save meshes
	for (const auto& mf : mesh_files) {
		auto binary = mf.mesh->toBinary();
		std::filesystem::path out = cache_dir / (mf.file_name + ".tmesh");
		std::ofstream f(out, std::ios::binary);
		f.write(reinterpret_cast<const char*>(binary.data()), binary.size());
	}
	TOAST_TRACE("AssetManager", "Saved {} meshes", mesh_files.size());

	// Save textures
	for (const auto& tex : textures) {
		if (tex.data.empty()) {
			continue;    // failed to load (already logged) - don't write a bogus empty file
		}
		std::string_view ext = tex.format == "image/jpeg" ? ".jpg" : tex.format == "image/ktx2" ? ".ktx2" : ".png";
		std::filesystem::path out = cache_dir / (tex.name + std::string(ext));
		std::ofstream f(out, std::ios::binary);
		f.write(reinterpret_cast<const char*>(tex.data.data()), tex.data.size());
	}
	TOAST_TRACE("AssetManager", "Saved {} texture intermediates", textures.size());

	// Save materials
	for (size_t i = 0; i < materials.size(); ++i) {
		const std::filesystem::path out = cache_dir / (material_file_names[i] + ".tmat");
		std::ofstream f(out);
		f << materials[i];
	}
	TOAST_TRACE("AssetManager", "Saved {} materials", materials.size());

	// Save scenes
	for (const auto& scene : scenes) {
		std::string name = scene["name"].get<std::string>();
		std::filesystem::path out = cache_dir / (name + ".json");
		std::ofstream f(out);
		f << scene.dump(2);
	}
	TOAST_TRACE("AssetManager", "Saved {} node intermediates", scenes.size());

	tg3_model_free(&model);
	tg3_error_stack_free(&errors);
}

static void jsonToTnode(const nlohmann::json& scene_json, const std::filesystem::path& out) {
	Prefab prefab;

	std::function<void(const nlohmann::json&, const std::string&)> walk = [&](const nlohmann::json& n,
	                                                                          const std::string& parent_uid_str) {
		Prefab::BasicNode basic;
		basic.name = n["name"].get<std::string>();
		basic.type = n.value("type", "toast::Node3D");
		if (basic.type == "Node") {
			basic.type = "toast::Node3D";
		}

		toast::UID uid = toast::UID::make();
		const std::string uid_str = uid.get();

		basic.fields.push_back({"m_uid", toast::FieldType::uid_t, false, uid});
		basic.fields.push_back({"m_name", toast::FieldType::string_t, false, basic.name});
		basic.fields.push_back({"m_local_enabled", toast::FieldType::bool_t, false, true});
		if (!parent_uid_str.empty()) {
			basic.fields.push_back({"m_parent", toast::FieldType::uid_t, false, toast::UID(toast::UID::fromString(parent_uid_str))});
		}

		if (n.contains("transform")) {
			Prefab::Group tg;
			tg.name = "Transform";
			const auto& t = n["transform"];
			const auto& p = t["pos"];
			const auto& r = t["rot"];
			const auto& s = t["scl"];
			tg.fields.push_back({
			  "m_position", toast::FieldType::vec3_t, false, glm::vec3 {p[0].get<float>(), p[1].get<float>(), p[2].get<float>()}
			});
			tg.fields.push_back({
			  "m_rotation",
			  toast::FieldType::quaternion_t,
			  false,
			  glm::quat {r[3].get<float>(), r[0].get<float>(), r[1].get<float>(), r[2].get<float>()}
			});
			tg.fields.push_back({
			  "m_scale", toast::FieldType::vec3_t, false, glm::vec3 {s[0].get<float>(), s[1].get<float>(), s[2].get<float>()}
			});
			basic.groups.push_back(std::move(tg));
		}

		if (basic.type == "toast::MeshNode" && n.contains("params")) {
			const auto& params = n["params"];
			// TODO: Update field names when MeshNode fields are created
			if (params.contains("mesh")) {
				const auto mesh_uid = params["mesh"].get<std::string>();
				if (mesh_uid.size() == 11) {
					basic.fields.push_back({"m_mesh", toast::FieldType::uid_t, false, toast::UID(toast::UID::fromString(mesh_uid))});
				} else {
					TOAST_WARN("AssetManager", "GLTF scene contains non-UID mesh reference '{}'; skipping m_mesh assignment", mesh_uid);
				}
			}
			if (params.contains("material")) {
				const auto material_uid = params["material"].get<std::string>();
				if (material_uid.size() == 11) {
					basic.fields.push_back(
					    {"m_material", toast::FieldType::uid_t, false, toast::UID(toast::UID::fromString(material_uid))}
					);
				} else {
					TOAST_WARN(
					    "AssetManager", "GLTF scene contains non-UID material reference '{}'; skipping m_material assignment", material_uid
					);
				}
			}
		}

		// toast::AmbientLight has no glTF equivalent (KHR_lights_punctual only covers
		// directional/point/spot), so it never appears here - nothing to map for it
		const bool is_light =
		    basic.type == "toast::DirectionalLight" || basic.type == "toast::PointLight" || basic.type == "toast::Spotlight";
		if (is_light && n.contains("params")) {
			const auto& params = n["params"];

			if (params.contains("color")) {
				const auto& c = params["color"];
				basic.fields.push_back({
				  "m_light_color", toast::FieldType::vec3_t, false, glm::vec3 {c[0].get<float>(), c[1].get<float>(), c[2].get<float>()}
				});
			}
			if (params.contains("intensity")) {
				// glTF intensity is physical (candela for point/spot, lux for directional); this engine's
				// m_intensity is a plain unitless shading multiplier with no such conversion - passed through
				// as-is since there's no established target scale to convert to yet
				basic.fields.push_back({"m_intensity", toast::FieldType::float_t, false, params["intensity"].get<float>()});
			}

			// PointLight/Spotlight only: glTF's range (0/absent = infinite) has no equivalent in this
			// engine's clustered-lighting model, which requires a finite culling radius - when range is
			// absent, leave m_attenuation at its class default rather than writing a bogus 0
			if ((basic.type == "toast::PointLight" || basic.type == "toast::Spotlight") && params.contains("range")) {
				basic.fields.push_back({"m_attenuation", toast::FieldType::float_t, false, params["range"].get<float>()});
			}

			if (basic.type == "toast::Spotlight") {
				if (params.contains("inner_cone_angle")) {
					basic.fields.push_back(
					    {"m_inner_radius", toast::FieldType::float_t, false, glm::degrees(params["inner_cone_angle"].get<float>())}
					);
				}
				if (params.contains("outer_cone_angle")) {
					basic.fields.push_back(
					    {"m_outer_radius", toast::FieldType::float_t, false, glm::degrees(params["outer_cone_angle"].get<float>())}
					);
				}
			}
		}

		prefab.nodes.push_back(std::move(basic));

		if (n.contains("children")) {
			for (const auto& c : n["children"]) {
				walk(c, uid_str);
			}
		}
	};

	walk(scene_json, "");

	std::ofstream f(out);
	f << prefab.toFile();
}
}

extern "C" {

void gltf_generate_intermediates(const char* path) noexcept {
	// Both FFI entry points are noexcept, so an uncaught exception anywhere below (json parsing, std::stoi
	// on a malformed percent-escape, std::any_cast, std::filesystem errors, ...) would otherwise call
	// std::terminate() - converting it to a logged error instead means a bad/unusual input file fails
	// loudly and diagnosably rather than crashing the whole editor or silently producing zero output
	try {
		std::filesystem::path dir {path};
		assets::generateIntermediates(dir);
	} catch (const std::exception& e) { TOAST_ERROR("AssetManager", "GLTF import failed: {}", e.what()); } catch (...) {
		TOAST_ERROR("AssetManager", "GLTF import failed with an unrecognized exception");
	}
}

void gltf_create_tnode(const char* json_path, const char* output_path) noexcept {
	try {
		std::ifstream f(json_path);
		assets::jsonToTnode(nlohmann::json::parse(f), std::filesystem::path(output_path));
	} catch (const std::exception& e) {
		TOAST_ERROR("AssetManager", "GLTF scene-to-tnode conversion failed: {}", e.what());
	} catch (...) { TOAST_ERROR("AssetManager", "GLTF scene-to-tnode conversion failed with an unrecognized exception"); }
}
}
