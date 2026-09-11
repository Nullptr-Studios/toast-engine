/*
 * @file slang_vfs.hpp
 * @author Xein
 * @date 17 Jul 2026
 */

#pragma once

#include <slang.h>
#include <string>
#include <string_view>
#include <vector>

namespace renderer {

/**
 * @class SlangVfs
 * @brief ISlangFileSystem implementation resolving virtual URIs through the AssetManager
 *
 * Lets @c import and @c #include inside .slang modules resolve against the engine VFS
 */
class SlangVfs final : public ISlangFileSystem {
public:
	static auto get() -> SlangVfs&;

	// ISlangUnknown
	SLANG_NO_THROW auto SLANG_MCALL queryInterface(const SlangUUID& uuid, void** out_object) -> SlangResult override;

	SLANG_NO_THROW auto SLANG_MCALL addRef() -> uint32_t override { return 1; }

	SLANG_NO_THROW auto SLANG_MCALL release() -> uint32_t override { return 1; }

	// ISlangCastable
	SLANG_NO_THROW auto SLANG_MCALL castAs(const SlangUUID& uuid) -> void* override;

	// ISlangFileSystem
	SLANG_NO_THROW auto SLANG_MCALL loadFile(const char* path, ISlangBlob** out_blob) -> SlangResult override;

	/// Rebuilds a clean "pack://relative" URI from paths Slang may have mangled
	static auto normalizeUri(std::string_view path) -> std::string;

	/// Creates an arc ISlangBlob owning a copy of the given bytes
	static auto makeBlob(const void* data, size_t size) -> ISlangBlob*;

	/**
	 * @brief Records every URI resolved through this file system while it is alive
	 *
	 * The shader cache needs to know which files a shader imports, so that editing an imported module
	 * recompiles everything that reads it. Slang's own `getDependencyFileCount()` is not that list: for a
	 * module loaded from source it reports nothing at all, even when the imports were resolved right here.
	 * Without this, a change to lighting.slang would leave every shader importing it running last build's
	 * SPIR-V.
	 *
	 * Deliberately not thread-safe. ShaderCache serializes compilation under its own mutex, and a recorder
	 * that quietly interleaved two compiles would file one shader's imports under another's name
	 */
	class Recorder {
	public:
		Recorder();
		~Recorder();

		Recorder(const Recorder&) = delete;
		auto operator=(const Recorder&) -> Recorder& = delete;
		Recorder(Recorder&&) = delete;
		auto operator=(Recorder&&) -> Recorder& = delete;

		/// @returns the URIs resolved so far, in load order and without duplicates
		[[nodiscard]]
		auto resolved() const -> const std::vector<std::string>& {
			return m_resolved;
		}

	private:
		friend class SlangVfs;

		void record(std::string uri);

		std::vector<std::string> m_resolved;
	};

private:
	/// Innermost active Recorder, or null. See Recorder for why one is enough
	static inline Recorder* s_recorder = nullptr;
};

}
