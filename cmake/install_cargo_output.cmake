# Copies one or more cargo-built binaries out of the shared target directory into their install location.
#
# BINARY_NAMES is a list, because spawning a cmake process per binary costs more than the copy does - the
# asset packer crate alone produces three.

if(NOT BINARY_NAMES)
    message(FATAL_ERROR "install_cargo_output: BINARY_NAMES is empty")
endif()

# cargo drops output in <target>/<profile> normally, but <target>/<triple>/<profile> when a target triple is
# configured (a .cargo/config.toml or CARGO_BUILD_TARGET). Resolve the directory once from the first binary
list(GET BINARY_NAMES 0 _first)
set(_probe "${_first}${EXE_SUFFIX}")

if(EXISTS "${CARGO_TARGET_DIR}/${PROFILE_NAME}/${_probe}")
    set(_src_dir "${CARGO_TARGET_DIR}/${PROFILE_NAME}")
else()
    file(GLOB _candidates "${CARGO_TARGET_DIR}/*/${PROFILE_NAME}/${_probe}")
    if(NOT _candidates)
        message(FATAL_ERROR
            "cargo output not found for '${_probe}'\n"
            "Searched:\n"
            "  ${CARGO_TARGET_DIR}/${PROFILE_NAME}/\n"
            "  ${CARGO_TARGET_DIR}/*/${PROFILE_NAME}/")
    endif()
    list(GET _candidates 0 _found)
    cmake_path(GET _found PARENT_PATH _src_dir)
endif()

file(MAKE_DIRECTORY "${DEST_DIR}")

foreach(_name IN LISTS BINARY_NAMES)
    set(_binary "${_name}${EXE_SUFFIX}")
    if(NOT EXISTS "${_src_dir}/${_binary}")
        message(FATAL_ERROR "cargo output not found: ${_src_dir}/${_binary}")
    endif()
    # file(COPY) compares timestamps, so an unchanged binary is not rewritten
    file(COPY "${_src_dir}/${_binary}" DESTINATION "${DEST_DIR}")
endforeach()

if(EXISTS "${_src_dir}/templates")
    file(COPY "${_src_dir}/templates" DESTINATION "${DEST_DIR}")
endif()
