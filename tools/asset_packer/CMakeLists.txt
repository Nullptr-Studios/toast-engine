project(asset_packer NONE)

set(CARGO_PROJECT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(CARGO_TARGET_DIR "${OUTPUT_ROOT}/$<CONFIG>/cargo")
set(CARGO_PROFILE_NAME "$<IF:$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>,release,debug>")
set(CARGO_PROFILE_FLAG "$<IF:$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>,release,dev>")

find_program(CARGO_EXECUTABLE cargo REQUIRED)

# One cargo invocation for all three binaries. They are three [[bin]] entries of a single crate, so the three
# targets this used to have each ran the same `cargo build` and then copied a different result out - building
# the crate three times over, concurrently, on one target directory.
# One cargo invocation for all three binaries. They are three [[bin]] entries of a single crate, so the three
# targets this used to have each ran the same `cargo build` and then copied a different result out - building
# the crate three times over, concurrently, on one target directory.
add_custom_target(packer ALL
        COMMAND ${CARGO_EXECUTABLE} build --profile "${CARGO_PROFILE_FLAG}" --target-dir "${CARGO_TARGET_DIR}"
        COMMAND ${CMAKE_COMMAND}
            -DCARGO_TARGET_DIR=${CARGO_TARGET_DIR}
            -DPROFILE_NAME=${CARGO_PROFILE_NAME}
            "-DBINARY_NAMES=packer;unpacker;pack_viewer"
            -DEXE_SUFFIX=${CMAKE_EXECUTABLE_SUFFIX}
            -DDEST_DIR=${OUTPUT_ROOT}/$<CONFIG>/packer
            -P ${CMAKE_SOURCE_DIR}/cmake/install_cargo_output.cmake
        WORKING_DIRECTORY ${CARGO_PROJECT_DIR}
        COMMENT "[cargo] Building asset packer/unpacker/pack_viewer (${CARGO_PROFILE_NAME})..."
        VERBATIM
)

# Kept as names so `add_dependencies(editor unpacker)` and friends still resolve; `packer` does the work
add_custom_target(unpacker ALL)
add_dependencies(unpacker packer)

add_custom_target(pack_viewer ALL)
add_dependencies(pack_viewer packer)
