# Acquires CppInterOp. A distribution or developer supplies a prebuilt one
# -- an install prefix or a build directory -- through CppInterOp_DIR or
# CMAKE_PREFIX_PATH, and it is consumed in place with nothing bundled.
# Otherwise the pinned tag (or CPPINTEROP_SOURCE_DIR) builds as an
# ExternalProject and stages for bundling; default backend is clang-repl,
# CPPJIT_USE_CLING builds against a provided cling. Coordinates exported to
# the parent scope, relative ones anchored at the runtime layout and
# absolute ones standing on their own (see cppinterop_paths()):
#   CPPJIT_INTEROP_LIBRARY            runtime library
#   CPPJIT_INTEROP_RUNTIME_INCLUDES   runtime include dirs, ':'-joined
#   CPPJIT_INTEROP_COMPILE_INCLUDES   include dirs for building the wrapper
#   CPPJIT_INTEROP_CLANG_MAJOR        clang major of the interpreter
#   CPPJIT_INTEROP_CLANG_DIR          resource-dir coordinate; empty = probe
#   CPPJIT_INTEROP_CLANG_RESOURCE_DIR builtin headers to bundle (install only)
include_guard(GLOBAL)
include(ExternalProject)

# Hint LLVM discovery at an active conda toolchain.
set(_llvm_hints "")
if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND _llvm_hints "$ENV{CONDA_PREFIX}/lib/cmake/llvm")
endif()

# Developer toggle: Cling from either ROOT or standalone supplies LLVM_DIR/Clang_DIR
option(CPPJIT_USE_CLING "Build CppInterOp against a prebuilt Cling C++ Interpreter (ROOT)" OFF)
mark_as_advanced(CPPJIT_USE_CLING)

set(Cling_DIR "" CACHE PATH "Cling CMake config dir; derived from LLVM_DIR when empty (cling mode only)")
# Surface Cling_DIR in ccmake/GUI only when cling mode is on.
if(CPPJIT_USE_CLING)
    mark_as_advanced(CLEAR Cling_DIR)
else()
    mark_as_advanced(Cling_DIR)
endif()

function(cppjit_add_cppinterop)
    # An explicit CppInterOp_DIR is authoritative, like LLVM_DIR below.
    if(DEFINED CppInterOp_DIR)
        find_package(CppInterOp CONFIG REQUIRED PATHS "${CppInterOp_DIR}" NO_DEFAULT_PATH)
    else()
        find_package(CppInterOp CONFIG QUIET)
    endif()

    set(_clang_resource_dir "")
    if(CppInterOp_FOUND)
        if(CPPINTEROP_SOURCE_DIR)
            message(FATAL_ERROR
                "An external CppInterOp and CPPINTEROP_SOURCE_DIR are mutually exclusive")
        endif()
        # The developer owns compatibility: warn, do not fail.
        if(CPPINTEROP_LLVM_VERSION_MAJOR LESS CPPJIT_LLVM_VERSION_MIN OR
           CPPINTEROP_LLVM_VERSION_MAJOR GREATER CPPJIT_LLVM_VERSION_MAX)
            message(WARNING
                "External CppInterOp embeds LLVM ${CPPINTEROP_LLVM_VERSION}, outside "
                "the tested ${CPPJIT_LLVM_VERSION_MIN}-${CPPJIT_LLVM_VERSION_MAX}")
        endif()
        # Builtin headers from a matching host LLVM when one is findable; an
        # empty coordinate leaves them to the wrapper's runtime probe.
        find_package(LLVM CONFIG QUIET HINTS ${_llvm_hints})
        set(_clang_dir "")
        if(LLVM_FOUND AND LLVM_VERSION_MAJOR EQUAL CPPINTEROP_LLVM_VERSION_MAJOR
           AND EXISTS "${LLVM_LIBRARY_DIR}/clang/${LLVM_VERSION_MAJOR}/include")
            set(_clang_dir "${LLVM_LIBRARY_DIR}/clang/${LLVM_VERSION_MAJOR}")
        endif()

        set(_library "${CPPINTEROP_LIBRARIES}")
        set(_compile_includes "${CPPINTEROP_INCLUDE_DIRS}")
        list(JOIN CPPINTEROP_INCLUDE_DIRS ":" _runtime_includes)
        set(_clang_major "${CPPINTEROP_LLVM_VERSION_MAJOR}")
        message(STATUS
            "CppInterOp: external ${CPPINTEROP_VERSION} (LLVM ${CPPINTEROP_LLVM_VERSION}) "
            "at ${CPPINTEROP_LIBRARIES}")
    else()
        if(DEFINED LLVM_DIR)
            # An explicit LLVM_DIR is authoritative: fail instead of falling back to a
            # different LLVM than the one requested. A failed find_package resets
            # LLVM_DIR to -NOTFOUND, so keep the requested value for the message.
            set(_llvm_dir_arg "${LLVM_DIR}")
            find_package(LLVM CONFIG PATHS "${LLVM_DIR}" NO_DEFAULT_PATH)
            if(NOT LLVM_FOUND)
                message(FATAL_ERROR
                    "No LLVMConfig.cmake under LLVM_DIR (${_llvm_dir_arg}); expected "
                    "<install prefix or build tree>/lib/cmake/llvm")
            endif()
        else()
            find_package(LLVM CONFIG QUIET HINTS ${_llvm_hints})
            if(NOT LLVM_FOUND)
                message(FATAL_ERROR
                    "No LLVM CMake package found. Install LLVM "
                    "${CPPJIT_LLVM_VERSION_MIN}-${CPPJIT_LLVM_VERSION_MAX} development packages "
                    "(apt: llvm-${CPPJIT_LLVM_VERSION_MAX}-dev libclang-${CPPJIT_LLVM_VERSION_MAX}-dev; "
                    "conda: llvmdev clangdev), or point cppjit at your own LLVM build with "
                    "-DLLVM_DIR=<prefix or build tree>/lib/cmake/llvm "
                    "(pip: --config-settings=cmake.define.LLVM_DIR=...)")
            endif()
        endif()

        message(STATUS "Found LLVM ${LLVM_VERSION} at ${LLVM_DIR}")
        if(LLVM_VERSION_MAJOR LESS CPPJIT_LLVM_VERSION_MIN OR
           LLVM_VERSION_MAJOR GREATER CPPJIT_LLVM_VERSION_MAX)
            message(FATAL_ERROR
                "LLVM ${LLVM_VERSION} is unsupported: the currently supported "
                "CppInterOp version (${CPPINTEROP_GIT_TAG}) only supports LLVM "
                "${CPPJIT_LLVM_VERSION_MIN}-${CPPJIT_LLVM_VERSION_MAX}")
        endif()

        if(DEFINED Clang_DIR)
            # An explicit Clang_DIR is authoritative, like LLVM_DIR above.
            set(_clang_dir_arg "${Clang_DIR}")
            find_package(Clang CONFIG PATHS "${Clang_DIR}" NO_DEFAULT_PATH)
            if(NOT Clang_FOUND)
                message(FATAL_ERROR
                    "No ClangConfig.cmake under Clang_DIR (${_clang_dir_arg}); expected "
                    "<install prefix or build tree>/lib/cmake/clang")
            endif()
        else()
            # Clang's package sits beside LLVM's in every supported layout; search
            # only there so an unrelated system clang cannot satisfy the lookup.
            find_package(Clang CONFIG QUIET HINTS "${LLVM_DIR}/../clang" NO_DEFAULT_PATH)
        endif()
        if(Clang_FOUND)
            message(STATUS "Found Clang at ${Clang_DIR}")
        endif()

        set(_args
            -DLLVM_DIR=${LLVM_DIR}
            -DCPPINTEROP_ENABLE_TESTING=${CPPJIT_ENABLE_CPPINTEROP_TESTS}
            -DBUILD_SHARED_LIBS=ON
            # The wheel ships a single unversioned library file.
            -DCPPINTEROP_SHARED_LIBRARY_VERSIONING=OFF
            -DCMAKE_INSTALL_PREFIX=${CPPINTEROP_STAGE_DIR}
            -DCMAKE_INSTALL_LIBDIR=lib
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_CXX_STANDARD=17
        )

        if(CPPJIT_USE_CLING)
            set(_cling_dir "${Cling_DIR}")
            if(NOT _cling_dir)
                # LLVM_DIR is <prefix>/lib/cmake/llvm; cling config sits at the
                # sibling <prefix>/lib/cmake/cling (setup-llvm flavor:cling layout).
                get_filename_component(_prefix "${LLVM_DIR}" DIRECTORY)   # <prefix>/lib/cmake
                get_filename_component(_prefix "${_prefix}" DIRECTORY)    # <prefix>/lib
                get_filename_component(_prefix "${_prefix}" DIRECTORY)    # <prefix>
                set(_cling_dir "${_prefix}/lib/cmake/cling")
            endif()
            message(STATUS "CppInterOp backend: Cling (found at ${_cling_dir})")
            list(APPEND _args
                -DCPPINTEROP_USE_CLING=ON
                -DCPPINTEROP_USE_REPL=OFF
                -DCling_DIR=${_cling_dir}
            )
        else()
            list(APPEND _args
                -DCPPINTEROP_USE_REPL=ON
                -DCPPINTEROP_USE_CLING=OFF
            )
        endif()

        if(Clang_DIR)
            list(APPEND _args -DClang_DIR=${Clang_DIR})
        endif()
        if(CMAKE_C_COMPILER)
            list(APPEND _args -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER})
        endif()
        if(CMAKE_CXX_COMPILER)
            list(APPEND _args -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
        endif()

        set(_source_args
            GIT_REPOSITORY ${CPPINTEROP_GIT_REPOSITORY}
            GIT_TAG        ${CPPINTEROP_GIT_TAG}
        )
        set(_log_args
            LOG_DOWNLOAD   ON
            LOG_CONFIGURE  ON
            LOG_BUILD      ON
            LOG_INSTALL    ON
            LOG_OUTPUT_ON_FAILURE ON
        )
        if(CPPINTEROP_SOURCE_DIR)
            message(STATUS "CppInterOp: building from local source at ${CPPINTEROP_SOURCE_DIR} "
                           "over the currently supported version ${CPPINTEROP_GIT_TAG}")
            # BUILD_ALWAYS recompiles uncommitted edits and refreshes the installed
            # libclangCppInterOp on every build; the sub-build keeps this incremental.
            set(_source_args
                SOURCE_DIR   "${CPPINTEROP_SOURCE_DIR}"
                BUILD_ALWAYS ON
            )
            # Stream sub-build output in the dev loop instead of hiding it in log files.
            set(_log_args "")
        endif()

        # Keep the symbol table in debug-info builds; .dynsym alone
        # suffices for the dlsym-based dispatch everywhere else.
        string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
        set(_install_lib_target install-clangCppInterOp)
        if(NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG" AND
           NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "RELWITHDEBINFO")
            string(APPEND _install_lib_target "-stripped")
        endif()

        # Install only the library and headers, not CppInterOp's full install tree.
        ExternalProject_Add(CppInterOp
            ${_source_args}
            PREFIX         "${CMAKE_BINARY_DIR}/CppInterOp"
            CMAKE_ARGS     ${_args}
            INSTALL_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR>
                            --target ${_install_lib_target} install-cppinterop-headers
            BUILD_BYPRODUCTS
                "${CPPINTEROP_STAGE_DIR}/lib/libclangCppInterOp${CMAKE_SHARED_LIBRARY_SUFFIX}"
            ${_log_args}
        )

        # Verify that the user-provided CppInterOp source override is legitimate.
        ExternalProject_Add_Step(CppInterOp verify_project
            COMMAND ${CMAKE_COMMAND}
                -DCPPINTEROP_BINARY_DIR=<BINARY_DIR>
                -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/VerifyCppInterOp.cmake"
            DEPENDEES configure
            DEPENDERS build
            COMMENT "Verifying the configured source tree is CppInterOp"
        )

        set(_clang_resource_dir "${LLVM_LIBRARY_DIR}/clang/${LLVM_VERSION_MAJOR}")
        if(NOT EXISTS "${_clang_resource_dir}/include")
            message(FATAL_ERROR
                "No builtin headers at ${_clang_resource_dir}/include; the LLVM at "
                "${LLVM_DIR} carries no clang resource directory")
        endif()

        set(_library "interop/lib/libclangCppInterOp${CMAKE_SHARED_LIBRARY_SUFFIX}")
        set(_compile_includes "${CPPINTEROP_STAGE_DIR}/include")
        set(_runtime_includes "interop/include")
        set(_clang_major "${LLVM_VERSION_MAJOR}")
        # The wheel reads its bundled copy; a raw build tree bundles nothing
        # and uses the build LLVM's headers where they are.
        if(SKBUILD)
            set(_clang_dir "interop/lib/clang/${LLVM_VERSION_MAJOR}")
        else()
            set(_clang_dir "${_clang_resource_dir}")
        endif()
    endif()

    set(CPPJIT_INTEROP_LIBRARY "${_library}" PARENT_SCOPE)
    set(CPPJIT_INTEROP_RUNTIME_INCLUDES "${_runtime_includes}" PARENT_SCOPE)
    set(CPPJIT_INTEROP_COMPILE_INCLUDES "${_compile_includes}" PARENT_SCOPE)
    set(CPPJIT_INTEROP_CLANG_MAJOR "${_clang_major}" PARENT_SCOPE)
    set(CPPJIT_INTEROP_CLANG_DIR "${_clang_dir}" PARENT_SCOPE)
    set(CPPJIT_INTEROP_CLANG_RESOURCE_DIR "${_clang_resource_dir}" PARENT_SCOPE)
endfunction()
