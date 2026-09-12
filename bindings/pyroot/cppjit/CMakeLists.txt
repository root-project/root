cmake_minimum_required(VERSION 3.20)
project(CppJIT LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)

# This option won't make a lot of sense since we only ship the shared library in site-packages
# Perhaps this should permanently be OFF and users can build their own CppInterOp if they want to run the tests?
option(CPPJIT_ENABLE_CPPINTEROP_TESTS "enable CppInterOp tests" OFF)
set(CPPINTEROP_GIT_REPOSITORY "https://github.com/compiler-research/CppInterOp.git" CACHE STRING "")
set(CPPINTEROP_GIT_TAG "9802d61921ad5688ae42e4e628d754fc1192244d" CACHE STRING "")
set(CPPINTEROP_SOURCE_DIR "" CACHE PATH
    "Override default CppInterOp built by ExternalProject_Add, with a path to local CppInterOp source")

# The full Development component requires libpython, which manylinux
# images do not ship and extension modules do not need.
set(Python_FIND_VIRTUALENV ONLY)
find_package(Python COMPONENTS Interpreter Development.Module)
if(NOT Python_FOUND)
    set(Python_FIND_VIRTUALENV STANDARD)
    find_package(Python COMPONENTS Interpreter Development.Module)
endif()
if(NOT Python_Development.Module_FOUND)
    message(FATAL_ERROR "Python development headers not found")
endif()

# The LLVM range the pinned CppInterOp supports; update together with the tag.
set(CPPJIT_LLVM_VERSION_MIN 20)
set(CPPJIT_LLVM_VERSION_MAX 22)

# Runtime fallback anchor for the dispatch (see cppinterop_paths()): the
# installed package under scikit-build-core, the build tree otherwise --
# a plain `cmake --build` yields a runnable tree under <build>/python.
if(SKBUILD)
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('platlib'))"
        OUTPUT_VARIABLE _python_platlib
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(CPPINTEROP_INSTALL_PREFIX "${_python_platlib}/cppjit")
else()
    set(CPPINTEROP_INSTALL_PREFIX "${CMAKE_BINARY_DIR}")
endif()

# CppInterOp stages into the build tree in the runtime layout; the wheel
# ships a subset of it.
set(CPPINTEROP_STAGE_DIR "${CMAKE_BINARY_DIR}/interop")

# Acquire CppInterOp: an external prebuilt one (CppInterOp_DIR /
# CMAKE_PREFIX_PATH), else the pinned ExternalProject. Exports the
# CPPJIT_INTEROP_* coordinates consumed below.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/AddCppInterOp.cmake)
cppjit_add_cppinterop()

# this libcppjit.so merges both cpyrt and the interop wrapper
file(GLOB CPYRT_SOURCES CONFIGURE_DEPENDS src/cpyrt/*.cxx)
set(INTEROP_SOURCES
    src/interop/interop_wrapper.cxx
    src/interop/cppinterop_dispatch.cxx
)

add_library(cppjit SHARED ${CPYRT_SOURCES} ${INTEROP_SOURCES})
if(TARGET CppInterOp)
    add_dependencies(cppjit CppInterOp)
endif()

# The wrapper anchors these coordinates at its own load location, falling
# back to the install prefix (see cppinterop_paths()); the clang major
# names the versioned compiler probed for the runtime resource dir.
target_compile_definitions(cppjit PRIVATE
    CPPINTEROP_INSTALL_PREFIX="${CPPINTEROP_INSTALL_PREFIX}"
    CPPINTEROP_LIBRARY="${CPPJIT_INTEROP_LIBRARY}"
    CPPINTEROP_INCLUDE_DIR="${CPPJIT_INTEROP_RUNTIME_INCLUDES}"
    CPPJIT_CLANG_MAJOR="${CPPJIT_INTEROP_CLANG_MAJOR}"
    CPPJIT_CLANG_INCLUDE_DIR="${CPPJIT_INTEROP_CLANG_DIR}"
)

target_include_directories(cppjit PRIVATE
    # src/ itself resolves the public "cpyrt/*.h" spellings against the
    # flattened src/cpyrt/ directory
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpyrt
    ${CMAKE_CURRENT_SOURCE_DIR}/src/interop
    ${CPPJIT_INTEROP_COMPILE_INCLUDES}
    ${Python_INCLUDE_DIRS}
)

target_compile_options(cppjit PRIVATE
    -Wall -Wno-strict-aliasing -Wno-register
)

if(CMAKE_COMPILER_IS_GNUCXX)
    target_link_options(cppjit PRIVATE -Wl,-Bsymbolic-functions)
endif()

# macOS needs to allow undefined Python symbols
if(APPLE)
    set_target_properties(cppjit PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup"
        SUFFIX ".so"
    )
endif()

set_target_properties(cppjit PROPERTIES
    OUTPUT_NAME "cppjit"
    PREFIX "lib"
)

# A plain build yields a runnable tree: the extension and the python
# sources assemble under <build>/python, beside <build>/interop.
if(NOT SKBUILD)
    set_target_properties(cppjit PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/cppjit"
    )
    add_custom_target(cppjit-python-tree ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/python/cppjit"
            "${CMAKE_BINARY_DIR}/python/cppjit"
    )
endif()

# the extension lives inside the package (import cppjit.libcppjit)
install(TARGETS cppjit
    LIBRARY DESTINATION cppjit
)

# scikit-build-core ships the python sources itself (wheel.packages); a
# plain CMake install lays them out here so PYTHONPATH=<prefix> works.
if(NOT SKBUILD)
    install(DIRECTORY python/cppjit
        DESTINATION .
        PATTERN "__pycache__" EXCLUDE
    )
endif()

# An external CppInterOp is consumed in place and nothing of it ships;
# the bundled one installs with the builtin headers of the build clang,
# laid out as a headers-only resource dir (only include/ ships).
if(TARGET CppInterOp)
    install(DIRECTORY "${CPPINTEROP_STAGE_DIR}/lib/"
        DESTINATION cppjit/interop/lib
    )
    install(DIRECTORY "${CPPINTEROP_STAGE_DIR}/include/"
        DESTINATION cppjit/interop/include
    )
    install(DIRECTORY "${CPPJIT_INTEROP_CLANG_RESOURCE_DIR}/include/"
        DESTINATION "cppjit/interop/lib/clang/${CPPJIT_INTEROP_CLANG_MAJOR}/include"
    )
endif()

# the public cpyrt API headers keep their installed cpyrt/ prefix
install(FILES
    src/cpyrt/API.h
    src/cpyrt/CommonDefs.h
    src/cpyrt/DispatchPtr.h
    src/cpyrt/PyException.h
    src/cpyrt/Reflex.h
    DESTINATION cppjit/interop/include/cpyrt
)
