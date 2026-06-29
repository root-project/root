# BUILD_BYPRODUCTS, the sub-build's binary dir (ExternalProject's
# default, passed as -B explicitly on the emscripten path), and
# IMPORTED_LOCATION (resolved via ExternalProject_Get_Property
# binary_dir) all need to agree on where googletest builds.
# ExternalProject's binary_dir defaults to
# ${CMAKE_CURRENT_BINARY_DIR}/<name>-prefix/src/<name>-build, so
# tracking CMAKE_CURRENT_BINARY_DIR keeps the three in sync whether
# this module is consumed standalone or via add_subdirectory (e.g.
# under root-project/root, where CMAKE_BINARY_DIR is root-build but
# this directory is root-build/interpreter/CppInterOp/unittests).
set(_gtest_byproduct_binary_dir
  ${CMAKE_CURRENT_BINARY_DIR}/googletest-prefix/src/googletest-build)
set(_gtest_byproducts
  ${_gtest_byproduct_binary_dir}/lib/libgtest.a
  ${_gtest_byproduct_binary_dir}/lib/libgtest_main.a
  ${_gtest_byproduct_binary_dir}/lib/libgmock.a
  ${_gtest_byproduct_binary_dir}/lib/libgmock_main.a
  )

if(WIN32)
  set(EXTRA_GTEST_OPTS
    -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG:PATH=${_gtest_byproduct_binary_dir}/lib/
    -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL:PATH=${_gtest_byproduct_binary_dir}/lib/
    -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE:PATH=${_gtest_byproduct_binary_dir}/lib/
    -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO:PATH=${_gtest_byproduct_binary_dir}/lib/
    -Dgtest_force_shared_crt=ON)
elseif(APPLE)
  set(EXTRA_GTEST_OPTS -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT})
endif()

include(ExternalProject)
# Forward parent CMAKE_CXX_FLAGS to the gtest sub-build so sanitizer
# and -stdlib=libc++ additions don't get dropped (else gtest builds
# against system defaults and ABI-clashes with the parent at link).
# gtest is third-party and not warning-clean under LLVM's -Werror
# regime (gcc's ASan instrumentation trips -Wmaybe-uninitialized in
# gtest-death-test.cc at -O3), so a trailing -w silences warnings.
set(GOOGLETEST_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
if(MSVC)
  set(GOOGLETEST_CMAKE_CXX_FLAGS "${GOOGLETEST_CMAKE_CXX_FLAGS} /w")
else()
  set(GOOGLETEST_CMAKE_CXX_FLAGS "${GOOGLETEST_CMAKE_CXX_FLAGS} -w")
endif()
if (EMSCRIPTEN)
  # FIXME: -sSUPPORT_LONGJMP=wasm in the default option causes a warning in the Emscripten build of Googletest
  # and as we treat warnings as errors in the ci, it causes the ci to fail.
  string(REPLACE "-sSUPPORT_LONGJMP=wasm" "" GOOGLETEST_CMAKE_CXX_FLAGS "${GOOGLETEST_CMAKE_CXX_FLAGS}")
  set(config_cmd emcmake${EMCC_SUFFIX} cmake)
  if(CMAKE_GENERATOR STREQUAL "Ninja")
    set(build_cmd emmake${EMCC_SUFFIX} ninja)
  else()
    set(build_cmd emmake${EMCC_SUFFIX} make)
  endif()
else()
  set(build_cmd ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR}/googletest-prefix/src/googletest-build/ --config $<CONFIG>)
endif()

set(_gtest_cmake_args
  -DCMAKE_BUILD_TYPE=$<CONFIG>
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  -DCMAKE_CXX_FLAGS=${GOOGLETEST_CMAKE_CXX_FLAGS}
  # HandleLLVMOptions puts -stdlib=libc++ / -fsanitize=*
  # in CMAKE_*_LINKER_FLAGS for LLVM_USE_SANITIZER.
  -DCMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}
  -DCMAKE_MODULE_LINKER_FLAGS=${CMAKE_MODULE_LINKER_FLAGS}
  -DCMAKE_SHARED_LINKER_FLAGS=${CMAKE_SHARED_LINKER_FLAGS}
  -DCMAKE_AR=${CMAKE_AR}
  -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
  ${EXTRA_GTEST_OPTS})

if(EMSCRIPTEN)
  set(_gtest_configure
    CONFIGURE_COMMAND ${config_cmd} -G ${CMAKE_GENERATOR}
      -S ${CMAKE_CURRENT_BINARY_DIR}/googletest-prefix/src/googletest/
      -B ${CMAKE_CURRENT_BINARY_DIR}/googletest-prefix/src/googletest-build/
      ${_gtest_cmake_args})
else()
  set(_gtest_configure CMAKE_ARGS ${_gtest_cmake_args})
endif()

ExternalProject_Add(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_SHALLOW FALSE
  GIT_TAG fa8438ae6b70c57010177de47a9f13d7041a6328
  UPDATE_COMMAND ""
  ${_gtest_configure}
  BUILD_COMMAND ${build_cmd}
  # Disable install step
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS ${_gtest_byproducts}
  # Wrap download, configure and build steps in a script to log output
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  LOG_OUTPUT_ON_FAILURE ON
  TIMEOUT 600
  )

# Specify include dirs for gtest and gmock
ExternalProject_Get_Property(googletest source_dir)
set(GTEST_INCLUDE_DIR ${source_dir}/googletest/include)
set(GMOCK_INCLUDE_DIR ${source_dir}/googlemock/include)
# Create the directories. Prevents bug https://gitlab.kitware.com/cmake/cmake/issues/15052
file(MAKE_DIRECTORY ${GTEST_INCLUDE_DIR} ${GMOCK_INCLUDE_DIR})

# Libraries
ExternalProject_Get_Property(googletest binary_dir)
if(WIN32)
  set(_G_LIBRARY_PATH  ${_gtest_byproduct_binary_dir}/lib)
else()
  set(_G_LIBRARY_PATH ${binary_dir}/lib/)
endif()

# Use gmock_main instead of gtest_main because it initializes gtest as well.
# Note: The libraries are listed in reverse order of their dependencies.
foreach(lib gtest gtest_main gmock gmock_main)
  add_library(${lib} IMPORTED STATIC GLOBAL)
  set_target_properties(${lib} PROPERTIES
    IMPORTED_LOCATION "${_G_LIBRARY_PATH}${CMAKE_STATIC_LIBRARY_PREFIX}${lib}${CMAKE_STATIC_LIBRARY_SUFFIX}"
    INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIRS}"
    )
  add_dependencies(${lib} googletest)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND
      ${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL 9)
    target_compile_options(${lib} INTERFACE -Wno-deprecated-copy)
  endif()
endforeach()
target_include_directories(gtest INTERFACE ${GTEST_INCLUDE_DIR})
target_include_directories(gmock INTERFACE ${GMOCK_INCLUDE_DIR})

set_property(TARGET gtest PROPERTY IMPORTED_LOCATION ${_G_LIBRARY_PATH}/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX})
set_property(TARGET gtest_main PROPERTY IMPORTED_LOCATION ${_G_LIBRARY_PATH}/${CMAKE_STATIC_LIBRARY_PREFIX}gtest_main${CMAKE_STATIC_LIBRARY_SUFFIX})
set_property(TARGET gmock PROPERTY IMPORTED_LOCATION ${_G_LIBRARY_PATH}/${CMAKE_STATIC_LIBRARY_PREFIX}gmock${CMAKE_STATIC_LIBRARY_SUFFIX})
set_property(TARGET gmock_main PROPERTY IMPORTED_LOCATION ${_G_LIBRARY_PATH}/${CMAKE_STATIC_LIBRARY_PREFIX}gmock_main${CMAKE_STATIC_LIBRARY_SUFFIX})
