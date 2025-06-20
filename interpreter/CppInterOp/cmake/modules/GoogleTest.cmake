set(_gtest_byproduct_binary_dir
  ${CMAKE_BINARY_DIR}/downloads/googletest-prefix/src/googletest-build)
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
if (EMSCRIPTEN)
  if (CMAKE_C_COMPILER MATCHES ".bat")
    set(config_cmd emcmake.bat cmake)
    if(CMAKE_GENERATOR STREQUAL "Ninja")
      set(build_cmd emmake.bat ninja)
    else()
      set(build_cmd emmake.bat make)
    endif()
  else()
    set(config_cmd emcmake cmake)
    if(CMAKE_GENERATOR STREQUAL "Ninja")
      set(build_cmd emmake ninja)
    else()
      set(build_cmd emmake make)
    endif()
  endif()
else()
  set(config_cmd ${CMAKE_COMMAND})
  set(build_cmd ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}/unittests/googletest-prefix/src/googletest-build/ --config $<CONFIG>)
endif()

ExternalProject_Add(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_SHALLOW 1
  GIT_TAG v1.17.0
  UPDATE_COMMAND ""
  # # Force separate output paths for debug and release builds to allow easy
  # # identification of correct lib in subsequent TARGET_LINK_LIBRARIES commands
  # CMAKE_ARGS -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG:PATH=DebugLibs
  #            -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE:PATH=ReleaseLibs
  #            -Dgtest_force_shared_crt=ON
  CONFIGURE_COMMAND ${config_cmd} -G ${CMAKE_GENERATOR}
  		-S ${CMAKE_BINARY_DIR}/unittests/googletest-prefix/src/googletest/
		-B ${CMAKE_BINARY_DIR}/unittests/googletest-prefix/src/googletest-build/
                -DCMAKE_BUILD_TYPE=$<CONFIG>
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                -DCMAKE_AR=${CMAKE_AR}
                -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
                ${EXTRA_GTEST_OPTS}
  BUILD_COMMAND ${build_cmd}
  # Disable install step
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS ${_gtest_byproducts}
  # Wrap download, configure and build steps in a script to log output
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
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
