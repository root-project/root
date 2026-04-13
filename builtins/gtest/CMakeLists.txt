# Add googletest
# http://stackoverflow.com/questions/9689183/cmake-googletest

set(_gtest_byproduct_binary_dir
  ${CMAKE_CURRENT_BINARY_DIR}/googletest-prefix/src/googletest-build)
set(_gtest_byproducts
  ${_gtest_byproduct_binary_dir}/lib/libgtest.a
  ${_gtest_byproduct_binary_dir}/lib/libgtest_main.a
  ${_gtest_byproduct_binary_dir}/lib/libgmock.a
  ${_gtest_byproduct_binary_dir}/lib/libgmock_main.a
  )

set(GTEST_CXX_FLAGS "${ROOT_EXTERNAL_CXX_FLAGS}")
if(MSVC)
   if(winrtdebug)
    set(GTEST_BUILD_TYPE Debug)
  else()
    set(GTEST_BUILD_TYPE Release)
  endif()
  set(_gtest_byproducts
    ${_gtest_byproduct_binary_dir}/lib/gtest.lib
    ${_gtest_byproduct_binary_dir}/lib/gtest_main.lib
    ${_gtest_byproduct_binary_dir}/lib/gmock.lib
    ${_gtest_byproduct_binary_dir}/lib/gmock_main.lib
  )
  if(CMAKE_GENERATOR MATCHES Ninja)
    set(GTEST_BUILD_COMMAND "BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR>")
  else()
    set(GTEST_BUILD_COMMAND "BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config ${GTEST_BUILD_TYPE}")
  endif()
  if(asan)
    if(NOT winrtdebug)
      set(gtestbuild "RelWithDebInfo")
    endif()
    set(GTEST_CXX_FLAGS "${ROOT_EXTERNAL_CXX_FLAGS} ${ASAN_EXTRA_CXX_FLAGS}")
  endif()
  set(EXTRA_GTEST_OPTS
    -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
    -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG:PATH=${_gtest_byproduct_binary_dir}/lib/
    -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL:PATH=${_gtest_byproduct_binary_dir}/lib/
    -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE:PATH=${_gtest_byproduct_binary_dir}/lib/
    -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO:PATH=${_gtest_byproduct_binary_dir}/lib/
    -Dgtest_force_shared_crt=ON
    ${GTEST_BUILD_COMMAND})
else()
  set(GTEST_BUILD_TYPE Release)
endif()
if(APPLE)
  set(EXTRA_GTEST_OPTS
    -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT})
endif()
set(ROOT_GOOGLETEST_VERSION 1.17.0) # https://github.com/google/googletest/releases/tag/v1.17.0 Date: Apr 30, 2025
set(ROOT_GOOGLETEST_HASH "65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c")
ExternalProject_Add(
  googletest
  # GIT_REPOSITORY https://github.com/google/googletest.git
  # GIT_SHALLOW 1
  # GIT_TAG v1.17.0
  URL ${lcgpackages}/googletest-${ROOT_GOOGLETEST_VERSION}.tar.gz
  URL_HASH SHA256=${ROOT_GOOGLETEST_HASH}
  UPDATE_COMMAND ""
  # # Force separate output paths for debug and release builds to allow easy
  # # identification of correct lib in subsequent TARGET_LINK_LIBRARIES commands
  # CMAKE_ARGS -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG:PATH=DebugLibs
  #            -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE:PATH=ReleaseLibs
  #            -Dgtest_force_shared_crt=ON
  CMAKE_ARGS -G ${CMAKE_GENERATOR}
                -DCMAKE_BUILD_TYPE=${GTEST_BUILD_TYPE}
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                -DCMAKE_CXX_FLAGS=${GTEST_CXX_FLAGS}
                -DCMAKE_AR=${CMAKE_AR}
                -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
                ${EXTRA_GTEST_OPTS}
  # Disable install step
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS ${_gtest_byproducts}
  # Wrap download, configure and build steps in a script to log output
  LOG_DOWNLOAD ON LOG_CONFIGURE ON LOG_BUILD ON LOG_OUTPUT_ON_FAILURE ON
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
set(_G_LIBRARY_PATH ${binary_dir}/lib/)

# Use gmock_main instead of gtest_main because it initializes gtest as well.
# Note: The libraries are listed in reverse order of their dependencies.
foreach(lib gtest gtest_main gmock gmock_main)
  add_library(${lib} IMPORTED STATIC GLOBAL)
  set_target_properties(${lib} PROPERTIES
    IMPORTED_LOCATION "${_G_LIBRARY_PATH}${CMAKE_STATIC_LIBRARY_PREFIX}${lib}${CMAKE_STATIC_LIBRARY_SUFFIX}"
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
