# Build Google Benchmark as an ExternalProject, mirroring GoogleTest.cmake,
# so the perf-as-validity tests get the library without a system dependency.
# BUILD_BYPRODUCTS, the explicit -B in CONFIGURE_COMMAND, and IMPORTED_LOCATION
# (resolved via ExternalProject_Get_Property binary_dir) must agree on where
# benchmark builds; tracking CMAKE_CURRENT_BINARY_DIR keeps them in sync.
set(_benchmark_byproduct_binary_dir
  ${CMAKE_CURRENT_BINARY_DIR}/googlebenchmark-prefix/src/googlebenchmark-build)
set(_benchmark_byproducts
  ${_benchmark_byproduct_binary_dir}/src/${CMAKE_STATIC_LIBRARY_PREFIX}benchmark${CMAKE_STATIC_LIBRARY_SUFFIX}
  )

if(APPLE)
  set(EXTRA_BENCHMARK_OPTS -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT})
endif()

include(ExternalProject)
# Forward parent CMAKE_CXX_FLAGS to the sub-build so sanitizer and
# -stdlib=libc++ additions don't get dropped (else benchmark builds against
# system defaults and ABI-clashes with the parent at link). Benchmark is
# third-party and not warning-clean under LLVM's -Werror/-pedantic regime
# (e.g. clang 22's -Wc2y-extensions fires on the library's __COUNTER__ use),
# so a trailing -w keeps a new upstream/compiler warning from turning the
# dependency build red.
set(GOOGLEBENCHMARK_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
if(MSVC)
  set(GOOGLEBENCHMARK_CMAKE_CXX_FLAGS "${GOOGLEBENCHMARK_CMAKE_CXX_FLAGS} /w")
else()
  set(GOOGLEBENCHMARK_CMAKE_CXX_FLAGS "${GOOGLEBENCHMARK_CMAKE_CXX_FLAGS} -w")
endif()

ExternalProject_Add(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_SHALLOW FALSE
  GIT_TAG v1.9.5
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR}
                -S ${CMAKE_CURRENT_BINARY_DIR}/googlebenchmark-prefix/src/googlebenchmark/
                -B ${_benchmark_byproduct_binary_dir}/
                -DCMAKE_BUILD_TYPE=$<CONFIG>
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                -DCMAKE_CXX_FLAGS=${GOOGLEBENCHMARK_CMAKE_CXX_FLAGS}
                -DCMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}
                -DCMAKE_SHARED_LINKER_FLAGS=${CMAKE_SHARED_LINKER_FLAGS}
                -DCMAKE_AR=${CMAKE_AR}
                # Benchmark's own tests need a separate gtest checkout and turn
                # its build red under -Werror on newer compilers; we only want
                # the library.
                -DBENCHMARK_ENABLE_TESTING=OFF
                -DBENCHMARK_ENABLE_GTEST_TESTS=OFF
                -DBENCHMARK_ENABLE_WERROR=OFF
                -DBENCHMARK_ENABLE_INSTALL=OFF
                ${EXTRA_BENCHMARK_OPTS}
  BUILD_COMMAND ${CMAKE_COMMAND} --build ${_benchmark_byproduct_binary_dir}/ --config $<CONFIG>
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS ${_benchmark_byproducts}
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  TIMEOUT 600
  )

ExternalProject_Get_Property(googlebenchmark source_dir)
set(BENCHMARK_INCLUDE_DIR ${source_dir}/include)
# Prevents bug https://gitlab.kitware.com/cmake/cmake/issues/15052
file(MAKE_DIRECTORY ${BENCHMARK_INCLUDE_DIR})

add_library(benchmark IMPORTED STATIC GLOBAL)
set_target_properties(benchmark PROPERTIES
  IMPORTED_LOCATION "${_benchmark_byproducts}"
  INTERFACE_INCLUDE_DIRECTORIES "${BENCHMARK_INCLUDE_DIR}"
  )
add_dependencies(benchmark googlebenchmark)
