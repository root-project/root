if (EMSCRIPTEN)
  # Build a .html shell alongside the .js main module.
  set(CMAKE_EXECUTABLE_SUFFIX ".html")

  string(REPLACE "@" "@@" ESCAPED_SYSROOT_PATH "${SYSROOT_PATH}")
  # Explanation of Emscripten-specific link flags for CppInterOp tests:
  #
  # MAIN_MODULE=1:
  # Enables the test binary as the main WebAssembly module, allowing dynamic
  # linking of side modules (TestSharedLib).
  #
  # WASM_BIGINT:
  # Enables JS BigInt integration in WASM so 64-bit integer types round-trip.
  #
  # ALLOW_MEMORY_GROWTH=1:
  # Lets the WebAssembly linear memory grow at runtime. Without it we hit
  # abortOnCannotGrowMemory during the tests.
  #
  # STACK_SIZE=32mb / INITIAL_MEMORY=64mb:
  # Chosen to match xeus-cpp-lite
  # (https://github.com/jupyter-xeus/xeus/blob/main/cmake/WasmBuildOptions.cmake#L35-L36).
  # Smaller values trigger out-of-bounds memory accesses while running the tests.
  #
  # --preload-file ${SYSROOT_PATH}/include@/include:
  # Preloads the system include directory into the Emscripten VFS so headers
  # are available at runtime.
  #
  # --emrun:
  # Makes emrun drive the .html shell and pipe stdout back to the terminal.
  set(CPPINTEROP_COMMON_WASM_TEST_COMPILE_FLAGS
    "SHELL:${CPPINTEROP_EXTRA_WASM_FLAGS}"
  )
  set(CPPINTEROP_COMMON_WASM_TEST_LINK_FLAGS
    "SHELL:${CPPINTEROP_EXTRA_WASM_FLAGS}"
    "SHELL:-s MAIN_MODULE=1"
    "SHELL:-s WASM_BIGINT"
    "SHELL:-s ALLOW_MEMORY_GROWTH=1"
    "SHELL:-s STACK_SIZE=32mb"
    "SHELL:-s INITIAL_MEMORY=64mb"
    "SHELL:--preload-file ${ESCAPED_SYSROOT_PATH}/include@/include"
    "SHELL:--emrun"
  )
endif()

add_cppinterop_unittest(CppInterOpTests
  CAPITest.cpp
  CommentReflectionTest.cpp
  CppInterOpThunksTest.cpp
  DiagnosticTest.cpp
  EnumReflectionTest.cpp
  FunctionReflectionTest.cpp
  HandleTypesTest.cpp
  InterpreterTest.cpp
  JitTest.cpp
  ResultTest.cpp
  ScopeReflectionTest.cpp
  TypeReflectionTest.cpp
  Utils.cpp
  VariableReflectionTest.cpp
)
if (NOT EMSCRIPTEN)
  # Emscripten has no GPU; CUDA tests are native-only. The VTableOverlay
  # tests need Cpp::Construct, whose placement-new JIT wrapper does not
  # compile under the Emscripten clang-repl (see FunctionReflection_Construct,
  # likewise skipped there).
  target_sources(CppInterOpTests PRIVATE CUDATest.cpp VTableOverlayTest.cpp)
endif()

if(NOT WIN32)
  set_source_files_properties(VariableReflectionTest.cpp PROPERTIES COMPILE_FLAGS
    "-Wno-pedantic"
  )
endif()

# CAPITestC.c is a pure C compilation test. Build it as a separate object
# library so it does not inherit C++-only flags (-Wno-deprecated-copy,
# -fno-rtti) that GCC's C frontend rejects under -Werror.
add_library(CAPITestC OBJECT CAPITestC.c)
target_compile_options(CAPITestC PRIVATE
  $<$<NOT:$<C_COMPILER_ID:MSVC>>:-Werror -Wall -Wno-unused-command-line-argument>
  $<$<C_COMPILER_ID:MSVC>:/WX /W3>)
target_include_directories(CAPITestC PRIVATE
  ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
if(TARGET CppInterOpTableGen)
  add_dependencies(CAPITestC CppInterOpTableGen)
endif()
target_link_libraries(CppInterOpTests PRIVATE CAPITestC)
target_compile_definitions(CppInterOpTests PRIVATE
  "CPPINTEROP_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/../../\""
)

set_source_files_properties(InterpreterTest.cpp PROPERTIES COMPILE_DEFINITIONS
  "LLVM_BINARY_DIR=\"${LLVM_BINARY_DIR}\""
)

add_cppinterop_unittest(DynamicLibraryManagerTests
  DynamicLibraryManagerTest.cpp
  Utils.cpp
)
if(EMSCRIPTEN)
  set(TEST_SHARED_LIBRARY_PATH "${CMAKE_CURRENT_BINARY_DIR}/bin/$<CONFIG>/")
  string(REPLACE "@" "@@" ESCAPED_TEST_SHARED_LIBRARY_PATH "${TEST_SHARED_LIBRARY_PATH}")
  target_link_options(DynamicLibraryManagerTests PRIVATE
    "SHELL:--preload-file ${ESCAPED_TEST_SHARED_LIBRARY_PATH}/libTestSharedLib.so@/libTestSharedLib.so"
  )
endif()

add_cppinterop_unittest(TracingTests
  TracingTests.cpp
  Utils.cpp
)
target_compile_definitions(TracingTests PRIVATE
  "CPPINTEROP_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/../../\""
  "CPPINTEROP_BINARY_DIR=\"${CMAKE_CURRENT_BINARY_DIR}/../../\""
)
# GCC 12 and (less often) 13/14 emit a -Wmaybe-uninitialized false positive
# inside libstdc++'s <regex> headers when the std::function subobject of
# std::__detail::_State is move-constructed. The analysis only fails once the
# inliner is perturbed, so sanitizer builds (which insert shadow-memory
# barriers and inline less) are where this reliably surfaces. Clang and newer
# GCCs are fine. See gcc bug 105562 and friends.
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set_source_files_properties(TracingTests.cpp
    PROPERTIES COMPILE_FLAGS "-Wno-maybe-uninitialized")
endif()

add_subdirectory(TestSharedLib)
add_dependencies(DynamicLibraryManagerTests TestSharedLib)

# Cross-TU VTableOverlay perf check, built as part of the test suite.
# Standalone-only (its Google Benchmark dependency is provisioned only then;
# keeps it out of embedded host builds such as ROOT). Excluded on WASM
# (Google Benchmark's WASM support is shaky, and the dispatch-cost claim is
# about CPU, not JS, execution). The dispatch loop is compiled into
# TestSharedLib so the call site cannot be devirtualized in this TU.
if(CPPINTEROP_BUILT_STANDALONE AND NOT EMSCRIPTEN AND BUILD_SHARED_LIBS)
  add_cppinterop_unittest(VTableOverlayCrossTUBench
    VTableOverlayCrossTUBench.cpp)
  target_link_libraries(VTableOverlayCrossTUBench PRIVATE benchmark TestSharedLib)
  add_dependencies(VTableOverlayCrossTUBench TestSharedLib)
endif()

# Downstream-DSO dlopen check that mirrors libcppyy-backend's contract:
# a SHARED lib using <CppInterOp/Dispatch.h> without linking against
# clangCppInterOp, dlopen'd by a standalone exec that also doesn't link.
# Any inline-header UND ref that the host lib can't satisfy at runtime
# trips here before the cppyy integration job.
if(NOT EMSCRIPTEN AND BUILD_SHARED_LIBS)
  add_subdirectory(TestDownstreamLib)

  set(_downstream_lib_path
    "${CMAKE_CURRENT_BINARY_DIR}/bin/$<CONFIG>/${CMAKE_SHARED_LIBRARY_PREFIX}TestDownstreamLib${CMAKE_SHARED_LIBRARY_SUFFIX}")

  add_executable(TestDownstreamLoader EXCLUDE_FROM_ALL TestDownstreamLoader.cpp)
  add_dependencies(CppInterOpUnitTests TestDownstreamLoader TestDownstreamLib)
  target_compile_definitions(TestDownstreamLoader PRIVATE
    TEST_DOWNSTREAM_LIB_PATH="${_downstream_lib_path}")
  if(UNIX AND NOT APPLE)
    target_link_libraries(TestDownstreamLoader PRIVATE dl)
  endif()
  set_output_directory(TestDownstreamLoader
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin/$<CONFIG>/
    LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin/$<CONFIG>/
  )
  add_test(NAME cppinterop-DownstreamLoaderTest
    COMMAND TestDownstreamLoader $<TARGET_FILE:clangCppInterOp>)
  set_tests_properties(cppinterop-DownstreamLoaderTest PROPERTIES
    TIMEOUT "${TIMEOUT_VALUE}")
endif()

# Dispatch Tests.
# Load libclangCppInterOp via dlopen(RTLD_LOCAL) and must NOT link against it
# directly, to verify true symbol isolation.
if(NOT EMSCRIPTEN AND BUILD_SHARED_LIBS)
  add_executable(DispatchTests EXCLUDE_FROM_ALL
    DispatchInit.cpp
    DispatchTest.cpp
    DispatchSmokeTest.cpp
  )
  add_dependencies(CppInterOpUnitTests DispatchTests)
  if(TARGET CppInterOpTableGen)
    add_dependencies(DispatchTests CppInterOpTableGen)
  endif()
  target_include_directories(DispatchTests PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR} ${GTEST_INCLUDE_DIR})
  target_link_libraries(DispatchTests PRIVATE ${gtest_libs} ${link_pthreads_lib})
  # Deliberately no target_link_libraries(... clangCppInterOp).
  target_compile_definitions(DispatchTests PRIVATE
    CPPINTEROP_LIB_PATH="${CMAKE_BINARY_DIR}/lib/libclangCppInterOp${CMAKE_SHARED_LIBRARY_SUFFIX}"
  )
  set_output_directory(DispatchTests
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin/$<CONFIG>/
    LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin/$<CONFIG>/
  )
  add_test(NAME cppinterop-DispatchTests COMMAND DispatchTests)
  set_tests_properties(cppinterop-DispatchTests PROPERTIES
    TIMEOUT "${TIMEOUT_VALUE}"
    ENVIRONMENT "CPLUS_INCLUDE_PATH=${CMAKE_BINARY_DIR}/etc")
  # Verify the binary is NOT linked against libclangCppInterOp. If this
  # check fails, the dispatch tests are not truly testing RTLD_LOCAL.
  if(UNIX AND NOT APPLE)
    add_test(NAME cppinterop-DispatchTests-isolation
      COMMAND sh -c "! ldd $<TARGET_FILE:DispatchTests> | grep -q $<TARGET_FILE_NAME:clangCppInterOp>"
    )
  elseif(APPLE)
    add_test(NAME cppinterop-DispatchTests-isolation
      COMMAND sh -c "! otool -L $<TARGET_FILE:DispatchTests> | grep -q $<TARGET_FILE_NAME:clangCppInterOp>"
    )
  endif()
endif()
