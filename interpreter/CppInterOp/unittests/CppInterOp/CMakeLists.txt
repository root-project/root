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
  ENABLE_DISPATCH
  CommentReflectionTest.cpp
  EnumReflectionTest.cpp
  FunctionReflectionTest.cpp
  InterpreterTest.cpp
  JitTest.cpp
  ScopeReflectionTest.cpp
  TypeReflectionTest.cpp
  Utils.cpp
  VariableReflectionTest.cpp
)
if (NOT EMSCRIPTEN)
  # Emscripten has no GPU; CUDA tests are native-only.
  target_sources(CppInterOpTests PRIVATE CUDATest.cpp)
endif()

if(NOT WIN32)
  set_source_files_properties(VariableReflectionTest.cpp PROPERTIES COMPILE_FLAGS
    "-Wno-pedantic"
  )
endif()

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

set_property(TEST cppinterop-TracingTests APPEND PROPERTY
  ENVIRONMENT "GTEST_DEATH_TEST_STYLE=threadsafe")

target_compile_definitions(TracingTests PRIVATE
  "CPPINTEROP_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/../../\""
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
