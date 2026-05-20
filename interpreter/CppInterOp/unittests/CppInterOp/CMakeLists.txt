if (EMSCRIPTEN)
  # So we create a html file, as well as the javascript file
  set(CMAKE_EXECUTABLE_SUFFIX ".html")
  # Omitting CUDATest.cpp since Emscripten build currently has no GPU support
  # For Emscripten builds linking to gtest_main will not suffice for gtest to run
  # the tests and an explicitly main.cpp is needed
  set(EXTRA_TEST_SOURCE_FILES main.cpp)

  string(REPLACE "@" "@@" ESCAPED_SYSROOT_PATH "${SYSROOT_PATH}")
  # Explanation of Emscripten-specific link flags for CppInterOpTests:
  #
  # MAIN_MODULE=1:
  # Enables building CppInterOpTests.js as the main WebAssembly module, allowing dynamic linking of side modules.
  #
  # WASM_BIGINT:
  # Ensures support for 64-bit integer types by enabling JavaScript BigInt integration in WASM.
  #
  # ALLOW_MEMORY_GROWTH=1:
  # Allows the WebAssembly memory to grow dynamically at runtime to accommodate increasing memory needs.
  # Would lead to an abortOnCannotGrowMemory error if memory cannot be grown while running the tests.
  #
  # STACK_SIZE=32mb: Allocates 32MB of stack space to handle deep recursion or large stack-allocated objects safely.
  # INITIAL_MEMORY=128mb: Sets the initial linear memory size to 128MB to reduce the likelihood of early memory expansion and improve performance.
  # The STACK_SIZE and INITIAL_MEMORY values are chosen based on what has been put to use for running xeus-cpp-lite.
  # Check https://github.com/jupyter-xeus/xeus/blob/main/cmake/WasmBuildOptions.cmake#L35-L36 for more details.
  # Not setting these flags would lead to a memory access out of bounds error while running the tests.
  #
  # --preload-file ${SYSROOT_PATH}/include@/include:
  # Preloads the system include directory into the Emscripten virtual filesystem to make headers accessible at runtime.
  #
  # --emrun
  # Makes it so that we run the html file created by this target, that we can capture the standard output
  # and output to the terminal

  # Common compile flags for WASM tests
  set(CPPINTEROP_COMMON_WASM_TEST_COMPILE_FLAGS
    "SHELL:${CPPINTEROP_EXTRA_WASM_FLAGS}"
  )

  # Common link flags for WASM tests
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
else()
  # Do not need main.cpp for native builds, but we do have GPU support for native builds
  set(EXTRA_TEST_SOURCE_FILES CUDATest.cpp)
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
  ${EXTRA_TEST_SOURCE_FILES}
)

if(NOT WIN32)
  set_source_files_properties(VariableReflectionTest.cpp PROPERTIES COMPILE_FLAGS
    "-Wno-pedantic"
)
endif ()

set_source_files_properties(InterpreterTest.cpp PROPERTIES COMPILE_DEFINITIONS
  "LLVM_BINARY_DIR=\"${LLVM_BINARY_DIR}\""
)

if(EMSCRIPTEN)
  target_compile_options(CppInterOpTests
    PUBLIC ${CPPINTEROP_COMMON_WASM_TEST_COMPILE_FLAGS}
  )
  target_link_options(CppInterOpTests
    PUBLIC ${CPPINTEROP_COMMON_WASM_TEST_LINK_FLAGS}
  )
endif()

unset(LLVM_LINK_COMPONENTS)

if (NOT EMSCRIPTEN)
  set(EXTRA_TEST_SOURCE_FILES "")
endif()

add_cppinterop_unittest(DynamicLibraryManagerTests 
  DynamicLibraryManagerTest.cpp 
  Utils.cpp 
  ${EXTRA_TEST_SOURCE_FILES}
)

if(EMSCRIPTEN)
  set(TEST_SHARED_LIBRARY_PATH "${CMAKE_CURRENT_BINARY_DIR}/bin/$<CONFIG>/")
  string(REPLACE "@" "@@" ESCAPED_TEST_SHARED_LIBRARY_PATH "${TEST_SHARED_LIBRARY_PATH}")
  target_compile_options(DynamicLibraryManagerTests
    PUBLIC ${CPPINTEROP_COMMON_WASM_TEST_COMPILE_FLAGS}
  )
  target_link_options(DynamicLibraryManagerTests
    PUBLIC
      ${CPPINTEROP_COMMON_WASM_TEST_LINK_FLAGS}
      "SHELL:--preload-file ${ESCAPED_TEST_SHARED_LIBRARY_PATH}/libTestSharedLib.so@/libTestSharedLib.so"
  )
endif()

add_dependencies(DynamicLibraryManagerTests TestSharedLib)

#export_executable_symbols_for_plugins(TestSharedLib)
add_subdirectory(TestSharedLib)

