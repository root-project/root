if (EMSCRIPTEN)
  # So we create a html file, as well as the javascript file
  set(CMAKE_EXECUTABLE_SUFFIX ".html")
  # Omitting CUDATest.cpp since Emscripten build currently has no GPU support
  # For Emscripten builds linking to gtest_main will not suffice for gtest to run
  # the tests and an explicitly main.cpp is needed
  set(EXTRA_TEST_SOURCE_FILES main.cpp)
else()
  # Do not need main.cpp for native builds, but we do have GPU support for native builds
  set(EXTRA_TEST_SOURCE_FILES CUDATest.cpp)
  set(EXTRA_PATH_TEST_BINARIES /CppInterOpTests/unittests/bin/$<CONFIG>/)
endif()

add_cppinterop_unittest(CppInterOpTests
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

if(EMSCRIPTEN)
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
  target_link_options(CppInterOpTests
    PUBLIC "SHELL: -fexceptions"
    PUBLIC "SHELL: -s MAIN_MODULE=1"
    PUBLIC "SHELL: -s WASM_BIGINT"
    PUBLIC "SHELL: -s ALLOW_MEMORY_GROWTH=1"
    PUBLIC "SHELL: -s STACK_SIZE=32mb"
    PUBLIC "SHELL: -s INITIAL_MEMORY=128mb"
    PUBLIC "SHELL: --preload-file ${ESCAPED_SYSROOT_PATH}/include@/include"
    PUBLIC "SHELL: --emrun"
  )
endif()

target_link_libraries(CppInterOpTests
  PRIVATE
  clangCppInterOp
)

set_output_directory(CppInterOpTests
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${EXTRA_PATH_TEST_BINARIES}
  LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${EXTRA_PATH_TEST_BINARIES}
)

if(NOT WIN32)
  set_source_files_properties(VariableReflectionTest.cpp PROPERTIES COMPILE_FLAGS
    "-Wno-pedantic"
)
endif()

set_source_files_properties(InterpreterTest.cpp PROPERTIES COMPILE_DEFINITIONS
  "LLVM_BINARY_DIR=\"${LLVM_BINARY_DIR}\""
)
export_executable_symbols(CppInterOpTests)

unset(LLVM_LINK_COMPONENTS)

if (NOT EMSCRIPTEN)
  set(EXTRA_TEST_SOURCE_FILES "")
  set(EXTRA_PATH_TEST_BINARIES /TestSharedLib/unittests/bin/$<CONFIG>/)
endif()

add_cppinterop_unittest(DynamicLibraryManagerTests DynamicLibraryManagerTest.cpp ${EXTRA_TEST_SOURCE_FILES})

target_link_libraries(DynamicLibraryManagerTests
  PRIVATE
  clangCppInterOp
)

if(EMSCRIPTEN)
  set(TEST_SHARED_LIBRARY_PATH "${CMAKE_CURRENT_BINARY_DIR}/TestSharedLib/unittests/bin/Release/")
  string(REPLACE "@" "@@" ESCAPED_TEST_SHARED_LIBRARY_PATH "${TEST_SHARED_LIBRARY_PATH}")
  # Check explanation of Emscripten-specific link flags for CppInterOpTests above for DynamicLibraryManagerTests as well.
  target_link_options(DynamicLibraryManagerTests
    PUBLIC "SHELL: -s MAIN_MODULE=1"
    PUBLIC "SHELL: -s WASM_BIGINT"
    PUBLIC "SHELL: -s ALLOW_MEMORY_GROWTH=1"
    PUBLIC "SHELL: -s STACK_SIZE=32mb"
    PUBLIC "SHELL: -s INITIAL_MEMORY=128mb"
    PUBLIC "SHELL: --preload-file ${ESCAPED_SYSROOT_PATH}/include@/include"
    PUBLIC "SHELL: --emrun"
    PUBLIC "SHELL: --preload-file ${ESCAPED_TEST_SHARED_LIBRARY_PATH}/libTestSharedLib.so@/libTestSharedLib.so"
  )
endif()

if (EMSCRIPTEN)
  if (BUILD_SHARED_LIBS)
    target_compile_definitions(CppInterOpTests PRIVATE "EMSCRIPTEN_SHARED_LIBRARY")
    target_compile_definitions(DynamicLibraryManagerTests PRIVATE "EMSCRIPTEN_SHARED_LIBRARY")
  else()
    target_compile_definitions(CppInterOpTests PRIVATE "EMSCRIPTEN_STATIC_LIBRARY")
    target_compile_definitions(DynamicLibraryManagerTests PRIVATE "EMSCRIPTEN_STATIC_LIBRARY")
  endif()
endif()

set_output_directory(DynamicLibraryManagerTests
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${EXTRA_PATH_TEST_BINARIES}
  LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${EXTRA_PATH_TEST_BINARIES}
)

add_dependencies(DynamicLibraryManagerTests TestSharedLib)

#export_executable_symbols_for_plugins(TestSharedLib)
add_subdirectory(TestSharedLib)
