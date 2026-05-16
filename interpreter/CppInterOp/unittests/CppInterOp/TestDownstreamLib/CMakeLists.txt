add_llvm_library(TestDownstreamLib
  SHARED
  DISABLE_LLVM_LINK_LLVM_DYLIB
  BUILDTREE_ONLY
  TestDownstreamLib.cpp)
add_dependencies(TestDownstreamLib CppInterOpTableGen)
# NDEBUG mirrors cppyy's Release build so the JitCall::Invoke asserts
# don't drag in extra UND refs. target_compile_options so the flag
# appends after add_llvm_library's `-UNDEBUG`.
target_compile_options(TestDownstreamLib PRIVATE -DNDEBUG)

set_output_directory(TestDownstreamLib
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/../bin/$<CONFIG>/
  LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/../bin/$<CONFIG>/
)

set_target_properties(TestDownstreamLib PROPERTIES FOLDER "Tests")
