# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

add_library(cppyy_backend STATIC
    clingwrapper/src/clingwrapper.cxx
    clingwrapper/src/cppinterop_dispatch.cxx)

target_include_directories(cppyy_backend PRIVATE
    ${CMAKE_BINARY_DIR}/etc/cppinterop
)

target_compile_definitions(cppyy_backend PRIVATE
  CPPINTEROP_DIR="${CMAKE_BINARY_DIR}/etc/cppinterop"
  CMAKE_SHARED_LIBRARY_SUFFIX="${CMAKE_SHARED_LIBRARY_SUFFIX}"
  # The CppInterOp JitCall debug assertions (AreArgumentsValid,
  # ReportInvokeStart) use Clang AST internals and cannot be resolved
  # through the dispatch mechanism. Disable them in this translation unit.
  NDEBUG
)

target_link_libraries(cppyy_backend Core)
# Ensure CppInterOpEtc responsible for staging the tablegen-generated .inc files is built before cppyy_backend. 
add_dependencies(cppyy_backend CppInterOpEtc)

if(NOT MSVC)
  target_compile_options(cppyy_backend PRIVATE -fPIC)
endif()
