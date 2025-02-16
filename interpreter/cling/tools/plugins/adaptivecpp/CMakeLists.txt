#-------------------------------------------------------------------------------
# CLING - the C++ LLVM-based InterpreterG :)
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#-------------------------------------------------------------------------------

include(ExternalProject)

add_definitions(-DADAPTIVECPP_ENABLED)

if(DEFINED CLING_ROOT_BUILD)
  set(CLANG_EXECUTABLE_PATH "${CMAKE_BINARY_DIR}/interpreter/llvm-project/llvm/bin/clang++")
  set(LLVM_DIR "${CMAKE_BINARY_DIR}/interpreter/llvm-project/llvm/lib/cmake/llvm")
else()
  set(CLANG_EXECUTABLE_PATH "${CMAKE_BINARY_DIR}/bin/clang++")
  set(LLVM_DIR "${CMAKE_BINARY_DIR}/lib/cmake/llvm")
endif()

# Paths for AdaptiveCPP
set(ADAPTIVE_CPP_INSTALL_DIR "${CMAKE_BINARY_DIR}/AdaptiveCppInstall")
set(ADAPTIVE_CPP_BINARY "${ADAPTIVE_CPP_INSTALL_DIR}/bin/acpp")

# Configuration header file for AdaptiveCPP
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/AdaptiveCppConfig.h.in
  ${CMAKE_BINARY_DIR}/tools/plugins/adaptivecpp/AdaptiveCppConfig.h
  @ONLY
)

# AdaptiveCpp CMake arguments
set(ADAPTIVE_CPP_CMAKE_ARGS
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  -DCLANG_EXECUTABLE_PATH=${CLANG_EXECUTABLE_PATH}
  -DLLVM_DIR=${LLVM_DIR}
  -DCMAKE_INSTALL_PREFIX=${ADAPTIVE_CPP_INSTALL_DIR}
)

# Wrap download, configure and build steps in a script to log output
set(ADAPTIVE_CPP_EXTRA_SETTINGS
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
)

# If ADAPTIVE_CPP_SOURCE_DIR variable is defined in the CMake configuration, we're
# skipping the download of the repository and use the passed directory.
if(DEFINED ADAPTIVE_CPP_SOURCE_DIR)
  list(APPEND ADAPTIVE_CPP_EXTRA_SETTINGS SOURCE_DIR ${ADAPTIVE_CPP_SOURCE_DIR})
else()
  list(APPEND ADAPTIVE_CPP_EXTRA_SETTINGS
    GIT_REPOSITORY https://github.com/AdaptiveCpp/AdaptiveCpp.git
    GIT_TAG v24.10.0
  )
endif()

list(APPEND ADAPTIVE_CPP_PATCHES_LIST "0001-Handle-Expected-T-from-TempFile-create-to-prevent-po.patch")
set(ADAPTIVE_CPP_PATCH_COMMAND
  ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_CURRENT_SOURCE_DIR}/patches <SOURCE_DIR>
  && git checkout <SOURCE_DIR>
  && git apply --ignore-space-change --ignore-whitespace ${ADAPTIVE_CPP_PATCHES_LIST}
)

ExternalProject_Add(
  adaptiveCpp
  UPDATE_COMMAND ""
  PATCH_COMMAND ${ADAPTIVE_CPP_PATCH_COMMAND}
  CMAKE_ARGS ${ADAPTIVE_CPP_CMAKE_ARGS}
  BUILD_COMMAND ${CMAKE_COMMAND} --build .
  INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
  BUILD_BYPRODUCTS ${ADAPTIVE_CPP_INSTALL_DIR}/bin/acpp
  ${ADAPTIVE_CPP_EXTRA_SETTINGS}
)

# Register the AdaptiveCpp binary as an imported target
add_library(adaptiveCppLibrary IMPORTED STATIC GLOBAL)
set_property(TARGET adaptiveCppLibrary PROPERTY IMPORTED_LOCATION ${ADAPTIVE_CPP_INSTALL_DIR}/lib/libAdaptiveCpp.a)

# AdaptiveCpp depends on LLVM and Clang, so ensure dependencies are built first
add_dependencies(adaptiveCpp clang llvm-config llvm-link)
