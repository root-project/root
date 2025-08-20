# This file allows users to call find_package(Cling) and pick up our targets.


# Compute the installation prefix from this LLVMConfig.cmake file location.
get_filename_component(CLING_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(CLING_INSTALL_PREFIX "${CLING_INSTALL_PREFIX}" PATH)
get_filename_component(CLING_INSTALL_PREFIX "${CLING_INSTALL_PREFIX}" PATH)
get_filename_component(CLING_INSTALL_PREFIX "${CLING_INSTALL_PREFIX}" PATH)

find_package(Clang REQUIRED CONFIG
             HINTS "/home/runner/work/root/root/build/lib/cmake/clang/")

set(CLING_EXPORTED_TARGETS "clingInterpreter;clingMetaProcessor;clingUtils")
set(CLING_CMAKE_DIR "${CLING_INSTALL_PREFIX}/lib/cmake/cling")
set(CLING_INCLUDE_DIRS "${CLING_INSTALL_PREFIX}/include")

# Provide all our library targets to users.
include("${CLING_CMAKE_DIR}/ClingTargets.cmake")

# By creating cling-tablegen-targets here, subprojects that depend on Cling's
# tablegen-generated headers can always depend on this target whether building
# in-tree with Cling or not.
if(NOT TARGET cling-tablegen-targets)
  add_custom_target(cling-tablegen-targets)
endif()
