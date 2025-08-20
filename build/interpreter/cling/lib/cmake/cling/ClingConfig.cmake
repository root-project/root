# This file allows users to call find_package(Cling) and pick up our targets.



find_package(Clang REQUIRED CONFIG
             HINTS "/home/runner/work/root/root/build/lib/cmake/clang/")

set(CLING_EXPORTED_TARGETS "clingInterpreter;clingMetaProcessor;clingUtils")
set(CLING_CMAKE_DIR "/home/runner/work/root/root/build/interpreter/cling/lib/cmake/cling")
set(CLING_INCLUDE_DIRS "/home/runner/work/root/root/interpreter/cling/include;/home/runner/work/root/root/build/interpreter/cling/include")

# Provide all our library targets to users.
include("/home/runner/work/root/root/build/interpreter/cling/lib/cmake/cling/ClingTargets.cmake")

# By creating cling-tablegen-targets here, subprojects that depend on Cling's
# tablegen-generated headers can always depend on this target whether building
# in-tree with Cling or not.
if(NOT TARGET cling-tablegen-targets)
  add_custom_target(cling-tablegen-targets)
endif()
