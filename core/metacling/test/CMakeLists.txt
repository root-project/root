# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# Make llvm and clang available here as we are mostly testing methods directly
# depending on them.
include_directories(SYSTEM
  ${CLANG_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${CLAD_INCLUDE_DIRS}
)

include_directories(
  ../res
  ../../clingutils/res
  ../../foundation/res
  ${CLING_INCLUDE_DIRS}
  )

set(LLVM_DEPS LLVMSupport)
if(NOT builtin_clang)

  # FIXME: Passing only LLVMSupport when we are using an external clang does not
  # expand the transitive dependencies. For instance LLVMSupport should expand
  # to LLVMSupport LLVMDemangle and optionally curses.
  # For a unknown reasons using clingInterpreter enables such expansion...
  set(LLVM_DEPS clingInterpreter)
  link_directories("${LLVM_LIBRARY_DIR}")
endif()

ROOT_ADD_UNITTEST_DIR(Core RIO ${LLVM_DEPS})

target_link_libraries(coremetaclingtestUnit ROOTUnitTestSupport)
