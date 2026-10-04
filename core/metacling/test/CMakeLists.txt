# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

include_directories(
  ../res
  ../../clingutils/res
  ../../foundation/res
  ${CLING_INCLUDE_DIRS}
  )

if(root7)
  set(ROOT7SRC TClingDataMemberInfoTestsROOT7.cxx)
endif()

ROOT_ADD_GTEST(
  TClingTest

  TClingTests.cxx
  TClingCallFuncTests.cxx
  TClingDataMemberInfoTests.cxx
  ${ROOT7SRC}
  TClingMethodInfoTests.cxx

  LIBRARIES
  Core
  RIO
  GTest::gmock
)

ROOT_ADD_GTEST(TClingLoadUnloadFile TClingLoadUnloadFileTests.cxx LIBRARIES Core RIO)

if(pyroot)
  ROOT_ADD_PYUNITTEST(MetaClingTests MetaClingTests.py)
endif()

# libCling must keep its LLVM private: another LLVM in the process, for example
# from a graphics driver or a Python package, must neither bind to it nor have
# libCling bind to its own (https://github.com/root-project/root/issues/19889).
set(macro ${CMAKE_CURRENT_SOURCE_DIR}/LLVMIsolation.C)
if(CMAKE_EXECUTABLE_FORMAT STREQUAL "ELF" AND NOT asan)

  # The canary's symbols must exist in the LLVM and Clang ROOT was built with.
  set(llvm_archives)
  foreach(lib LLVMSupport LLVMCore clangAST)
    if(TARGET ${lib})
      list(APPEND llvm_archives $<TARGET_FILE:${lib}>)
    elseif(EXISTS ${LLVM_LIBRARY_DIR}/lib${lib}.a)
      list(APPEND llvm_archives ${LLVM_LIBRARY_DIR}/lib${lib}.a)
    endif()
  endforeach()
  list(LENGTH llvm_archives n_llvm_archives)
  if(n_llvm_archives EQUAL 3)
    string(REPLACE ";" ":" llvm_archives "${llvm_archives}")
    ROOT_ADD_TEST(test-metacling-llvm-isolation-canary-names
      COMMAND root.exe -l -b -q "${macro}(\"names\")"
      ENVIRONMENT LLVM_ISOLATION_ARCHIVES=${llvm_archives}
      PASSREGEX "LLVM isolation names: missing=0 of")
  endif()

  add_library(LLVMIsolationCanary SHARED LLVMIsolationCanary.cxx)
  set(canary $<TARGET_FILE:LLVMIsolationCanary>)
  set(dictionary -f ${CMAKE_CURRENT_BINARY_DIR}/G__LLVMIsolation.cxx
                 ${CMAKE_CURRENT_SOURCE_DIR}/LLVMIsolation.h ${CMAKE_CURRENT_SOURCE_DIR}/LLVMIsolationLinkDef.h)

  # The canary comes first in the global scope, before libCling is loaded.
  ROOT_ADD_TEST(test-metacling-llvm-isolation-jit-canary
    COMMAND root.exe -l -b -q ${macro}
    ENVIRONMENT LD_PRELOAD=${canary}
    PASSREGEX "LLVM isolation: JIT OK" FAILREGEX "LLVM isolation canary")
  ROOT_ADD_TEST(test-metacling-llvm-isolation-rootcling-canary
    COMMAND rootcling ${dictionary}
    ENVIRONMENT LD_PRELOAD=${canary}
    FAILREGEX "LLVM isolation canary")
  # The canary must catch a library that relies on the LLVM of its host.
  add_library(LLVMIsolationTestCALLER SHARED LLVMIsolationTestLibrary.cxx)
  target_compile_definitions(LLVMIsolationTestCALLER PRIVATE CALLER)
  ROOT_ADD_TEST(test-metacling-llvm-isolation-canary-selftest
    COMMAND root.exe -l -b -q ${macro}
    ENVIRONMENT LD_PRELOAD=${canary} LLVM_ISOLATION_LOAD=$<TARGET_FILE:LLVMIsolationTestCALLER>
    PASSREGEX "LLVM isolation canary: _ZN4llvm4errsEv resolved outside of libCling")

  # Every binding performed by the dynamic linker, including those of the
  # canary's references, once it is preloaded or loaded into the global scope
  # by the macro. LD_DEBUG=bindings is specific to glibc.
  include(CheckSymbolExists)
  check_symbol_exists(__GLIBC__ features.h ROOT_HAVE_GLIBC)
  if(ROOT_HAVE_GLIBC)
    ROOT_ADD_TEST(test-metacling-llvm-isolation-jit-bindings
      COMMAND ${CMAKE_COMMAND} -DOUT=${CMAKE_CURRENT_BINARY_DIR}/llvm-isolation-jit -DEXPECT=JIT.OK
              -P ${CMAKE_CURRENT_SOURCE_DIR}/CheckLLVMBindings.cmake
              -- $<TARGET_FILE:root.exe> -l -b -q ${macro}
      ENVIRONMENT LLVM_ISOLATION_LOAD=${canary})
    ROOT_ADD_TEST(test-metacling-llvm-isolation-rootcling-bindings
      COMMAND ${CMAKE_COMMAND} -DOUT=${CMAKE_CURRENT_BINARY_DIR}/llvm-isolation-rootcling
              -P ${CMAKE_CURRENT_SOURCE_DIR}/CheckLLVMBindings.cmake
              -- $<TARGET_FILE:rootcling> ${dictionary}
      ENVIRONMENT LD_PRELOAD=${canary})
  endif()
endif()
