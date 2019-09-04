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

if(APPLE)
  # ignore
  # ld: warning: direct access in function 'unsigned long std::__1::__str_find_first_of<char, unsigned long, std::__1::char_traits<char>, 18446744073709551615ul>(char const*, unsigned long, char const*, unsigned long, unsigned long)' from file '../CMakeFiles/ClingUtils.dir/src/TClingUtils.cxx.o' to global weak symbol 'std::__1::char_traits<char>::eq(char, char)' from file 'CMakeFiles/coreclingutilstestUnit.dir/TClingUtilsTests.cxx.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
  # ld: warning: direct access in function 'unsigned long std::__1::__str_find_first_of<char, unsigned long, std::__1::char_traits<char>, 18446744073709551615ul>(char const*, unsigned long, char const*, unsigned long, unsigned long)' from file '../../../interpreter/llvm/src/lib/libLLVMProfileData.a(InstrProf.cpp.o)' to global weak symbol 'std::__1::char_traits<char>::eq(char, char)' from file 'CMakeFiles/coreclingutilstestUnit.dir/TClingUtilsTests.cxx.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -w")
endif()

ROOT_ADD_UNITTEST_DIR(Core RIO ${CLING_LIBRARIES} $<TARGET_OBJECTS:ClingUtils>)
