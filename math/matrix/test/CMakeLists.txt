#Copyright(C) 1995 - 2023, Rene Brun and Fons Rademakers.
#All rights reserved.
#
#For the licensing terms see $ROOTSYS / LICENSE.
#For the list of contributors see $ROOTSYS / README / CREDITS.

ROOT_ADD_GTEST(testMatrixTSparse testMatrixTSparse.cxx LIBRARIES Matrix)
# Disable FMA on RISC-V to avoid 1-ULP differences between sparse and dense paths
if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|riscv")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(testMatrixTSparse PRIVATE -ffp-contract=off)
  endif()
endif()
ROOT_ADD_GTEST(testMatrixTDecomp testMatrixTDecomp.cxx LIBRARIES Matrix)
ROOT_ADD_GTEST(testMatrixT testMatrixT.cxx LIBRARIES Matrix)