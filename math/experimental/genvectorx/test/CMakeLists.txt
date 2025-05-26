# Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# @author Danilo Piparo CERN

    set(CMAKE_CXX_COMPILER ${CMAKE_BINARY_DIR}/interpreter/llvm-project/llvm/tools/AdaptiveCpp/bin/acpp)

    ROOT_EXECUTABLE(testGenvectorSYCL testGenVectorSYCL.cxx) # LIBRARIES GenVectorSYCL)

    target_link_libraries(testGenvectorSYCL PUBLIC GenVectorSYCL stdc++)
    add_sycl_to_target(TARGET testGenvectorSYCL SOURCES testGenVectorSYCL.cxx)
    target_link_libraries(testGenvectorSYCL INTERFACE GenVectorSYCL acpp-rt)
    target_compile_options(testGenvectorSYCL PUBLIC "--acpp-clang=${CMAKE_BINARY_DIR}/interpreter/llvm-project/llvm/bin/clang++")
    target_link_options(testGenvectorSYCL PUBLIC "--acpp-clang=${CMAKE_BINARY_DIR}/interpreter/llvm-project/llvm/bin/clang++")

    target_compile_options(testGenvectorSYCL PUBLIC ${SYCL_COMPILER_FLAGS})
    ROOT_ADD_TEST(test-genvector-genvectorsycl COMMAND testGenvectorSYCL)

