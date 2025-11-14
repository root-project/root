# Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# @author Monica Dessole CERN


    ROOT_EXECUTABLE(testGenvectorSYCL testGenVectorSYCL.cxx) # LIBRARIES GenVectorSYCL)
    target_link_libraries(testGenvectorSYCL PUBLIC GenVectorSYCL )
    add_sycl_to_root_target(TARGET testGenvectorSYCL 
    SOURCES testGenVectorSYCL.cxx 
    COMPILE_DEFINITIONS ROOT_MATH_SYCL
    DEPENDENCIES GenVectorSYCL
    )


    ROOT_EXECUTABLE(testGenvectorSYCLInvMasses testGenVectorSYCLInvMasses.cxx) # LIBRARIES GenVectorSYCL)
    target_link_libraries(testGenvectorSYCLInvMasses PUBLIC GenVectorSYCL )
    add_sycl_to_root_target(TARGET testGenvectorSYCLInvMasses 
    SOURCES testGenvectorSYCLInvMasses.cxx 
    COMPILE_DEFINITIONS ROOT_MATH_SYCL
    DEPENDENCIES GenVectorSYCL
    )

    ROOT_ADD_TEST(test-genvector-genvectorsycl COMMAND testGenvectorSYCL)
    ROOT_ADD_TEST(test-genvector-genvectorsyclinvmasses COMMAND testGenvectorSYCLInvMasses)

