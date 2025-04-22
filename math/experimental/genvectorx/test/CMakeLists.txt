# Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# @author Danilo Piparo CERN


    ROOT_EXECUTABLE(testGenvectorSYCL testGenVectorSYCL.cxx) # LIBRARIES GenVectorSYCL)
    if (oneapi)
        target_link_options(testGenvectorSYCL PUBLIC -E)
    endif()
    target_link_libraries(testGenvectorSYCL PUBLIC GenVectorSYCL )
    #target_include_directories(testGenvectorSYCL PUBLIC $<TARGET_PROPERTY:GenVectorSYCL,INCLUDE_DIRECTORIES>)
    #add_custom_target(genexdebug COMMAND ${CMAKE_COMMAND} -E echo "GenVectorSYCL inc dirs $<TARGET_PROPERTY:GenVectorSYCL,INCLUDE_DIRECTORIES>")
    add_sycl_to_root_target(TARGET testGenvectorSYCL 
    SOURCES testGenVectorSYCL.cxx 
    COMPILE_DEFINITIONS ROOT_MATH_SYCL
    DEPENDENCIES GenVectorSYCL
    )
    ROOT_ADD_TEST(test-genvector-genvectorsycl COMMAND testGenvectorSYCL)

