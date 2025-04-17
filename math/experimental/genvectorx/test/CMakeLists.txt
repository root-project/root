# Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# @author Danilo Piparo CERN

ROOT_EXECUTABLE(rotationApplication rotationApplication.cxx LIBRARIES GenVector)
ROOT_ADD_TEST(test-genvector-rotationApplication COMMAND rotationApplication)

ROOT_EXECUTABLE(boost testBoost.cxx LIBRARIES GenVector)
ROOT_ADD_TEST(test-genvector-boost COMMAND boost)

ROOT_EXECUTABLE(testGenvector testGenVector.cxx LIBRARIES GenVector Hist Smatrix)
ROOT_ADD_TEST(test-genvector-genvector COMMAND testGenvector)

ROOT_EXECUTABLE(testVectorIO testVectorIO.cxx LIBRARIES GenVector Tree Hist Gpad Physics)
ROOT_ADD_TEST(test-genvector-vectorio COMMAND testVectorIO
                                      COPY_TO_BUILDDIR ${CMAKE_CURRENT_SOURCE_DIR}/Track.h)

ROOT_EXECUTABLE(coordinates3D coordinates3D.cxx LIBRARIES GenVector)
ROOT_ADD_TEST(test-genvector-coordinates3D COMMAND coordinates3D)

ROOT_EXECUTABLE(coordinates4D coordinates4D.cxx LIBRARIES GenVector)
ROOT_ADD_TEST(test-genvector-coordinates4D COMMAND coordinates4D)

if (sycl)
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
endif()
