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

add_library(staticTests OBJECT staticTests.cxx)
target_link_libraries(staticTests PRIVATE GenVector)
