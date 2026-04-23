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
target_include_directories(testVectorIO PRIVATE ./)
ROOT_ADD_TEST(test-genvector-vectorio COMMAND testVectorIO
                                      COPY_TO_BUILDDIR ${CMAKE_CURRENT_SOURCE_DIR}/Track.h
                                      FIXTURES_SETUP vectoriofixture)

ROOT_GENERATE_DICTIONARY(GVTrackDict Track.h LINKDEF TrackLinkDef.h OPTIONS -inlineInputHeader)
ROOT_EXECUTABLE(testVectorIOdict testVectorIO.cxx GVTrackDict.cxx LIBRARIES GenVector Tree Hist Gpad Physics)
REFLEX_GENERATE_DICTIONARY(GVTrackDictXml Track.h SELECTION TrackDict.xml)
ROOT_EXECUTABLE(testVectorIOdictXml testVectorIO.cxx GVTrackDictXml.cxx LIBRARIES GenVector Tree Hist Gpad Physics)
target_include_directories(testVectorIOdict PRIVATE ./)
target_include_directories(testVectorIOdictXml PRIVATE ./)
target_compile_definitions(testVectorIOdictXml PRIVATE USE_REFLEX)
ROOT_ADD_TEST(test-genvector-vectoriodict COMMAND testVectorIOdict
                                      FIXTURES_REQUIRED vectoriofixture FIXTURES_SETUP vectoriodictrun)
ROOT_ADD_TEST(test-genvector-vectoriodictxml COMMAND testVectorIOdictXml
                                      FIXTURES_REQUIRED vectoriodictrun)

ROOT_EXECUTABLE(coordinates3D coordinates3D.cxx LIBRARIES GenVector)
ROOT_ADD_TEST(test-genvector-coordinates3D COMMAND coordinates3D)

ROOT_EXECUTABLE(coordinates4D coordinates4D.cxx LIBRARIES GenVector)
ROOT_ADD_TEST(test-genvector-coordinates4D COMMAND coordinates4D)

add_library(staticTests OBJECT staticTests.cxx)
target_link_libraries(staticTests PRIVATE GenVector)
