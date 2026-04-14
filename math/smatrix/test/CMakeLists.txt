# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# @author Danilo Piparo CERN

project(smatrix-tests)

set(Libraries Core RIO Tree Rint Matrix MathCore Hist)

set(TestSource
    testSMatrix.cxx
    testOperations.cxx
    testKalman.cxx
    testIO.cxx
    testInversion.cxx
    stressOperations.cxx
    stressKalman.cxx)


ROOT_GENERATE_DICTIONARY(TrackDict Track.h LINKDEF TrackLinkDef.h OPTIONS -inlineInputHeader)

#---Build and add all the defined test in the list---------------
foreach(file ${TestSource})
  get_filename_component(testname ${file} NAME_WE)
  if (testname STREQUAL "testIO")
    ROOT_EXECUTABLE(${testname} ${file} TrackDict.cxx LIBRARIES ${Libraries})
  else()
    ROOT_EXECUTABLE(${testname} ${file} LIBRARIES ${Libraries})
  endif()
  target_include_directories(${testname} PRIVATE ./)
  if (DEFINED ENV{CLHEPBASE})
    target_include_directories(${testname} PRIVATE $ENV{CLHEPBASE})
    target_compile_definitions(${testname} PRIVATE HAVE_CLHEP)
    if (WIN32)
      target_link_libraries(${testname} PRIVATE "$ENV{CLHEPBASE}/lib/CLHEP.lib")
    else()
      target_link_libraries(${testname} PRIVATE "CLHEP" "m")
      target_link_directories(${testname} PRIVATE "$ENV{CLHEPBASE}/lib")
    endif()
  endif()
  ROOT_ADD_TEST(smatrix-${testname} COMMAND ${testname} LABELS ${${testname}_LABELS})
endforeach()
