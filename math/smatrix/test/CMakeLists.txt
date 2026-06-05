# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# @author Danilo Piparo CERN

project(smatrix-tests)

set(Libraries Core RIO Tree Rint Matrix MathCore Hist Graf Gpad)

set(TestSource
    testSMatrix.cxx
    testOperations.cxx
    testKalman.cxx
    testIO.cxx
    testInversion.cxx
    stressOperations.cxx
    stressKalman.cxx)


ROOT_GENERATE_DICTIONARY(TrackDict Track.h LINKDEF TrackLinkDef.h OPTIONS -inlineInputHeader)
REFLEX_GENERATE_DICTIONARY(TrackDictXml Track.h SELECTION TrackDict.xml)

#---Build and add all the defined test in the list---------------
foreach(file ${TestSource})

  get_filename_component(testname ${file} NAME_WE)
  if (testname STREQUAL "testIO")
    set(variants "" "Xml")
  else()
    set(variants "")
  endif()
  foreach(variant ${variants})
    if (testname STREQUAL "testIO")
      ROOT_EXECUTABLE(${testname}${variant} ${file} TrackDict${variant}.cxx LIBRARIES ${Libraries})
      if (variant STREQUAL "Xml")
        target_compile_definitions(${testname}${variant} PRIVATE USE_REFLEX)
      endif()
    else()
      ROOT_EXECUTABLE(${testname}${variant} ${file} LIBRARIES ${Libraries})
    endif()
    target_include_directories(${testname}${variant} PRIVATE ./)
    if (DEFINED ENV{CLHEPBASE})
      target_include_directories(${testname}${variant} PRIVATE $ENV{CLHEPBASE})
      target_compile_definitions(${testname}${variant} PRIVATE HAVE_CLHEP)
      if (WIN32)
        target_link_libraries(${testname}${variant} PRIVATE "$ENV{CLHEPBASE}/lib/CLHEP.lib")
      else()
        target_link_libraries(${testname}${variant} PRIVATE "CLHEP" "m")
        target_link_directories(${testname}${variant} PRIVATE "$ENV{CLHEPBASE}/lib")
      endif()
    endif()
    ROOT_ADD_TEST(smatrix-${testname}${variant} COMMAND ${testname}${variant} LABELS ${${testname}${variant}_LABELS})
  endforeach()
endforeach()

ROOT_EXECUTABLE(testReadKalman kalman.C LIBRARIES ${Libraries})
ROOT_ADD_TEST(smatrix-testReadKalman COMMAND testReadKalman slc3_ia32_gcc323 LABELS ${testReadKalman_LABELS} COPY_TO_BUILD_DIR kalman_slc3_ia32_gcc323.root)

ROOT_EXECUTABLE(testMatrixOperations matrixOperations.C LIBRARIES ${Libraries})
ROOT_ADD_TEST(smatrix-testMatrixOperations COMMAND testMatrixOperations slc3_ia32_gcc323 LABELS ${testMatrixOperations_LABELS} COPY_TO_BUILD_DIR testOperations_slc3_ia32_gcc323.root)
