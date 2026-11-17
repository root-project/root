# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT ntuple extra utils package
# @author Jakob Blomer CERN
############################################################################

set(NTUPLEUTIL_HEADERS
  ROOT/RNTupleImporter.hxx
  ROOT/RNTupleExporter.hxx
  ROOT/RNTupleInspector.hxx
)

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTNTupleUtil
HEADERS
  ${NTUPLEUTIL_HEADERS}
SOURCES
  src/RNTupleImporter.cxx
  src/RNTupleExporter.cxx
  src/RNTupleInspector.cxx
LINKDEF
  LinkDef.h
DEPENDENCIES
  ROOTNTuple
  Tree
  Hist
)

set(RELATIVE_INC_HEADERS ${NTUPLEUTIL_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(ROOTNTupleUtil
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)

if(MSVC)
  target_compile_definitions(ROOTNTupleUtil PRIVATE _USE_MATH_DEFINES)
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
