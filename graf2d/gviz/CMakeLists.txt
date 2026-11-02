# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT graf2d/gviz package
# @author Pere Mato, CERN
############################################################################

set(GVIZ_HEADERS
  TGraphEdge.h
  TGraphNode.h
  TGraphStruct.h  
)

ROOT_STANDARD_LIBRARY_PACKAGE(Gviz
  HEADERS
    ${GVIZ_HEADERS}
  SOURCES
    src/TGraphEdge.cxx
    src/TGraphNode.cxx
    src/TGraphStruct.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  LIBRARIES
    ${GRAPHVIZ_LIBRARIES}
  DEPENDENCIES
    Hist
    Graf
)

set(RELATIVE_INC_HEADERS ${GVIZ_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(Gviz
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)

if(MSVC)
  set_target_properties(Gviz PROPERTIES COMPILE_FLAGS "-DGVDLL")
endif()
target_include_directories(Gviz PRIVATE ${GRAPHVIZ_INCLUDE_DIR}/graphviz)
