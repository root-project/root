############################################################################
# CMakeLists.txt file for building ROOT graf2d/gviz package
# @author Pere Mato, CERN
############################################################################

include_directories(${GRAPHVIZ_INCLUDE_DIR}/graphviz)

ROOT_STANDARD_LIBRARY_PACKAGE(Gviz
  HEADERS
    TGraphEdge.h
    TGraphNode.h
    TGraphStruct.h
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
