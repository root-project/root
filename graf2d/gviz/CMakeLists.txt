############################################################################
# CMakeLists.txt file for building ROOT graf2d/gviz package
# @author Pere Mato, CERN
############################################################################

include_directories(${GRAPHVIZ_INCLUDE_DIR}/graphviz)

ROOT_STANDARD_LIBRARY_PACKAGE(Gviz
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              DEPENDENCIES Hist Graf
                              LIBRARIES ${GRAPHVIZ_LIBRARIES})
