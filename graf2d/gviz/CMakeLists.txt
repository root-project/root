############################################################################
# CMakeLists.txt file for building ROOT graf2d/gviz package
# @author Pere Mato, CERN
############################################################################


ROOT_USE_PACKAGE(hist/hist)
ROOT_USE_PACKAGE(graf2d/graf)

include_directories(${GRAPHVIZ_INCLUDE_DIR}/graphviz)

ROOT_GENERATE_DICTIONARY(G__Gviz *.h MODULE Gviz LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM")

ROOT_LINKER_LIBRARY(Gviz *.cxx G__Gviz.cxx
                    LIBRARIES ${GRAPHVIZ_LIBRARIES}
                    DEPENDENCIES Hist Graf)
ROOT_INSTALL_HEADERS()
