############################################################################
# CMakeLists.txt file for building ROOT graf2d/gviz package
# @author Pere Mato, CERN
############################################################################


ROOT_USE_PACKAGE(hist/hist)
ROOT_USE_PACKAGE(graf2d/graf)

include_directories(${GRAPHVIZ_INCLUDE_DIR}/graphviz)

ROOT_GENERATE_DICTIONARY(G__Gviz *.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(Gviz LINKDEF LinkDef.h DEPENDENCIES Hist Graf)
ROOT_LINKER_LIBRARY(Gviz *.cxx G__Gviz.cxx 
                    LIBRARIES ${GRAPHVIZ_GRAPH_LIBRARY} ${GRAPHVIZ_GVC_LIBRARY} 
                    DEPENDENCIES Hist Graf)
ROOT_INSTALL_HEADERS()
