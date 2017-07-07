############################################################################
# CMakeLists.txt file for building ROOT graf2d/gviz package
# @author Pere Mato, CERN
############################################################################

include_directories(${GRAPHVIZ_INCLUDE_DIR}/graphviz)

ROOT_GENERATE_DICTIONARY(G__Gviz *.h MODULE Gviz LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM" DEPENDENCIES Hist Graf)

ROOT_LINKER_LIBRARY(Gviz *.cxx G__Gviz.cxx
                    LIBRARIES ${GRAPHVIZ_LIBRARIES}
                    DEPENDENCIES Hist Graf)
ROOT_INSTALL_HEADERS()
