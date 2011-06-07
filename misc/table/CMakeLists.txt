############################################################################
# CMakeLists.txt file for building ROOT misc/table package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(tree/tree)
ROOT_USE_PACKAGE(io/io)
ROOT_USE_PACKAGE(graf3d/g3d)

ROOT_GENERATE_DICTIONARY(G__Table *.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(Table LINKDEF LinkDef.h DEPENDENCIES Tree Gpad Graf3d Graf Hist RIO MathCore)
ROOT_LINKER_LIBRARY(Table *.cxx G__Table.cxx LIBRARIES Core Cint DEPENDENCIES Tree Gpad Graf3d Graf Hist RIO MathCore)
ROOT_INSTALL_HEADERS()
