############################################################################
# CMakeLists.txt file for building ROOT tree/tree package
# @author Pere Mato, CERN
############################################################################

set(libname Tree)

ROOT_USE_PACKAGE(io/io)
ROOT_USE_PACKAGE(net/net)
ROOT_USE_PACKAGE(math/mathcore)
ROOT_USE_PACKAGE(core/metautils)
ROOT_USE_PACKAGE(core/thread)
ROOT_USE_PACKAGE(hist/hist)

ROOT_GENERATE_DICTIONARY(G__${libname} *.h LINKDEF LinkDef.h)
ROOT_GENERATE_DICTIONARY(ManualTree TTree.h  LINKDEF LinkDef2.h)

ROOT_GENERATE_ROOTMAP(${libname} LINKDEF LinkDef.h DEPENDENCIES Net RIO Thread )

ROOT_LINKER_LIBRARY(${libname} *.cxx G__${libname}.cxx DEPENDENCIES Net RIO Thread)
ROOT_INSTALL_HEADERS()

