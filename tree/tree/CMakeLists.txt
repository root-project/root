############################################################################
# CMakeLists.txt file for building ROOT tree/tree package
# @author Pere Mato, CERN
############################################################################

set(libname Tree)

include_directories(SYSTEM ${TBB_INCLUDE_DIRS})

ROOT_GENERATE_DICTIONARY(G__${libname} *.h MODULE ${libname} LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM" DEPENDENCIES Net RIO Thread Imt)
#ROOT_GENERATE_DICTIONARY(ManualTree TTree.h  LINKDEF LinkDef2.h)

ROOT_LINKER_LIBRARY(${libname} *.cxx G__${libname}.cxx LIBRARIES ${TBB_LIBRARIES} DEPENDENCIES Net RIO Thread Imt)
ROOT_INSTALL_HEADERS()

