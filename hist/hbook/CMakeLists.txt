############################################################################
# CMakeLists.txt file for building ROOT hist/hbook package
############################################################################

set(libname Hbook)

ROOT_GENERATE_DICTIONARY(G__${libname} *.h  MODULE ${libname} LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM" DEPENDENCIES Hist Matrix Tree Graf TreePlayer RIO)

ROOT_LINKER_LIBRARY(${libname} *.cxx G__${libname}.cxx DEPENDENCIES Hist Matrix Tree Graf TreePlayer RIO minicern )
ROOT_INSTALL_HEADERS()

