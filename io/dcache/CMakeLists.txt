############################################################################
# CMakeLists.txt file for building ROOT io/dcache package
############################################################################

include_directories(SYSTEM ${DCAP_INCLUDE_DIRS})

ROOT_GENERATE_DICTIONARY(G__DCache *.h MODULE DCache LINKDEF LinkDef.h DEPENDENCIES Net RIO)

ROOT_LINKER_LIBRARY(DCache *.cxx G__DCache.cxx LIBRARIES Core ${DCAP_LIBRARIES} DEPENDENCIES Net RIO)
ROOT_INSTALL_HEADERS()
