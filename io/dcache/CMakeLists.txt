############################################################################
# CMakeLists.txt file for building ROOT io/dcache package
############################################################################

ROOT_USE_PACKAGE(io/io)
ROOT_USE_PACKAGE(net/net)
include_directories(${DCAP_INCLUDE_DIRS})
add_definitions(-pthread)

ROOT_GENERATE_DICTIONARY(G__DCache *.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(DCache LINKDEF LinkDef.h DEPENDENCIES Net RIO)
ROOT_LINKER_LIBRARY(DCache *.cxx G__DCache.cxx LIBRARIES Core ${DCAP_LIBRARIES} DEPENDENCIES Net RIO)
ROOT_INSTALL_HEADERS()
