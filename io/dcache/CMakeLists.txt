############################################################################
# CMakeLists.txt file for building ROOT io/dcache package
############################################################################

ROOT_USE_PACKAGE(io/io)
ROOT_USE_PACKAGE(net/net)
include_directories(${DCAP_INCLUDE_DIRS})
add_definitions(-pthread)

ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-ignored-qualifiers)

ROOT_GENERATE_DICTIONARY(G__DCache *.h MODULE DCache LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(DCache *.cxx G__DCache.cxx LIBRARIES Core ${DCAP_LIBRARIES} DEPENDENCIES Net RIO)
ROOT_INSTALL_HEADERS()
