############################################################################
# CMakeLists.txt file for building ROOT io/dcache package
############################################################################

ROOT_USE_PACKAGE(io/io)
ROOT_USE_PACKAGE(net/net)
include_directories(${DCAP_INCLUDE_DIRS})
add_definitions(-pthread)

# Workaround for useless const qualifier in dcap
if(${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-qualifiers")
endif()

ROOT_GENERATE_DICTIONARY(G__DCache *.h MODULE DCache LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(DCache *.cxx G__DCache.cxx LIBRARIES Core ${DCAP_LIBRARIES} DEPENDENCIES Net RIO)
ROOT_INSTALL_HEADERS()
