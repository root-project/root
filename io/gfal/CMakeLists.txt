############################################################################
# CMakeLists.txt file for building ROOT io/gfal package
############################################################################

ROOT_USE_PACKAGE(io/io)
ROOT_USE_PACKAGE(net/net)
include_directories(${GFAL_INCLUDE_DIRS})

add_definitions(-D_FILE_OFFSET_BITS=64)
if(GFAL_LIBRARY MATCHES gfal2)
  add_definitions(-D_GFAL2_API_)
endif()

ROOT_GENERATE_DICTIONARY(G__GFAL *.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(GFAL LINKDEF LinkDef.h DEPENDENCIES Net RIO)
ROOT_LINKER_LIBRARY(GFAL *.cxx G__GFAL.cxx LIBRARIES Core ${GFAL_LIBRARIES} DEPENDENCIES Net RIO)
ROOT_INSTALL_HEADERS()
