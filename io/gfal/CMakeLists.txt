############################################################################
# CMakeLists.txt file for building ROOT io/gfal package
############################################################################

include_directories(${GFAL_INCLUDE_DIRS})

add_definitions(-D_FILE_OFFSET_BITS=64)

if(GFAL_LIBRARY MATCHES gfal2)
  add_definitions(-D_GFAL2_API_)
endif()

ROOT_GENERATE_DICTIONARY(G__GFAL *.h MODULE GFAL LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(GFAL *.cxx G__GFAL.cxx LIBRARIES Core ${GFAL_LIBRARIES} DEPENDENCIES Net RIO)
ROOT_INSTALL_HEADERS()
