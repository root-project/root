############################################################################
# CMakeLists.txt file for building ROOT sql/odbc package
############################################################################

ROOT_USE_PACKAGE(io/io)
ROOT_USE_PACKAGE(net/net)

include_directories(${ODBC_INCLUDE_DIR})

ROOT_GENERATE_DICTIONARY(G__ODBC *.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(RODBC LINKDEF LinkDef.h DEPENDENCIES Net RIO)
ROOT_LINKER_LIBRARY(RODBC *.cxx G__ODBC.cxx LIBRARIES Core ${ODBC_LIBRARIES} DEPENDENCIES Net RIO)
ROOT_INSTALL_HEADERS()
