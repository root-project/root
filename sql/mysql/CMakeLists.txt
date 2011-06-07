############################################################################
# CMakeLists.txt file for building ROOT sql/mysql package
############################################################################

ROOT_USE_PACKAGE(io/io)
ROOT_USE_PACKAGE(net/net)

include_directories(${MYSQL_INCLUDE_DIR})

ROOT_GENERATE_DICTIONARY(G__MySQL *.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(RMySQL LINKDEF LinkDef.h DEPENDENCIES Net RIO)
ROOT_LINKER_LIBRARY(RMySQL *.cxx G__MySQL.cxx LIBRARIES Core ${MYSQL_LIBRARIES} DEPENDENCIES Net RIO)
ROOT_INSTALL_HEADERS()
