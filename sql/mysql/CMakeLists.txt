############################################################################
# CMakeLists.txt file for building ROOT sql/mysql package
############################################################################

include_directories(${MYSQL_INCLUDE_DIR})

ROOT_GENERATE_DICTIONARY(G__MySQL *.h MODULE RMySQL LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(RMySQL *.cxx G__MySQL.cxx LIBRARIES Core ${MYSQL_LIBRARIES} DEPENDENCIES Net RIO)
ROOT_INSTALL_HEADERS()
