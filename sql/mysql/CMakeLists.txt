############################################################################
# CMakeLists.txt file for building ROOT sql/mysql package
############################################################################

include_directories(${MYSQL_INCLUDE_DIR})

ROOT_GENERATE_DICTIONARY(G__RMySQL *.h MODULE RMySQL LINKDEF LinkDef.h DEPENDENCIES Net RIO)

ROOT_LINKER_LIBRARY(RMySQL *.cxx G__RMySQL.cxx LIBRARIES Core ${MYSQL_LIBRARIES} DEPENDENCIES Net RIO)
ROOT_INSTALL_HEADERS()
