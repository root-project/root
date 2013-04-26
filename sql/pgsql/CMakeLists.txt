############################################################################
# CMakeLists.txt file for building ROOT sql/pgsql package
############################################################################

ROOT_USE_PACKAGE(io/io)
ROOT_USE_PACKAGE(net/net)

include_directories(${POSTGRESQL_INCLUDE_DIR})

ROOT_GENERATE_DICTIONARY(G__PgSQL *.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(RPgSQL LINKDEF LinkDef.h DEPENDENCIES Net RIO)
ROOT_LINKER_LIBRARY(RPgSQL *.cxx G__PgSQL.cxx LIBRARIES Core ${POSTGRESQL_LIBRARIES} DEPENDENCIES Net RIO)
ROOT_INSTALL_HEADERS()
