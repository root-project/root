############################################################################
# CMakeLists.txt file for building ROOT sql/pgsql package
############################################################################

include_directories(${POSTGRESQL_INCLUDE_DIR})

ROOT_GENERATE_DICTIONARY(G__PgSQL *.h MODULE PgSQL LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(PgSQL *.cxx G__PgSQL.cxx LIBRARIES Core ${POSTGRESQL_LIBRARIES} DEPENDENCIES Net RIO)
ROOT_INSTALL_HEADERS()
