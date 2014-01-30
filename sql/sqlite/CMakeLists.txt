############################################################################
# CMakeLists.txt file for building ROOT sql/pgsql package
############################################################################

ROOT_USE_PACKAGE(io/io)
ROOT_USE_PACKAGE(net/net)

include_directories(${SQLITE_INCLUDE_DIR})

ROOT_GENERATE_DICTIONARY(G__SQLite *.h MODULE RSQLite LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(RSQLite *.cxx G__SQLite.cxx LIBRARIES Core ${SQLITE_LIBRARIES} DEPENDENCIES Net RIO)
ROOT_INSTALL_HEADERS()
