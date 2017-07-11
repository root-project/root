############################################################################
# CMakeLists.txt file for building ROOT sql/pgsql package
############################################################################

include_directories(${SQLITE_INCLUDE_DIR})

ROOT_GENERATE_DICTIONARY(G__RSQLite *.h MODULE RSQLite LINKDEF LinkDef.h DEPENDENCIES Net RIO)

ROOT_LINKER_LIBRARY(RSQLite *.cxx G__RSQLite.cxx LIBRARIES Core ${SQLITE_LIBRARIES} DEPENDENCIES Net RIO)
ROOT_INSTALL_HEADERS()
