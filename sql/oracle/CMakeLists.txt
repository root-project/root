############################################################################
# CMakeLists.txt file for building ROOT sql/oracle package
############################################################################

ROOT_USE_PACKAGE(io/io)
ROOT_USE_PACKAGE(net/net)

include_directories(${ORACLE_INCLUDE_DIR})
add_definitions(${ORACLE_DEFINITIONS})

ROOT_GENERATE_DICTIONARY(G__Oracle *.h MODULE Oracle LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(Oracle *.cxx G__Oracle.cxx LIBRARIES Core ${ORACLE_LIBRARIES} DEPENDENCIES Net RIO)
ROOT_INSTALL_HEADERS()
