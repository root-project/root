############################################################################
# CMakeLists.txt file for building ROOT sql/oracle package
############################################################################

ROOT_USE_PACKAGE(io/io)
ROOT_USE_PACKAGE(net/net)

include_directories(${ORACLE_INCLUDE_DIR})

ROOT_GENERATE_DICTIONARY(G__Oracle *.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(Oracle LINKDEF LinkDef.h DEPENDENCIES Net RIO)
ROOT_LINKER_LIBRARY(Oracle *.cxx G__Oracle.cxx LIBRARIES Core ${ORACLE_LIBRARIES} DEPENDENCIES Net RIO)
ROOT_INSTALL_HEADERS()
