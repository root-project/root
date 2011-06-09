############################################################################
# CMakeLists.txt file for building ROOT io/rfio package
############################################################################

ROOT_USE_PACKAGE(io/io)
ROOT_USE_PACKAGE(net/net)
include_directories(${CASTOR_INCLUDE_DIR})

ROOT_GENERATE_DICTIONARY(G__RFIO *.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(RFIO LINKDEF LinkDef.h DEPENDENCIES Net RIO)
ROOT_LINKER_LIBRARY(RFIO *.cxx G__RFIO.cxx LIBRARIES Cint Core ${CASTOR_LIBRARIES} DEPENDENCIES Net RIO)
ROOT_INSTALL_HEADERS()
