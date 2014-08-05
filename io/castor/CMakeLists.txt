############################################################################
# CMakeLists.txt file for building ROOT io/castor package
############################################################################

ROOT_USE_PACKAGE(io/io)
ROOT_USE_PACKAGE(net/net)
include_directories(${CASTOR_INCLUDE_DIR})

if(CASTOR_VERSION VERSION_GREATER "2.0.0")
  add_definitions(-DR__CASTOR2)
endif()

ROOT_GENERATE_DICTIONARY(G__CASTOR *.h MODULE RCastor LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(RCastor *.cxx G__CASTOR.cxx LIBRARIES Core ${CASTOR_LIBRARIES} ${CASTOR_rfio_LIBRARY}
                    ${CASTOR_ns_LIBRARY} ${CASTOR_common_LIBRARY} ${CASTOR_client_LIBRARY}
                    DEPENDENCIES Net RIO)

ROOT_INSTALL_HEADERS()
