############################################################################
# CMakeLists.txt file for building ROOT io/castor package
############################################################################

include_directories(${CASTOR_INCLUDE_DIR})

if(CASTOR_VERSION VERSION_GREATER "2.0.0")
  add_definitions(-DR__CASTOR2)
endif()

ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-shadow)

ROOT_STANDARD_LIBRARY_PACKAGE(RCastor
                              LIBRARIES ${CASTOR_LIBRARIES} ${CASTOR_rfio_LIBRARY}
                                        ${CASTOR_ns_LIBRARY} ${CASTOR_common_LIBRARY}
                                        ${CASTOR_client_LIBRARY}
                              DEPENDENCIES Net RIO Core)
