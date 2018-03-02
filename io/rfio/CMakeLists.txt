############################################################################
# CMakeLists.txt file for building ROOT io/rfio package
############################################################################


ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-shadow)

if(CASTOR_FOUND)
include_directories(${CASTOR_INCLUDE_DIR})
ROOT_STANDARD_LIBRARY_PACKAGE(RFIO
                              LIBRARIES ${CASTOR_LIBRARIES} ${CASTOR_rfio_LIBRARY}
                                        ${CASTOR_common_LIBRARY} ${CASTOR_client_LIBRARY}
                              DEPENDENCIES Core Net RIO)
else()
include_directories(${DPM_INCLUDE_DIR})
ROOT_STANDARD_LIBRARY_PACKAGE(RFIO
                              LIBRARIES ${DPM_LIBRARIES}
                              DEPENDENCIES Core Net RIO)
endif()
