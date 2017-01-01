
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include_directories(${CMAKE_SOURCE_DIR}/mpi/rootmpi/inc/)
add_definitions(-w)

SET(ROOTMPI_SOURCES src/main.cxx src/TROOTMpi.cxx)
add_executable(rootmpi ${ROOTMPI_SOURCES})
target_link_libraries(rootmpi Core Rint)
install(TARGETS rootmpi DESTINATION bin)