SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
#################
#MPI ENVIRONMENT#
#################
FIND_PACKAGE(MPI REQUIRED)

add_subdirectory(rmpi)
add_subdirectory(rootmpi)
