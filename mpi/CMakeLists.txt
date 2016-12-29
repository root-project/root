SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
#################
#MPI ENVIRONMENT#
#################
FIND_PACKAGE(MPI REQUIRED)

##################
#ROOT ENVIRONMENT#
##################
SET(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

add_subdirectory(rmpi)
add_subdirectory(rootmpi)
