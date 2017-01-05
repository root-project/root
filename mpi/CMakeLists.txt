SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
#################
#MPI ENVIRONMENT#
#################
FIND_PACKAGE(MPI REQUIRED)

include_directories(${MPI_CXX_INCLUDE_PATH})
ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-overloaded-virtual ${MPI_CXX_COMPILE_FLAGS} -Wno-literal-suffix)

add_subdirectory(rmpi)
add_subdirectory(rootmpi)
