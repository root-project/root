#---Set Link and include directories--------------------------------------------------------------
include_directories(${ROOT_INCLUDE_DIRS})
link_directories(${ROOT_LIBRARY_DIR})

#---Set Flags-------------------------------------------------------------------------------------
add_definitions(${ROOT_DEFINITIONS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ROOT_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ROOT_C_FLAGS}")
set(CMAKE_fortran_FLAGS "${CMAKE_fortran_FLAGS} ${ROOT_fortran_FLAGS}")
