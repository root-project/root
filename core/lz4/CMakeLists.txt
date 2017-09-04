############################################################################
# CMakeLists.txt file for building ROOT core/lz4 package
############################################################################


#---The builtin LZ4 library is built using the CMake ExternalProject standard module
#   in cmake/modules/SearchInstalledSoftare.cmake

#---Declare ZipLZ4 sources as part of libCore-------------------------------
set(LZ4_headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/ZipLZ4.h)
set(LZ4_sources ${CMAKE_CURRENT_SOURCE_DIR}/src/ZipLZ4.cxx)

list(APPEND LibCore_SRCS ${LZ4_sources})
list(APPEND LibCore_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/inc)

set(LibCore_SRCS ${LibCore_SRCS} PARENT_SCOPE)
set(LibCore_INCLUDE_DIRS ${LibCore_INCLUDE} PARENT_SCOPE)

foreach(dir ${LZ4_INCLUDE_DIR})
	list(APPEND LibCore_INCLUDE_DIRS ${dir})
endforeach()

install(FILES ${LZ4_headers} COMPONENT headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
