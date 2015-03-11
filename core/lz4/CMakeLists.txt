############################################################################
# CMakeLists.txt file for building ROOT core/lz4 package
############################################################################


#---The builtin LZ4 library is built using the CMake ExternalProject standard module
#   in cmake/modules/SearchInstalledSoftare.cmake

#---Declare ZipLZ4 sources as part of libCore-------------------------------
set(headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/ZipLZ4.h)
set(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/ZipLZ4.c)


include_directories(${LZ4_INCLUDE_DIR})
ROOT_OBJECT_LIBRARY(Lz4 ${sources})

if(builtin_lz4)
  add_dependencies(Lz4 LZ4)
endif()

ROOT_INSTALL_HEADERS()
install(FILES ${LZ4_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
