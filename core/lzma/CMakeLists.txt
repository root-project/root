############################################################################
# CMakeLists.txt file for building ROOT core/lzma package
############################################################################


#---The builtin LMZA library is built using the CMake ExternalProject standard module
#   in cmake/modules/SearchInstalledSoftare.cmake

#---Declare ZipLZMA sources as part of libCore-------------------------------
set(headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/ZipLZMA.h)
set(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/ZipLZMA.c)


include_directories(${LZMA_INCLUDE_DIR})
ROOT_OBJECT_LIBRARY(Lzma ${sources})

if(builtin_lzma)
  add_dependencies(Lzma LZMA)
endif()

ROOT_INSTALL_HEADERS()
install(FILES ${LZMA_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
