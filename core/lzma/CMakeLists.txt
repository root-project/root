############################################################################
# CMakeLists.txt file for building ROOT core/lzma package
############################################################################


#---The builtin LMZA library is built using the CMake ExternalProject standard module
#   in cmake/modules/SearchInstalledSoftare.cmake

#---Declare ZipLZMA sources as part of libCore------------------------------- 
set(LZMA_headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/ZipLZMA.h)
set(LZMA_sources ${CMAKE_CURRENT_SOURCE_DIR}/src/ZipLZMA.c)

list(APPEND LibCore_SRCS ${LZMA_sources})
list(APPEND LibCore_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/inc)

set(LibCore_SRCS ${LibCore_SRCS} PARENT_SCOPE)
set(LibCore_INCLUDE_DIRS ${LibCore_INCLUDE} PARENT_SCOPE)

install(FILES ${LZMA_headers} DESTINATION include)

