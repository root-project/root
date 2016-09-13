# Build core/clib
# Define all header and source files which are needed
# to build the Core library
# The Core library is build in the main CMakeLists.txt
# of the core subdirectory.

Set(ZipOldHeaders
  ${CMAKE_CURRENT_SOURCE_DIR}/inc/Compression.h
  ${CMAKE_CURRENT_SOURCE_DIR}/inc/RZip.h
)

Set(ZipOldSource
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ZDeflate.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ZInflate.c
)

# Depending on the system either the old and the new
# or only the old compression is build
set(ZLIB_HEADERS ${ZipOldHeaders})
set(ZLIB_SRCS ${ZipOldSource} ${CMAKE_CURRENT_SOURCE_DIR}/src/Compression.cxx ${CMAKE_CURRENT_SOURCE_DIR}/src/RZip.cxx)

# Define all the header files which should be installed when
# doing a "make install"
install(FILES ${ZLIB_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT headers)
set_property(GLOBAL APPEND PROPERTY ROOT_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/inc)

ROOT_OBJECT_LIBRARY(Zip ${ZLIB_SRCS})
