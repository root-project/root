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

Set(ZipNewHeaders
  ${CMAKE_CURRENT_SOURCE_DIR}/inc/zlib.h
  ${CMAKE_CURRENT_SOURCE_DIR}/inc/zconf.h
)

Set(ZipNewSource
  ${CMAKE_CURRENT_SOURCE_DIR}/src/adler32.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/compress.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/crc32.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/deflate.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/gzclose.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/gzlib.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/gzread.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/gzwrite.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/infback.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/inffast.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/inflate.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/inftrees.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/trees.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/uncompr.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/zutil.c
)

# Depending on the system either the old and the new
# or only the old compression is build
if(builtin_zlib)
  set(ZLIB_HEADERS ${ZipNewHeaders} ${ZipOldHeaders})
  set(ZLIB_SRCS ${ZipNewSource} ${ZipOldSource} ${CMAKE_CURRENT_SOURCE_DIR}/src/Compression.cxx ${CMAKE_CURRENT_SOURCE_DIR}/src/RZip.cxx)
else()
  set(ZLIB_HEADERS ${ZipOldHeaders})
  set(ZLIB_SRCS ${ZipOldSource} ${CMAKE_CURRENT_SOURCE_DIR}/src/Compression.cxx ${CMAKE_CURRENT_SOURCE_DIR}/src/RZip.cxx)
endif()

# Define all the header files which should be installed when
# doing a "make install"
install(FILES ${ZLIB_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT headers)
set_property(GLOBAL APPEND PROPERTY ROOT_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/inc)

# Add all source files of this subdirectory to the source files
# needed to build the Core library
# Variables are only defined in this directory-. Some of the
# variables are needed in the parent scope so they are exported
# to the parent scope
list(APPEND LibCore_SRCS ${ZLIB_SRCS})
list(APPEND LibCore_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(LibCore_SRCS ${LibCore_SRCS} PARENT_SCOPE)
set(LibCore_INCLUDE_DIRS ${LibCore_INCLUDE} PARENT_SCOPE)
