project(ZLIB C)

set(ZLIB_PUBLIC_HEADERS
    zconf.h
    zlib.h
)

set(ZLIB_PRIVATE_HEADERS
    crc32.h
    deflate.h
    gzguts.h
    inffast.h
    inffixed.h
    inflate.h
    inftrees.h
    trees.h
    zutil.h
)

set(ZLIB_SOURCES
    adler32.c
    compress.c
    crc32.c
    deflate.c
    gzclose.c
    gzlib.c
    gzread.c
    gzwrite.c
    inflate.c
    infback.c
    inftrees.c
    inffast.c
    trees.c
    uncompr.c
    zutil.c
)

file(STRINGS zlib.h ZLIB_H REGEX "^#define ZLIB_VERSION \"[^\"]*\"$")
string(REGEX REPLACE "^.*ZLIB_VERSION \"([0-9]+).*$" "\\1" ZLIB_VERSION_MAJOR "${ZLIB_H}")
string(REGEX REPLACE "^.*ZLIB_VERSION \"[0-9]+\\.([0-9]+).*$" "\\1" ZLIB_VERSION_MINOR  "${ZLIB_H}")
string(REGEX REPLACE "^.*ZLIB_VERSION \"[0-9]+\\.[0-9]+\\.([0-9]+).*$" "\\1" ZLIB_VERSION_PATCH "${ZLIB_H}")
set(ZLIB_VERSION_STRING "${ZLIB_VERSION_MAJOR}.${ZLIB_VERSION_MINOR}.${ZLIB_VERSION_PATCH}" CACHE INTERNAL "")

set(ZLIB_VERSION_TWEAK "")
if( "${ZLIB_H}" MATCHES "ZLIB_VERSION \"[0-9]+\\.[0-9]+\\.[0-9]+\\.([0-9]+)")
		set(ZLIB_VERSION_TWEAK "${CMAKE_MATCH_1}")
		string(APPEND ZLIB_VERSION_STRING ".${ZLIB_VERSION_TWEAK}")
endif()

set(ZLIB_VERSION "${ZLIB_VERSION_STRING}" CACHE INTERNAL "")
set(ZLIB_VERSION_MAJOR "${ZLIB_VERSION_MAJOR}" CACHE INTERNAL "")
set(ZLIB_VERSION_MINOR "${ZLIB_VERSION_MINOR}" CACHE INTERNAL "")
set(ZLIB_VERSION_PATCH "${ZLIB_VERSION_PATCH}" CACHE INTERNAL "")

add_library(zlib STATIC ${ZLIB_SOURCES} ${ZLIB_PUBLIC_HEADERS} ${ZLIB_PRIVATE_HEADERS})
target_compile_options(zlib PRIVATE -fPIC)
target_include_directories(zlib INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<INSTALL_INTERFACE:include>)
add_library(ZLIB::ZLIB ALIAS zlib)

set(ZLIB_LIBRARY zlib CACHE INTERNAL "")
set(ZLIB_LIBRARIES zlib CACHE INTERNAL "")
set(ZLIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "")
set(ZLIB_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "")

install(FILES ${ZLIB_PUBLIC_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT headers)
install(TARGETS zlib DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ZLIB REQUIRED_VARS ZLIB_LIBRARY ZLIB_INCLUDE_DIR VERSION_VAR ZLIB_VERSION)

if(NOT ZLIB_FOUND)
  message(FATAL_ERROR "Setup of builtin ZLIB project failed!")
endif()
