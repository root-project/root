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

set(ZLIBCF_PRIVATE_HEADERS
    crc32.h
    deflate_cf.h
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

set(ZLIBCF_SOURCES
    adler32_cf.c
    compress.c
    crc32_cf.c
    deflate_cf.c
    gzclose.c
    gzlib.c
    gzread.c
    gzwrite.c
    inflate.c
    infback.c
    inftrees.c
    inffast.c
    trees_cf.c
    uncompr.c
    zutil.c
)

unset(ZLIB_FOUND CACHE)
unset(ZLIB_FOUND PARENT_SCOPE)
set(ZLIB_FOUND TRUE CACHE BOOL "" FORCE)

file(STRINGS zlib.h ZLIB_H REGEX "^#define ZLIB_VERSION \"[^\"]*\"$")
string(REGEX REPLACE "^.*ZLIB_VERSION \"([0-9]+).*$" "\\1" ZLIB_VERSION_MAJOR "${ZLIB_H}")
string(REGEX REPLACE "^.*ZLIB_VERSION \"[0-9]+\\.([0-9]+).*$" "\\1" ZLIB_VERSION_MINOR  "${ZLIB_H}")
string(REGEX REPLACE "^.*ZLIB_VERSION \"[0-9]+\\.[0-9]+\\.([0-9]+).*$" "\\1" ZLIB_VERSION_PATCH "${ZLIB_H}")
set(ZLIB_VERSION_STRING "${ZLIB_VERSION_MAJOR}.${ZLIB_VERSION_MINOR}.${ZLIB_VERSION_PATCH}")

set(ZLIB_VERSION_TWEAK "")
if("${ZLIB_H}" MATCHES "ZLIB_VERSION \"[0-9]+\\.[0-9]+\\.[0-9]+\\.([0-9]+)")
  set(ZLIB_VERSION_TWEAK "${CMAKE_MATCH_1}")
  string(APPEND ZLIB_VERSION_STRING ".${ZLIB_VERSION_TWEAK}")
endif()

set(ZLIB_VERSION ${ZLIB_VERSION_STRING} CACHE INTERNAL "")
set(ZLIB_VERSION_STRING ${ZLIB_VERSION_STRING} CACHE INTERNAL "")

set(ZLIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "")
set(ZLIB_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "")

if((CMAKE_SYSTEM_PROCESSOR MATCHES "amd64|x86_64|AMD64|X86_64|aarch64") AND (CMAKE_SYSTEM_NAME MATCHES "Linux"))
   # Calling helper to avoid using old unsupported binutils (e.g. with SL6)
   # macro is returning extra ${ROOT_DEFINITIONS} used after in ZLIB-CF
   root_check_assembler()
   # Calling helper to avoid using old unsupported binutils (e.g. with Centos7
   # and native gcc compiler 4.8.5)
   # Macros are returning bools SSE_SUPPORT & AVX2_SUPPORT
   root_check_sse41()
   root_check_avx2()
   if(SSE_SUPPORT OR AVX2_SUPPORT)
      set(ZLIB_CF TRUE CACHE INTERNAL "")
   endif()
endif()

if(ZLIB_CF)
   add_library(ZLIB STATIC ${ZLIB_PUBLIC_HEADERS} ${ZLIBCF_PRIVATE_HEADERS} ${ZLIBCF_SOURCES})
else()
   add_library(ZLIB STATIC ${ZLIB_PUBLIC_HEADERS} ${ZLIB_PRIVATE_HEADERS} ${ZLIB_SOURCES})
endif()

set_target_properties(ZLIB PROPERTIES C_VISIBILITY_PRESET hidden POSITION_INDEPENDENT_CODE ON)
target_include_directories(ZLIB INTERFACE $<BUILD_INTERFACE:${ZLIB_INCLUDE_DIR}>)

if((CMAKE_SYSTEM_PROCESSOR MATCHES "amd64|x86_64|AMD64|X86_64") AND (CMAKE_SYSTEM_NAME MATCHES "Linux"))
   target_compile_options(ZLIB PRIVATE -Wno-unused-function -O3 -mpclmul -Wno-attribute-alias ${ROOT_DEFINITIONS})
else()
if(NOT MSVC)
    target_compile_options(ZLIB PRIVATE -O3)
endif()
endif()

add_library(ZLIB::ZLIB ALIAS ZLIB)

set(ZLIB_LIBRARY $<TARGET_FILE:ZLIB> CACHE INTERNAL "")
set(ZLIB_LIBRARIES ZLIB::ZLIB CACHE INTERNAL "")

if(DEFINED ZLIB_LIBRARY_DEBUG)
  set(ZLIB_LIBRARY_DEBUG ${ZLIB_LIBRARY} CACHE INTERNAL "")
endif()

if(DEFINED ZLIB_LIBRARY_RELEASE)
  set(ZLIB_LIBRARY_RELEASE ${ZLIB_LIBRARY} CACHE INTERNAL "")
endif()

set_property(GLOBAL APPEND PROPERTY ROOT_BUILTIN_TARGETS ZLIB::ZLIB)
