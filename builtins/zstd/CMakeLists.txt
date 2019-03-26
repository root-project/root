project(zstd LANGUAGES C)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(AddZstdCompilationFlags)
ADD_ZSTD_COMPILATION_FLAGS()

# Always hide XXHash symbols
add_definitions(-DXXH_NAMESPACE=ZSTD_)

file(READ zstd.h CONTENT)
string(REGEX MATCH ".*define ZSTD_VERSION_MAJOR *([0-9]+).*define ZSTD_VERSION_MINOR *([0-9]+).*define ZSTD_VERSION_RELEASE *([0-9]+)" VERSION_REGEX "${CONTENT}")
set(ZSTD_VERSION_MAJOR ${CMAKE_MATCH_1})
set(ZSTD_VERSION_MINOR ${CMAKE_MATCH_2})
set(ZSTD_VERSION_PATCH ${CMAKE_MATCH_3})
set(ZSTD_VERSION_STRING "${ZSTD_VERSION_MAJOR}.${ZSTD_VERSION_MINOR}.${ZSTD_VERSION_PATCH}")

set(ZSTD_SOURCES
        entropy_common.c
        fse_decompress.c
        threading.c
        pool.c
        zstd_common.c
        error_private.c
        xxhash.c
        hist.c
        fse_compress.c
        huf_compress.c
        zstd_compress.c
        zstdmt_compress.c
        zstd_fast.c
        zstd_double_fast.c
        zstd_lazy.c
        zstd_opt.c
        zstd_ldm.c
        huf_decompress.c
        zstd_decompress.c
        zstd_decompress_block.c
        zstd_ddict.c
        cover.c
        fastcover.c
        divsufsort.c
        zdict.c
        zbuff_common.c
        zbuff_compress.c
        zbuff_decompress.c
)

set(ZSTD_PUBLIC_HEADERS
        zstd.h
)

set(ZSTD_PRIVATE_HEADERS
        zstd.h
        debug.h
        pool.h
        threading.h
        bitstream.h
        error_private.h
        zstd_errors.h
        fse.h
        huf.h
        mem.h
        zstd_internal.h
        hist.h
        zstd_compress_internal.h
        zstd_fast.h
        zstd_double_fast.h
        zstd_lazy.h
        zstd_opt.h
        zstd_ldm.h
        zstdmt_compress.h
        zstd_decompress_internal.h
        zstd_decompress_block.h
        zstd_ddict.h
        zdict.h
        cover.h
        zbuff.h
)

set(ZSTD_VERSION ${ZSTD_VERSION_STRING} CACHE INTERNAL "" FORCE)
set(ZSTD_VERSION_STRING ${ZSTD_VERSION_STRING} CACHE INTERNAL "" FORCE)
set(ZSTD_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "" FORCE)
set(ZSTD_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "" FORCE)

add_library(zstd STATIC ${ZSTD_PUBLIC_HEADERS} ${ZSTD_PRIVATE_HEADERS} ${ZSTD_SOURCES})
set_target_properties(zstd PROPERTIES COMPILE_DEFINITIONS "ZSTD_HEAPMODE=0;_CRT_SECURE_NO_WARNINGS" C_VISIBILITY_PRESET hidden)
target_include_directories(zstd INTERFACE $<BUILD_INTERFACE:${ZSTD_INCLUDE_DIR}>)
target_compile_options(zstd PRIVATE -fPIC -Wno-unused-variable -O3)
add_library(ZSTD::ZSTD ALIAS zstd)

set(ZSTD_LIBRARY $<TARGET_FILE:ZSTD> CACHE INTERNAL "")
set(ZSTD_LIBRARIES ZSTD::ZSTD CACHE INTERNAL "")

set_property(GLOBAL APPEND PROPERTY ROOT_BUILTIN_TARGETS ZSTD::ZSTD)
