project(zstd LANGUAGES C)

find_package(xxHash REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(AddZstdCompilationFlags)
ADD_ZSTD_COMPILATION_FLAGS()

file(READ zstd.h CONTENT)
string(REGEX MATCH ".*define ZSTD_VERSION_MAJOR *([0-9]+).*define ZSTD_VERSION_MINOR *([0-9]+).*define ZSTD_VERSION_RELEASE *([0-9]+)" VERSION_REGEX "${CONTENT}")
set(ZSTD_VERSION_MAJOR ${CMAKE_MATCH_1})
set(ZSTD_VERSION_MINOR ${CMAKE_MATCH_2})
set(ZSTD_VERSION_PATCH ${CMAKE_MATCH_3})
set(ZSTD_VERSION_STRING "${ZSTD_VERSION_MAJOR}.${ZSTD_VERSION_MINOR}.${ZSTD_VERSION_PATCH}")

set(ZSTD_SOURCES
        common/error_private.c
        common/fse_decompress.c
        common/pool.c
        common/zstd_common.c
        common/entropy_common.c
        common/threading.c
        compress/fse_compress.c
        compress/zstd_compress_superblock.c
        compress/zstd_fast.c
        compress/hist.c
        compress/zstd_compress_literals.c
        compress/zstd_opt.c
        compress/zstd_lazy.c
        compress/huf_compress.c
        compress/zstd_compress_sequences.c
        compress/zstd_double_fast.c
        compress/zstdmt_compress.c
        compress/zstd_compress.c
        compress/zstd_ldm.c
        decompress/huf_decompress.c
        decompress/zstd_ddict.h
        decompress/zstd_decompress_block.c
        decompress/zstd_ddict.c
        decompress/zstd_decompress.c
        dictBuilder/cover.c
        dictBuilder/divsufsort.c
        dictBuilder/fastcover.c
        dictBuilder/zdict.c
)

set(ZSTD_HEADERS
       zstd.h
       zstd_errors.h
       zdict.h
       common/bitstream.h
       common/threading.h
       common/zstd_internal.h
       common/compiler.h
       common/debug.h
       common/error_private.h
       common/huf.h
       common/pool.h
       common/zstd_deps.h
       common/cpu.h
       common/entropy_common.c
       common/fse.h
       common/mem.h
       compress/zstd_compress_internal.h
       compress/zstd_ldm.h
       compress/zstd_compress_superblock.h
       compress/zstd_fast.h
       compress/hist.h
       compress/zstd_compress_literals.h
       compress/zstd_cwksp.h
       compress/zstd_opt.h
       decompress/zstd_lazy.h
       decompress/zstd_compress_sequences.h
       decompress/zstd_double_fast.h
       decompress/zstdmt_compress.h
       decompress/zstd_ddict.h
       decompress/zstd_decompress_internal.h
       decompress/zstd_decompress_block.h
       dictBuilder/cover.h
       dictBuilder/divsufsort.h
)

set(ZSTD_VERSION ${ZSTD_VERSION_STRING} CACHE INTERNAL "" FORCE)
set(ZSTD_VERSION_STRING ${ZSTD_VERSION_STRING} CACHE INTERNAL "" FORCE)
set(ZSTD_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "" FORCE)
set(ZSTD_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "" FORCE)

add_library(zstd STATIC ${ZSTDHEADERS} ${ZSTD_SOURCES})
target_compile_definitions(zstd PRIVATE "XXH_NAMESPACE=ZSTD_")
set_target_properties(zstd PROPERTIES COMPILE_DEFINITIONS "ZSTD_HEAPMODE=0;_CRT_SECURE_NO_WARNINGS" C_VISIBILITY_PRESET hidden)
target_include_directories(zstd PRIVATE ${ZSTD_INCLUDE_DIR} INTERFACE $<BUILD_INTERFACE:${ZSTD_INCLUDE_DIR}>)
target_link_libraries(zstd PRIVATE xxHash::xxHash)
if(NOT MSVC)
  target_compile_options(zstd PRIVATE -fPIC -Wno-unused-variable -O3)
endif()
add_library(ZSTD::ZSTD ALIAS zstd)

set(ZSTD_LIBRARY $<TARGET_FILE:ZSTD> CACHE INTERNAL "")
set(ZSTD_LIBRARIES ZSTD::ZSTD CACHE INTERNAL "")

set_property(GLOBAL APPEND PROPERTY ROOT_BUILTIN_TARGETS ZSTD::ZSTD)
