project(LZMA C)

unset(LZMA_FOUND CACHE)
unset(LZMA_FOUND PARENT_SCOPE)
set(LZMA_FOUND TRUE CACHE BOOL "" FORCE)

include(CheckIncludeFiles)

check_include_files(CommonCrypto/CommonDigest.h HAVE_COMMONCRYPTO_COMMONDIGEST_H)
check_include_files(inttypes.h HAVE_INTTYPES_H)
check_include_files(limits.h HAVE_LIMITS_H)
check_include_files(memory.h HAVE_MEMORY_H)
check_include_files(sha256.h HAVE_SHA256_H)
check_include_files(sha2.h HAVE_SHA2_H)
check_include_files(stdbool.h HAVE_STDBOOL_H)
check_include_files(stdint.h HAVE_STDINT_H)
check_include_files(string.h HAVE_STRING_H)
check_include_files(strings.h HAVE_STRINGS_H)
check_include_files(sys/types.h HAVE_SYS_TYPES_H)

add_definitions(-DHAVE_CONFIG_H)
configure_file(config.h.in ${CMAKE_CURRENT_BINARY_DIR}/include/config.h)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(api check common delta lz lzma rangecoder simple tuklib)

find_package(Threads REQUIRED)

if(WIN32)
  set(LZMA_DEFINITIONS -DMYTHREAD_WIN95)
else()
  set(LZMA_DEFINITIONS -DMYTHREAD_POSIX)
endif()

file(STRINGS api/lzma/version.h LZMA_H REGEX "^#define LZMA_VERSION_[A-Z]+[ ]+[0-9]+$")
string(REGEX REPLACE ".+LZMA_VERSION_MAJOR[ ]+([0-9]+).*$" "\\1" LZMA_VERSION_MAJOR "${LZMA_H}")
string(REGEX REPLACE ".+LZMA_VERSION_MINOR[ ]+([0-9]+).*$" "\\1" LZMA_VERSION_MINOR "${LZMA_H}")
string(REGEX REPLACE ".+LZMA_VERSION_PATCH[ ]+([0-9]+).*$" "\\1" LZMA_VERSION_PATCH "${LZMA_H}")
set(LZMA_VERSION_STRING "${LZMA_VERSION_MAJOR}.${LZMA_VERSION_MINOR}.${LZMA_VERSION_PATCH}")
unset(LZMA_H)

set(LZMA_VERSION ${LZMA_VERSION_STRING} CACHE INTERNAL "")
set(LZMA_VERSION_STRING ${LZMA_VERSION_STRING} CACHE INTERNAL "")

set(LZMA_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/api CACHE INTERNAL "")
set(LZMA_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/api CACHE INTERNAL "")

file(GLOB_RECURSE LZMA_HEADERS *.h)
file(GLOB_RECURSE LZMA_SOURCES *.c)

add_library(lzma STATIC ${LZMA_HEADERS} ${LZMA_SOURCES})
target_compile_options(lzma PRIVATE -w -fvisibility=hidden)
target_compile_definitions(lzma PRIVATE ${LZMA_DEFINITIONS})
target_link_libraries(lzma Threads::Threads)

set_target_properties(lzma PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED True)

add_library(LZMA::LZMA ALIAS lzma)

set(LZMA_LIBRARY $<TARGET_FILE:lzma> CACHE INTERNAL "")
set(LZMA_LIBRARIES LZMA::LZMA CACHE INTERNAL "")

set_property(GLOBAL APPEND PROPERTY ROOT_BUILTIN_TARGETS LZMA::LZMA)
