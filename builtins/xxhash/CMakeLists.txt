# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

project(xxhash C)

unset(xxHash_FOUND CACHE)
unset(xxHash_FOUND PARENT_SCOPE)
set(xxHash_FOUND TRUE CACHE BOOL "" FORCE)

file(STRINGS xxhash.h XXHASH_H REGEX "^#define XXH_VERSION_[A-Z]+[ ]+[0-9]+$")
string(REGEX REPLACE ".+XXH_VERSION_MAJOR[ ]+([0-9]+).*$"   "\\1" xxHash_VERSION_MAJOR "${XXHASH_H}")
string(REGEX REPLACE ".+XXH_VERSION_MINOR[ ]+([0-9]+).*$"   "\\1" xxHash_VERSION_MINOR "${XXHASH_H}")
string(REGEX REPLACE ".+XXH_VERSION_RELEASE[ ]+([0-9]+).*$" "\\1" xxHash_VERSION_PATCH "${XXHASH_H}")
set(xxHash_VERSION_STRING "${xxHash_VERSION_MAJOR}.${xxHash_VERSION_MINOR}.${xxHash_VERSION_PATCH}")

set(xxHash_VERSION ${xxHash_VERSION_STRING} CACHE INTERNAL "")
set(xxHash_VERSION_STRING ${xxHash_VERSION_STRING} CACHE INTERNAL "")

set(xxHash_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "")
set(xxHash_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "")

add_library(xxhash STATIC xxhash.h xxhash.c)
set_target_properties(xxhash PROPERTIES C_VISIBILITY_PRESET hidden POSITION_INDEPENDENT_CODE ON)
target_include_directories(xxhash INTERFACE $<BUILD_INTERFACE:${xxHash_INCLUDE_DIR}>)

add_library(xxHash::xxHash ALIAS xxhash)

set(xxHash_LIBRARY $<TARGET_FILE:xxhash> CACHE INTERNAL "")
set(xxHash_LIBRARIES xxHash::xxHash CACHE INTERNAL "")

set_property(GLOBAL APPEND PROPERTY ROOT_BUILTIN_TARGETS xxHash::xxHash)
