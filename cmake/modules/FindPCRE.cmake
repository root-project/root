# Find the PCRE includes and library.
# 
# This module defines
# PCRE_INCLUDE_DIR, where to locate PCRE header files
# PCRE_LIBRARIES, the libraries to link against to use Pythia6
# PCRE_FOUND.  If false, you cannot build anything that requires Pythia6.

if(PCRE_CONFIG_EXECUTABLE)
  set(PCRE_FIND_QUIETLY 1)
endif()
set(PCRE_FOUND 0)


find_program(PCRE_CONFIG_EXECUTABLE pcre-config)

if(PCRE_CONFIG_EXECUTABLE)
  execute_process(COMMAND ${PCRE_CONFIG_EXECUTABLE} --version OUTPUT_VARIABLE PCRE_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${PCRE_CONFIG_EXECUTABLE} --cflags OUTPUT_VARIABLE PCRE_CFLAGS)
  string( REGEX MATCHALL "-I[^;]+" PCRE_INCLUDE_DIR "${PCRE_CFLAGS}" )
  string( REPLACE "-I" "" PCRE_INCLUDE_DIR "${PCRE_INCLUDE_DIRS}")
  if(NOT  PCRE_INCLUDE_DIR)
    execute_process(COMMAND ${PCRE_CONFIG_EXECUTABLE} --prefix OUTPUT_VARIABLE PCRE_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(PCRE_INCLUDE_DIR ${PCRE_PREFIX}/include)
  endif()
  execute_process(COMMAND ${PCRE_CONFIG_EXECUTABLE} --libs OUTPUT_VARIABLE PCRE_LIBRARIES OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(PCRE_FOUND 1)  
endif()

if(PCRE_FOUND)
  if(NOT PCRE_FIND_QUIETLY)
    message(STATUS "Found PCRE version ${PCRE_VERSION} using ${PCRE_CONFIG_EXECUTABLE}")
  endif()
endif()

mark_as_advanced(PCRE_CONFIG_EXECUTABLE)
