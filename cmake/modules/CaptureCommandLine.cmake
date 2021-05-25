# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

#
# Try to capture the initial set of cmake command line args passed by
# the user for configuration.
# Original Recipe taken from http://stackoverflow.com/questions/10205986/how-to-capture-cmake-command-line-arguments
#
# Note: The entries will live on CMakeCache.txt, so re-configuring with
# a command line that doesn't include an option won't remove it. You need
# to remove the CMakeCache.txt file, or override the value via the command line.
#

get_cmake_property(CACHE_VARS CACHE_VARIABLES)
foreach(CACHE_VAR ${CACHE_VARS})
  get_property(CACHE_VAR_HELPSTRING CACHE ${CACHE_VAR} PROPERTY HELPSTRING)
  if(CACHE_VAR_HELPSTRING STREQUAL "No help, variable specified on the command line.")
    get_property(CACHE_VAR_TYPE CACHE ${CACHE_VAR} PROPERTY TYPE)
    if(CACHE_VAR_TYPE STREQUAL UNINITIALIZED)
      set(CACHE_VAR_TYPE)
    else()
      set(CACHE_VAR_TYPE :${CACHE_VAR_TYPE})
    endif()
    set(CMAKE_INVOKE_ARGS "${CMAKE_INVOKE_ARGS} -D${CACHE_VAR}${CACHE_VAR_TYPE}=\"${${CACHE_VAR}}\"")
    # Record the variable also in the cache    
    set(${CACHE_VAR}-CACHED "${${CACHE_VAR}}" CACHE STRING "" FORCE)
  endif()
endforeach()

# Record the full command line invocation.
set(CMAKE_INVOKE "${CMAKE_COMMAND} ${CMAKE_INVOKE_ARGS} ${CMAKE_CURRENT_SOURCE_DIR}" CACHE STRING "Command used to invoke cmake" FORCE)
# Create a simple shell script that allows us to reinvoke cmake with the captured command line.
if(NOT WIN32)
  if (NOT ${CMAKE_GENERATOR} STREQUAL "Unix Makefiles")
    set(RECMAKE_GENERATOR "-G ${CMAKE_GENERATOR}")
  endif()
  set(RECMAKE_REPLAY_FILE ${CMAKE_BINARY_DIR}/recmake_replay.sh)
  set(RECMAKE_INITIAL_FILE ${CMAKE_BINARY_DIR}/recmake_initial.sh)
  if (NOT EXISTS ${RECMAKE_INITIAL_FILE})
      FILE(WRITE ${RECMAKE_INITIAL_FILE} "#!/bin/sh\n"
              "rm -f CMakeCache.txt\n"
              "${CMAKE_INVOKE} ${RECMAKE_GENERATOR}\n")
  endif()
  if (EXISTS ${RECMAKE_REPLAY_FILE})
    FILE(APPEND ${RECMAKE_REPLAY_FILE} "${CMAKE_INVOKE}\n")
  else()
    FILE(WRITE ${RECMAKE_REPLAY_FILE} "#!/bin/sh\n"
          "rm -f CMakeCache.txt\n"
          "${CMAKE_INVOKE} ${RECMAKE_GENERATOR}\n")
  endif()
 endif()
