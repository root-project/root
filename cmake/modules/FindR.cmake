# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# CMake module to find R
# - Try to find R
# Once done, this will define
#
#  R_FOUND - system has R
#  R_INCLUDE_DIRS - the R include directories
#  R_LIBRARIES - link these to use R
#  R_ROOT_DIR - As reported by R
#  R_EXECUTABLE - the R executable
#  R_SCRIPT - the Rscript executable, which runs R non-interactively
#
# Autor: Omar Andres Zapata Mesa 31/05/2013
# Contributor: Blake Madden 2023-12-10

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(CMAKE_FIND_APPBUNDLE "LAST")
endif()

# Lists subdirectories in a directory
# https://stackoverflow.com/questions/7787823/cmake-how-to-get-the-name-of-all-subdirectories-of-a-directory
macro(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  set(dirlist "")
  foreach(child ${children})
    if(IS_DIRECTORY ${curdir}/${child})
      list(APPEND dirlist ${child})
    endif()
  endforeach()
  set(${result} ${dirlist})
endmacro()

find_program(R_EXECUTABLE NAMES R R.exe)
find_program(R_SCRIPT NAMES Rscript Rscript.exe)

# If not found and we are on Windows, try to look for it in the default installation path
if(WIN32 AND R_EXECUTABLE MATCHES "R_EXECUTABLE-NOTFOUND")
  GET_FILENAME_COMPONENT(RX64PATH "C:\\Program Files\\R" REALPATH)
  GET_FILENAME_COMPONENT(RX86PATH "C:\\Program Files (x86)\\R" REALPATH)
  # default 64-bit Windows installation
  if(EXISTS "${RX64PATH}")
    SUBDIRLIST(SUBDIRS "${RX64PATH}")
    foreach(subdir ${SUBDIRS})
      if(${subdir} MATCHES "R[-]([0-9][.]).*")
        set(R_VERSIONED_SUBDIR "${subdir}")
        if(EXISTS "${RX64PATH}\\${R_VERSIONED_SUBDIR}\\bin\\x64\\R.exe")
          set(R_EXECUTABLE "${RX64PATH}\\${R_VERSIONED_SUBDIR}\\bin\\x64\\R.exe")
        endif()
        if(EXISTS "${RX64PATH}\\${R_VERSIONED_SUBDIR}\\bin\\x64\\Rscript.exe")
          set(R_SCRIPT "${RX64PATH}\\${R_VERSIONED_SUBDIR}\\bin\\x64\\Rscript.exe")
        endif()
        if (R_EXECUTABLE AND R_SCRIPT)
          break()
        endif()
        unset(R_EXECUTABLE)
        unset(R_SCRIPT)
      endif()
    endforeach()
  # ...or the 32-bit installation
  elseif(EXISTS "${RX86PATH}")
    SUBDIRLIST(SUBDIRS "${RX86PATH}")
    foreach(subdir ${SUBDIRS})
      if(${subdir} MATCHES "R[-]([0-9][.]).*")
        set(R_VERSIONED_SUBDIR "${subdir}")
        if(EXISTS "${RX86PATH}\\${R_VERSIONED_SUBDIR}\\bin\\x86\\R.exe")
          set(R_EXECUTABLE "${RX86PATH}\\${R_VERSIONED_SUBDIR}\\bin\\x86\\R.exe")
        endif()
        if(EXISTS "${RX86PATH}\\${R_VERSIONED_SUBDIR}\\bin\\x86\\Rscript.exe")
          set(R_SCRIPT "${RX86PATH}\\${R_VERSIONED_SUBDIR}\\bin\\x86\\Rscript.exe")
        endif()
        if (R_EXECUTABLE AND R_SCRIPT)
          break()
        endif()
        unset(R_EXECUTABLE)
        unset(R_SCRIPT)
      endif()
    endforeach()
  endif()
endif()

#---searching R installtion using R executable
if(R_EXECUTABLE)
  execute_process(COMMAND ${R_EXECUTABLE} RHOME
                  OUTPUT_VARIABLE R_ROOT_DIR
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  find_path(R_INCLUDE_DIR R.h
            HINTS ${R_ROOT_DIR}
            PATHS /usr/local/lib /usr/local/lib64 /usr/share
            PATH_SUFFIXES include R/include
            DOC "Path to file R.h")

  find_library(R_LIBRARY R
            HINTS ${R_ROOT_DIR}/lib
            DOC "R library (example libR.a, libR.dylib, etc.).")
  # On Windows, this may not be defined. Fall back to the library
  # folder that holds the DLLs.
  if(R_LIBRARY MATCHES "R_LIBRARY-NOTFOUND" AND EXISTS "${R_ROOT_DIR}\\library")
    set(R_LIBRARY "${R_ROOT_DIR}\\library")
  endif()
endif()

#---setting include dirs and libraries
set(R_LIBRARIES ${R_LIBRARY})
set(R_INCLUDE_DIRS ${R_INCLUDE_DIR})
foreach(_cpt ${R_FIND_COMPONENTS})
  execute_process(COMMAND echo "cat(find.package('${_cpt}'))"
                  COMMAND ${R_EXECUTABLE} --vanilla --slave
                  RESULT_VARIABLE _rc
                  ERROR_QUIET
                  OUTPUT_VARIABLE _cpt_path
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _rc)
    set(R_${_cpt}_FOUND 1)
  endif()

  find_library(R_${_cpt}_LIBRARY
               lib${_cpt}.so lib${_cpt}.dylib
               HINTS ${_cpt_path}/lib)
  if(R_${_cpt}_LIBRARY)
    mark_as_advanced(R_${_cpt}_LIBRARY)
    list(APPEND R_LIBRARIES ${R_${_cpt}_LIBRARY})
  endif()

  find_path(R_${_cpt}_INCLUDE_DIR ${_cpt}.h HINTS  ${_cpt_path} PATH_SUFFIXES include R/include)
  if(R_${_cpt}_INCLUDE_DIR)
    mark_as_advanced(R_${_cpt}_INCLUDE_DIR)
    list(APPEND R_INCLUDE_DIRS ${R_${_cpt}_INCLUDE_DIR})
  endif()

endforeach()

# Handle the QUIETLY and REQUIRED arguments and set R_FOUND to TRUE if all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(R HANDLE_COMPONENTS REQUIRED_VARS R_EXECUTABLE R_INCLUDE_DIR R_LIBRARY)
mark_as_advanced(R_FOUND R_EXECUTABLE R_INCLUDE_DIR R_LIBRARY)
