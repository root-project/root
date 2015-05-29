# CMake module to find R
# - Try to find R
# Once done, this will define
#
#  R_FOUND - system has R
#  R_INCLUDE_DIRS - the R include directories
#  R_LIBRARIES - link these to use R
#  R_ROOT_DIR - As reported by R
# Autor: Omar Andres Zapata Mesa 31/05/2013

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(CMAKE_FIND_APPBUNDLE "LAST")
endif()

find_program(R_EXECUTABLE NAMES R R.exe)

#---searching R installtion unsing R executable
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
endif()

#---setting include dirs and libraries
set(R_LIBRARIES ${R_LIBRARY})
set(R_INCLUDE_DIRS ${R_INCLUDE_DIR})
foreach(_cpt ${R_FIND_COMPONENTS})
  execute_process(COMMAND echo "cat(find.package('${_cpt}'))"
                  COMMAND ${R_EXECUTABLE} --vanilla --slave
                  OUTPUT_VARIABLE _cpt_path
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

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

  if(R_${_cpt}_INCLUDE_DIR AND R_${_cpt}_LIBRARY)
    list(REMOVE_ITEM R_FIND_COMPONENTS ${_cpt})
  endif()
endforeach()

# Handle the QUIETLY and REQUIRED arguments and set R_FOUND to TRUE if all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(R DEFAULT_MSG R_EXECUTABLE R_INCLUDE_DIR R_LIBRARY)
mark_as_advanced(R_FOUND R_EXECUTABLE R_INCLUDE_DIR R_LIBRARY)

