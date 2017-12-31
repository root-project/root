#.rst:
# FindIPOPT
# ---------
#
# Try to locate the IPOPT library
#
# On non Windows systems, use pkg-config to try to locate the library,
# if this fails then try to locate the library in the directory pointed by
# the IPOPT_DIR enviromental variable.
#
# On Windows systems,  just try to find the library using the IPOPT_DIR
# enviromental variable.
#
# Create the following variables::
#
#  IPOPT_INCLUDE_DIRS - Directories to include to use IPOPT
#  IPOPT_LIBRARIES    - Default library to link against to use IPOPT
#  IPOPT_DEFINITIONS  - Flags to be added to linker's options
#  IPOPT_LINK_FLAGS   - Flags to be added to linker's options
#  IPOPT_FOUND        - If false, don't try to use IPOPT

#=============================================================================
# Copyright (C) 2008-2010 RobotCub Consortium
# Copyright (C) 2016 iCub Facility - Istituto Italiano di Tecnologia
#   Authors: Ugo Pattacini <ugo.pattacini@iit.it>
#   Authors: Daniele E. Domenichelli <daniele.domenichelli@iit.it>
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of YCM, substitute the full
#  License text for the above reference.)


if(NOT WIN32)
  # On non Windows systems we use PkgConfig to find IPOPT
  find_package(PkgConfig QUIET)
  if(PKG_CONFIG_FOUND)

    if(IPOPT_FIND_VERSION)
      if(IPOPT_FIND_VERSION_EXACT)
        pkg_check_modules(_PC_IPOPT QUIET ipopt=${IPOPT_FIND_VERSION})
      else()
        pkg_check_modules(_PC_IPOPT QUIET ipopt>=${IPOPT_FIND_VERSION})
      endif()
    else()
      pkg_check_modules(_PC_IPOPT QUIET ipopt)
    endif()


    if(_PC_IPOPT_FOUND)
      set(IPOPT_INCLUDE_DIRS ${_PC_IPOPT_INCLUDE_DIRS} CACHE PATH "IPOPT include directory")
      set(IPOPT_DEFINITIONS ${_PC_IPOPT_CFLAGS_OTHER} CACHE STRING "Additional compiler flags for IPOPT")
      set(IPOPT_LIBRARIES "" CACHE STRING "IPOPT libraries" FORCE)
      foreach(_LIBRARY IN ITEMS ${_PC_IPOPT_LIBRARIES})
        find_library(${_LIBRARY}_PATH
                     NAMES ${_LIBRARY}
                     PATHS ${_PC_IPOPT_LIBRARY_DIRS})
        # Workaround for https://github.com/robotology/icub-main/issues/418 
        if(${_LIBRARY}_PATH)
          list(APPEND IPOPT_LIBRARIES ${${_LIBRARY}_PATH})
        endif()
      endforeach()
    else()
      set(IPOPT_DEFINITIONS "")
    endif()

  endif()

  set(IPOPT_LINK_FLAGS "")

  # If pkg-config fails, try to find the package using IPOPT_DIR
  if(NOT _PC_IPOPT_FOUND)
    set(IPOPT_DIR_TEST $ENV{IPOPT_DIR})
    if(IPOPT_DIR_TEST)
      set(IPOPT_DIR $ENV{IPOPT_DIR} CACHE PATH "Path to IPOPT build directory")
    else()
      set(IPOPT_DIR /usr            CACHE PATH "Path to IPOPT build directory")
    endif()

    set(IPOPT_INCLUDE_DIRS ${IPOPT_DIR}/include/coin)
    find_library(IPOPT_LIBRARIES ipopt ${IPOPT_DIR}/lib
                                     ${IPOPT_DIR}/lib/coin
                                     NO_DEFAULT_PATH)

    if(IPOPT_LIBRARIES)
      find_file(IPOPT_DEP_FILE ipopt_addlibs_cpp.txt ${IPOPT_DIR}/share/doc/coin/Ipopt
                                                     ${IPOPT_DIR}/share/coin/doc/Ipopt
                                                     NO_DEFAULT_PATH)
      mark_as_advanced(IPOPT_DEP_FILE)

      if(IPOPT_DEP_FILE)
        # parse the file and acquire the dependencies
        file(READ ${IPOPT_DEP_FILE} IPOPT_DEP)
        string(REGEX REPLACE "-[^l][^ ]* " "" IPOPT_DEP ${IPOPT_DEP})
        string(REPLACE "-l"                "" IPOPT_DEP ${IPOPT_DEP})
        string(REPLACE "\n"                "" IPOPT_DEP ${IPOPT_DEP})
        string(REPLACE "ipopt"             "" IPOPT_DEP ${IPOPT_DEP})       # remove any possible auto-dependency
        separate_arguments(IPOPT_DEP)

        # use the find_library command in order to prepare rpath correctly
        foreach(LIB ${IPOPT_DEP})
          find_library(IPOPT_SEARCH_FOR_${LIB} ${LIB} ${IPOPT_DIR}/lib
                                                      ${IPOPT_DIR}/lib/coin
                                                      ${IPOPT_DIR}/lib/coin/ThirdParty
                                                      NO_DEFAULT_PATH)
          if(IPOPT_SEARCH_FOR_${LIB})
            # handle non-system libraries (e.g. coinblas)
            set(IPOPT_LIBRARIES ${IPOPT_LIBRARIES} ${IPOPT_SEARCH_FOR_${LIB}})
          else()
            # handle system libraries (e.g. gfortran)
            set(IPOPT_LIBRARIES ${IPOPT_LIBRARIES} ${LIB})
          endif()
          mark_as_advanced(IPOPT_SEARCH_FOR_${LIB})
        endforeach()
      endif()
    endif()

    set(IPOPT_DEFINITIONS "")
    set(IPOPT_LINK_FLAGS "")
  endif()

# Windows platforms
else()
  include(SelectLibraryConfigurations)

  set(IPOPT_DIR $ENV{IPOPT_DIR} CACHE PATH "Path to IPOPT build directory")

  set(IPOPT_INCLUDE_DIRS ${IPOPT_DIR}/include/coin)
  find_library(IPOPT_IPOPT_LIBRARY_RELEASE libipopt ${IPOPT_DIR}/lib
                                                    ${IPOPT_DIR}/lib/coin
                                                    NO_DEFAULT_PATH)
  find_library(IPOPT_IPOPT_LIBRARY_DEBUG   libipoptD ${IPOPT_DIR}/lib
                                                     ${IPOPT_DIR}/lib/coin
                                                     NO_DEFAULT_PATH)

  select_library_configurations(IPOPT_IPOPT)
  set(IPOPT_LIBRARIES ${IPOPT_IPOPT_LIBRARY})

  # Some old version of binary releases of IPOPT have Intel fortran
  # libraries embedded in the library, newer releases require them to
  # be explicitly linked.
  if(IPOPT_IPOPT_LIBRARY)
    get_filename_component(_MSVC_DIR "${CMAKE_LINKER}" DIRECTORY)

    # Find the lib.exe executable
    find_program(LIB_EXECUTABLE
                 NAMES lib.exe
                 HINTS "${_MSVC_BINDIR}"
                       "C:/Program Files/Microsoft Visual Studio 10.0/VC/bin"
                       "C:/Program Files (x86)/Microsoft Visual Studio 10.0/VC/bin"
                       "C:/Program Files/Microsoft Visual Studio 11.0/VC/bin"
                       "C:/Program Files (x86)/Microsoft Visual Studio 11.0/VC/bin"
                       "C:/Program Files/Microsoft Visual Studio 12.0/VC/bin"
                       "C:/Program Files (x86)/Microsoft Visual Studio 12.0/VC/bin"
                       "C:/Program Files/Microsoft Visual Studio 14.0/VC/bin"
                       "C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/bin"
                 DOC "Path to the lib.exe executable")
    mark_as_advanced(LIB_EXECUTABLE)

    # backup PATH environment variable
    set(_path $ENV{PATH})

    # Add th MSVC "Common7/IDE" dir containing the dlls in the PATH when needed.
    get_filename_component(_MSVC_LIBDIR "${_MSVC_BINDIR}/../../Common7/IDE" ABSOLUTE)
    if(NOT EXISTS "${_MSVC_LIBDIR}")
      get_filename_component(_MSVC_LIBDIR "${_MSVC_BINDIR}/../../../Common7/IDE" ABSOLUTE)
    endif()

    if(EXISTS "${_MSVC_LIBDIR}")
      set(_MSVC_LIBDIR_FOUND 0)
      file(TO_CMAKE_PATH "$ENV{PATH}" _env_path)
      foreach(_dir ${_env_path})
        if("${_dir}" STREQUAL ${_MSVC_LIBDIR})
          set(_MSVC_LIBDIR_FOUND 1)
        endif()
      endforeach()
      if(NOT _MSVC_LIBDIR_FOUND)
        file(TO_NATIVE_PATH "${_MSVC_LIBDIR}" _MSVC_LIBDIR)
        set(ENV{PATH} "$ENV{PATH};${_MSVC_LIBDIR}")
      endif()
    endif()

    if(IPOPT_IPOPT_LIBRARY_RELEASE)
      set(_IPOPT_LIB ${IPOPT_IPOPT_LIBRARY_RELEASE})
    else()
      set(_IPOPT_LIB ${IPOPT_IPOPT_LIBRARY_DEBUG})
    endif()

    execute_process(COMMAND ${LIB_EXECUTABLE} /list "${_IPOPT_LIB}"
                    OUTPUT_VARIABLE _lib_output)

    set(ENV{PATH} "${_path}")
    unset(_path)

    if(NOT "${_lib_output}" MATCHES "libifcoremd.dll")
      get_filename_component(_IPOPT_IPOPT_LIBRARY_DIR "${_IPOPT_LIB}" DIRECTORY)

      foreach(_lib ifconsol
                   libifcoremd
                   libifportmd
                   libmmd
                   libirc
                   svml_dispmd)
        string(TOUPPER "${_lib}" _LIB)
        find_library(IPOPT_${_LIB}_LIBRARY_RELEASE ${_lib} ${_IPOPT_IPOPT_LIBRARY_DIR})
        find_library(IPOPT_${_LIB}_LIBRARY_DEBUG ${_lib}d ${_IPOPT_IPOPT_LIBRARY_DIR})
        select_library_configurations(IPOPT_${_LIB})
        list(APPEND IPOPT_LIBRARIES ${IPOPT_${_LIB}_LIBRARY})
      endforeach()
    endif()
  endif()

  set(IPOPT_DEFINITIONS "")
  if(MSVC)
    set(IPOPT_LINK_FLAGS "/NODEFAULTLIB:libcmt.lib;libcmtd.lib")
  else()
    set(IPOPT_LINK_FLAGS "")
  endif()

endif()

mark_as_advanced(IPOPT_INCLUDE_DIRS
                 IPOPT_LIBRARIES
                 IPOPT_DEFINITIONS
                 IPOPT_LINK_FLAGS)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(IPOPT DEFAULT_MSG IPOPT_LIBRARIES)
