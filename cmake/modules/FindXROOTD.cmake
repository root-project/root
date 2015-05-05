#
# Try to find XROOTD
# Once done this will define
#
#  XROOTD_FOUND - system has XROOTD
#  XROOTD_INCLUDE_DIR - the XROOTD include directory
#  XROOTD_INCLUDE_DIRS - with additonal include directories (non cached)
#  XROOTD_LIBRARIES - The libraries needed to use XROOTD
#  XROOTD_CFLAGS - Additional compilation flags (defines)
#  XROOTD_OLDPACK - old-style packaging for XROOTD libraries
#  XROOTD_NOMAIN - No main available: xproofd not build
#  XROOTD_NOOLDCLNT - No old client available: use built-in version
#

if(XROOTD_XrdClient_LIBRARY AND XROOTD_INCLUDE_DIR)
  set(XROOTD_FIND_QUIETLY TRUE)
endif()

set(searchpath ${XROOTD_ROOT_DIR} $ENV{XROOTD_ROOT_DIR} $ENV{XRDSYS} /opt/xrootd)

find_path(XROOTD_INCLUDE_DIR NAMES XrdVersion.hh
  HINTS ${searchpath}
  PATH_SUFFIXES include include/xrootd
)

if (XROOTD_INCLUDE_DIR)
  file(STRINGS ${XROOTD_INCLUDE_DIR}/XrdVersion.hh xrdvers REGEX "^#define XrdVERSION ")
  string(REGEX REPLACE "#define[ ]+XrdVERSION[ ]+" "" xrdvers ${xrdvers})
  string(REGEX REPLACE "[^v\\.]+" "" xrdversdots ${xrdvers})
  if (${xrdversdots} STREQUAL "v..")
    # Regular version string; parse it out
    string(REGEX MATCH "[0-9\\.]+" xrdvers ${xrdvers})
    string(REGEX REPLACE "^([^.]*)\\.(.*)\\.(.*)" "\\1" xrdversmajor ${xrdvers})
    string(REGEX REPLACE "^([^.]*)\\.(.*)\\.(.*)" "\\2" xrdversminor ${xrdvers})
    string(REGEX REPLACE "^([^.]*)\\.(.*)\\.(.*)" "\\3" xrdverspatch ${xrdvers})
    math(EXPR xrdversnum ${xrdversmajor}*100000000+${xrdversminor}*10000+${xrdverspatch})
  else ()
    # Old version string: we keep only the first numerics, i.e. the date
    string(REGEX REPLACE "[v\"]" "" xrdvers ${xrdvers})
    string(SUBSTRING ${xrdvers} 0 8 xrdversnum)
  endif ()
  if ( ${xrdversnum} EQUAL 300030000 )
     SET(XROOTD_FOUND FALSE)
     message(WARNING " >>> Cannot build with XRootD version 3.3.0: please install >=3.3.1 or <= 3.2.x")
  else()
     SET(XROOTD_FOUND TRUE)
  endif ()
endif()

if(XROOTD_FOUND)
  # Set include dirs and compiler macro variable

  if(NOT XROOTD_FIND_QUIETLY )
    message(STATUS "Found Xrootd version num: ${xrdvers} (setting -DROOTXRDVERS=${xrdversnum})")
  endif()
  set(XROOTD_CFLAGS "-DROOTXRDVERS=${xrdversnum}")

  if ( ${xrdversnum} LESS 300010000 )
     set(XROOTD_OLDPACK TRUE)
     set(XROOTD_INCLUDE_DIRS ${XROOTD_INCLUDE_DIR})
     message(STATUS "Setting OLDPACK TRUE")
  else()
     set(XROOTD_OLDPACK FALSE)
     find_path (XROOTD_INC_PRIV_DIR NAMES XrdClientConn.hh
        HINTS ${searchpath}
        PATH_SUFFIXES include/private/XrdClient include/xrootd/private/XrdClient
     )
     if (XROOTD_INC_PRIV_DIR)
        set(XROOTD_INCLUDE_DIRS ${XROOTD_INCLUDE_DIR} ${XROOTD_INCLUDE_DIR}/private)
     else()
        set(XROOTD_INCLUDE_DIRS ${XROOTD_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/proof/xrdinc)
     endif()
  endif()
endif()

if(XROOTD_FOUND)
  # Search for the required libraries; this depends on packaging ...

  if(XROOTD_OLDPACK)
    foreach(l XrdNet XrdOuc XrdSys XrdClient Xrd)
      find_library(XROOTD_${l}_LIBRARY
         NAMES ${l} 
         HINTS ${searchpath}
         PATH_SUFFIXES lib lib64)
      list(APPEND XROOTD_LIBRARIES ${XROOTD_${l}_LIBRARY})
    endforeach()

    if(${xrdversnum} GREATER 20100729)
      find_library(XROOTD_XrdNetUtil_LIBRARY
        NAMES XrdNetUtil
        HINTS ${searchpath}
        PATH_SUFFIXES lib lib64)
      list(APPEND XROOTD_LIBRARIES ${XROOTD_XrdNetUtil_LIBRARY})
    endif ()
  else()

    # libXrdMain (dropped in versions >= 4)
    find_library(XROOTD_XrdMain_LIBRARY
       NAMES XrdMain
       HINTS ${searchpath}
       PATH_SUFFIXES lib lib64)
    if (XROOTD_XrdMain_LIBRARY)
       list(APPEND XROOTD_LIBRARIES ${XROOTD_XrdMain_LIBRARY})
    else ()
       set(XROOTD_NOMAIN TRUE)
       if(NOT XROOTD_FIND_QUIETLY)
          message(STATUS "             libXrdMain not found: xproofd will be a wrapper around xrootd")
       endif () 
    endif ()

    # libXrdUtils
    find_library(XROOTD_XrdUtils_LIBRARY
       NAMES XrdUtils
       HINTS ${searchpath}
       PATH_SUFFIXES lib lib64)
    if (XROOTD_XrdUtils_LIBRARY)
       list(APPEND XROOTD_LIBRARIES ${XROOTD_XrdUtils_LIBRARY})
    endif ()

    # libXrdClient (old client; will be dropped at some point)
    find_library(XROOTD_XrdClient_LIBRARY
       NAMES XrdClient
       HINTS ${searchpath}
       PATH_SUFFIXES lib lib64)
    if (XROOTD_XrdClient_LIBRARY)
       list(APPEND XROOTD_LIBRARIES ${XROOTD_XrdClient_LIBRARY})
    else ()
       set(XROOTD_NOOLDCLNT TRUE)
       if(NOT XROOTD_FIND_QUIETLY)
          message(STATUS "             libXrdClient not found: use built-in")
       endif () 
    endif ()

    # libXrdCl
    if(${xrdversnum} GREATER 300030000)
       find_library(XROOTD_XrdCl_LIBRARY
          NAMES XrdCl
          HINTS ${searchpath}
          PATH_SUFFIXES lib lib64)
       if (XROOTD_XrdCl_LIBRARY)
          list(APPEND XROOTD_LIBRARIES ${XROOTD_XrdCl_LIBRARY})
       endif ()
    endif ()

  endif()

  if(XROOTD_LIBRARIES)
    set(XROOTD_FOUND TRUE)
    if(NOT XROOTD_FIND_QUIETLY )
      message(STATUS "             include_dirs: ${XROOTD_INCLUDE_DIRS}")
      message(STATUS "             libraries: ${XROOTD_LIBRARIES}")
    endif() 
  else ()
    set(XROOTD_FOUND FALSE)
  endif ()
endif()

mark_as_advanced(XROOTD_INCLUDE_DIR
                 XROOTD_XrdMain_LIBRARY
                 XROOTD_XrdUtils_LIBRARY
                 XROOTD_XrdClient_LIBRARY
                 XROOTD_XrdCl_LIBRARY
                 XROOTD_XrdNetUtil_LIBRARY
                 XROOTD_XrdNet_LIBRARY
                 XROOTD_XrdSys_LIBRARY
                 XROOTD_XrdOuc_LIBRARY
                 XROOTD_Xrd_LIBRARY )

