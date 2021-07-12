# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

include(ExternalProject)

find_package(OpenSSL REQUIRED)

set(XROOTD_VERSION "5.5.2")
set(XROOTD_VERSIONNUM 500050002)
set(lcgpackages http://lcgpackages.web.cern.ch/lcgpackages/tarFiles/sources)
set(XROOTD_SRC_URI ${lcgpackages}/xrootd-${XROOTD_VERSION}.tar.gz)
set(XROOTD_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/XROOTD-prefix)
message(STATUS "Downloading and building XROOTD version ${XROOTD_VERSION}")

# Guess under which directory XRootD will install its libraires
set(XROOTD_LIBDIR "lib")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND ${CMAKE_SIZEOF_VOID_P} EQUAL 8
AND NOT CMAKE_CROSSCOMPILING AND NOT EXISTS "/etc/debian_version")
  set(XROOTD_LIBDIR "lib64")
endif()

set(libname ${CMAKE_SHARED_LIBRARY_PREFIX}XrdCl${CMAKE_SHARED_LIBRARY_SUFFIX})
list(APPEND XROOTD_CLIENT_LIBRARIES ${XROOTD_PREFIX}/${XROOTD_LIBDIR}/${libname})

set(libname ${CMAKE_SHARED_LIBRARY_PREFIX}XrdUtils${CMAKE_SHARED_LIBRARY_SUFFIX})
list(APPEND XROOTD_UTILS_LIBRARIES ${XROOTD_PREFIX}/${XROOTD_LIBDIR}/${libname})

ExternalProject_Add(
    BUILTIN_XROOTD
    URL ${XROOTD_SRC_URI}
    URL_HASH SHA256=ec4e0490b8ee6a3254a4ea4449342aa364bc95b78dc9a8669151be30353863c6
    INSTALL_DIR ${XROOTD_PREFIX}
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
               -DCMAKE_PREFIX_PATH:STRING=${OPENSSL_PREFIX}
               -DCMAKE_BUILD_TYPE=Release
               -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_CXX_FLAGS=${ROOT_EXTERNAL_CXX_FLAGS}
               -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}
               -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
               -DENABLE_PYTHON=OFF
               -DENABLE_CEPH=OFF
               -DXRDCL_LIB_ONLY=ON
               ${XROOTD_WITH_OPENSSL3}
               -DCMAKE_INSTALL_RPATH:STRING=${XROOTD_PREFIX}/${XROOTD_LIBDIR}
    INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
    LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1
    BUILD_BYPRODUCTS ${XROOTD_CLIENT_LIBRARIES} ${XROOTD_UTILS_LIBRARIES}
    TIMEOUT 600
)

# CMake checks for existence when a target is linked to XRootD
file(MAKE_DIRECTORY ${XROOTD_PREFIX}/include/xrootd)

if(builtin_openssl)
  add_dependencies(BUILTIN_XROOTD OPENSSL)
endif()

list(APPEND XROOTD_CLIENT_LIBRARIES OpenSSL::SSL)
list(APPEND XROOTD_UTILS_LIBRARIES OpenSSL::SSL)

set(XROOTD_INCLUDE_DIRS ${XROOTD_PREFIX}/include/xrootd CACHE INTERNAL "" FORCE)
set(XROOTD_CLIENT_LIBRARIES ${XROOTD_CLIENT_LIBRARIES} CACHE INTERNAL "" FORCE)
set(XROOTD_UTILS_LIBRARIES ${XROOTD_UTILS_LIBRARIES} CACHE INTERNAL "" FORCE)

list(APPEND CMAKE_BUILD_RPATH ${XROOTD_PREFIX}/${XROOTD_LIBDIR})
add_dependencies(XRootD BUILTIN_XROOTD)

set_property(GLOBAL APPEND PROPERTY ROOT_BUILTIN_TARGETS BUILTIN_XROOTD)

install(DIRECTORY ${XROOTD_PREFIX}/${XROOTD_LIBDIR}/ DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries FILES_MATCHING PATTERN "libXrd*")
install(DIRECTORY ${XROOTD_PREFIX}/include/xrootd/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT headers)
if(APPLE)
  # XRootD libraries on mac need the LC_RPATH variable set. The build process already takes care of setting
  #   * BUILD_RPATH = build/XROOTD-prefix/../src
  #   * INSTALL_RPATH = build/lib
  # Since the install directory for the builtin_xrootd target corresponds to the build directory of the main project.
  # Use a post install script to change the LC_RPATH variable of the libraries in the ROOT install folder.
  install(SCRIPT ${CMAKE_CURRENT_LIST_DIR}/XROOTDApplePostInstall.cmake
          CODE "xrootd_libs_change_rpath(${XROOTD_PREFIX}/${XROOTD_LIBDIR} ${CMAKE_INSTALL_FULL_LIBDIR})"
  )
endif()
