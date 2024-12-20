# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

include(ExternalProject)

set(XROOTD_VERSION "5.7.2")

set(XROOTD_PREFIX ${CMAKE_BINARY_DIR})
message(STATUS "Downloading and building XROOTD version ${XROOTD_VERSION}")

list(REMOVE_ITEM XROOTD_CLIENT_LIBRARIES OpenSSL::SSL)
list(REMOVE_ITEM XROOTD_UTILS_LIBRARIES OpenSSL::SSL)

set(libname ${CMAKE_SHARED_LIBRARY_PREFIX}XrdCl${CMAKE_SHARED_LIBRARY_SUFFIX})
list(APPEND XROOTD_CLIENT_LIBRARIES ${XROOTD_PREFIX}/lib/${libname})
list(REMOVE_DUPLICATES XROOTD_CLIENT_LIBRARIES)

set(libname ${CMAKE_SHARED_LIBRARY_PREFIX}XrdUtils${CMAKE_SHARED_LIBRARY_SUFFIX})
list(APPEND XROOTD_UTILS_LIBRARIES ${XROOTD_PREFIX}/lib/${libname})
list(REMOVE_DUPLICATES XROOTD_UTILS_LIBRARIES)

ExternalProject_Add(
    BUILTIN_XROOTD
    URL http://lcgpackages.web.cern.ch/lcgpackages/tarFiles/sources/xrootd-${XROOTD_VERSION}.tar.gz
    URL_HASH SHA256=c14c537edc66824ad3ca3c610240f9386c68993cbbcd28473ad3b42c8d14ba67
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
               -DCMAKE_INSTALL_LIBDIR=<INSTALL_DIR>/lib
               -DENABLE_PYTHON=OFF
               -DENABLE_CEPH=OFF
               -DXRDCL_LIB_ONLY=ON
               -DCMAKE_INSTALL_RPATH:STRING=${XROOTD_PREFIX}/lib
               -DOPENSSL_ROOT_DIR=${OPENSSL_ROOT_DIR}
    INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
    LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1 LOG_OUTPUT_ON_FAILURE 1
    BUILD_BYPRODUCTS ${XROOTD_CLIENT_LIBRARIES} ${XROOTD_UTILS_LIBRARIES}
    TIMEOUT 600
)

# CMake checks for existence when a target is linked to XRootD
file(MAKE_DIRECTORY ${XROOTD_PREFIX}/include/xrootd)

if(builtin_openssl)
  add_dependencies(BUILTIN_XROOTD OPENSSL)
endif()

list(APPEND XROOTD_CLIENT_LIBRARIES OpenSSL::SSL)
list(REMOVE_DUPLICATES XROOTD_CLIENT_LIBRARIES)
list(APPEND XROOTD_UTILS_LIBRARIES OpenSSL::SSL)
list(REMOVE_DUPLICATES XROOTD_UTILS_LIBRARIES)

set(XROOTD_INCLUDE_DIRS ${XROOTD_PREFIX}/include/xrootd CACHE INTERNAL "" FORCE)
set(XROOTD_CLIENT_LIBRARIES ${XROOTD_CLIENT_LIBRARIES} CACHE INTERNAL "" FORCE)
set(XROOTD_UTILS_LIBRARIES ${XROOTD_UTILS_LIBRARIES} CACHE INTERNAL "" FORCE)

list(APPEND CMAKE_BUILD_RPATH ${XROOTD_PREFIX}/lib)
add_dependencies(XRootD BUILTIN_XROOTD)

set_property(GLOBAL APPEND PROPERTY ROOT_BUILTIN_TARGETS BUILTIN_XROOTD)

install(DIRECTORY ${XROOTD_PREFIX}/lib/ DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries FILES_MATCHING PATTERN "libXrd*")
install(DIRECTORY ${XROOTD_PREFIX}/include/xrootd/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/xrootd COMPONENT headers)
if(APPLE)
  # XRootD libraries on mac need the LC_RPATH variable set. The build process already takes care of setting
  #   * BUILD_RPATH = build/XROOTD-prefix/../src
  #   * INSTALL_RPATH = build/lib
  # Since the install directory for the builtin_xrootd target corresponds to the build directory of the main project.
  # Use a post install script to change the LC_RPATH variable of the libraries in the ROOT install folder.
  install(SCRIPT ${CMAKE_CURRENT_LIST_DIR}/XROOTDApplePostInstall.cmake
          CODE "xrootd_libs_change_rpath(${XROOTD_PREFIX}/lib ${CMAKE_INSTALL_FULL_LIBDIR})"
  )
endif()
