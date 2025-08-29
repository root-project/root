# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

include(ExternalProject)

set(XROOTD_VERSION "5.8.0")
set(XROOTD_PREFIX ${CMAKE_BINARY_DIR}/XROOTD-prefix)

message(STATUS "Downloading and building XROOTD version ${XROOTD_VERSION}")

set(XROOTD_INCLUDE_DIRS ${XROOTD_PREFIX}/include/xrootd CACHE INTERNAL "" FORCE)
set(XRDCL_NAME ${CMAKE_SHARED_LIBRARY_PREFIX}XrdCl${CMAKE_SHARED_LIBRARY_SUFFIX})
set(XRDUTILS_NAME ${CMAKE_SHARED_LIBRARY_PREFIX}XrdUtils${CMAKE_SHARED_LIBRARY_SUFFIX})
set(XROOTD_CLIENT_LIBRARIES ${XROOTD_PREFIX}/lib/${XRDCL_NAME} CACHE INTERNAL "" FORCE)
set(XROOTD_UTILS_LIBRARIES ${XROOTD_PREFIX}/lib/${XRDUTILS_NAME} CACHE INTERNAL "" FORCE)
set(XROOTD_LIBRARIES ${XROOTD_PREFIX}/lib/${XRDCL_NAME} CACHE INTERNAL "" FORCE)

if(APPLE)
  set(rpath_origin "@loader_path")
else()
  set(rpath_origin "$ORIGIN")
endif()

ExternalProject_Add(
    BUILTIN_XROOTD
    URL http://lcgpackages.web.cern.ch/lcgpackages/tarFiles/sources/xrootd-${XROOTD_VERSION}.tar.gz
    URL_HASH SHA256=fb5810e18670759292d22a3b5c92d21e1c301e999f319340fd84a6bc8ada6ca2
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
               # The INSTALL_RPATH has to be relative, so we can relocate the
               # XRootD install to the ROOT install directory:
               -DCMAKE_INSTALL_RPATH:STRING=${rpath_origin}
               -DOpenSSL_ROOT=${OPENSSL_ROOT} #For CMake <3.27
               -DOPENSSL_ROOT=${OPENSSL_ROOT}
    INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
    LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1 LOG_OUTPUT_ON_FAILURE 1
    BUILD_BYPRODUCTS ${XROOTD_UTILS_LIBRARIES} ${XROOTD_CLIENT_LIBRARIES}
    TIMEOUT 600
)

# CMake checks for existence when a target is linked to XRootD
file(MAKE_DIRECTORY ${XROOTD_PREFIX}/include/xrootd)

if(builtin_openssl)
  add_dependencies(BUILTIN_XROOTD OPENSSL)
endif()

install(DIRECTORY ${XROOTD_PREFIX}/lib/ DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries FILES_MATCHING PATTERN "libXrd*")
install(DIRECTORY ${XROOTD_PREFIX}/include/xrootd/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/xrootd COMPONENT headers)
set(CMAKE_BUILD_RPATH ${CMAKE_BUILD_RPATH} ${XROOTD_PREFIX}/lib PARENT_SCOPE)

set_property(GLOBAL APPEND PROPERTY ROOT_BUILTIN_TARGETS BUILTIN_XROOTD)
