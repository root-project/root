# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# **PLEASE UPDATE ALSO THE FOLLOWING LINE WHEN UPDATING THE VERSION**
# 10 Apr 2023, https://github.com/civetweb/civetweb/releases/tag/v1.16
set(ROOT_CIVETWEB_VERSION 1.16)
set(ROOT_CIVETWEB_HASH "f0e471c1bf4e7804a6cfb41ea9d13e7d623b2bcc7bc1e2a4dd54951a24d60285")
set(ROOT_CIVETWEB_PREFIX ${CMAKE_BINARY_DIR}/builtins/CIVETWEB-prefix)
set(ROOT_CIVETWEB_LIBRARY ${ROOT_CIVETWEB_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}civetweb${CMAKE_STATIC_LIBRARY_SUFFIX})

if (NOT DEFINED GIT_EXECUTABLE)
  set(GIT_EXECUTABLE "git")
endif()

include(ExternalProject)

# Clear cache variables set by find_package(CIVETWEB)
# to ensure that we use the builtin version
foreach(var CIVETWEB_LIBRARIES CIVETWEB_LIBRARY CIVETWEB_LIBRARY_DEBUG CIVETWEB_LIBRARY_RELEASE CIVETWEB_FOUND CIVETWEB_VERSION CIVETWEB_INCLUDE_DIR CIVETWEB_LIBRARY CIVETWEB_LIBRARIES)
  unset(${var})
  unset(${var} CACHE)
endforeach()

if(WIN32 AND NOT CMAKE_GENERATOR MATCHES Ninja)
  if(winrtdebug)
    set(ROOT_CIVETWEB_BUILD_COMMAND_FLAGS "--config Debug")
  else()
    set(ROOT_CIVETWEB_BUILD_COMMAND_FLAGS "--config $<IF:$<CONFIG:Debug,RelWithDebInfo>,RelWithDebInfo,Release>")
  endif()
endif()

if(ssl)
  if(OPENSSL_VERSION)
    string(REPLACE "." ";" lst ${OPENSSL_VERSION})
    list(GET lst 0 ssl_major)
    list(GET lst 1 ssl_minor)
  else()
    MESSAGE(SEND_ERROR "No openSSL version defined.")
  endif()

  if((${ssl_major} EQUAL "1") AND (${ssl_minor} EQUAL "1"))
    MESSAGE(STATUS "Use SSL API VERSION 1.1 for civetweb")
    set(ROOT_SSL_API "-DCIVETWEB_SSL_OPENSSL_API_1_1=ON" "-DCIVETWEB_SSL_OPENSSL_API_3_0=OFF" "-DCIVETWEB_SSL_OPENSSL_API_1_0=OFF" "-DCIVETWEB_ENABLE_SSL=ON")
  elseif((${ssl_major} EQUAL "3"))
    MESSAGE(STATUS "Use SSL API VERSION 3.${ssl_minor} for civetweb")
    set(ROOT_SSL_API "-DCIVETWEB_SSL_OPENSSL_API_3_0=ON" "-DCIVETWEB_SSL_OPENSSL_API_1_1=OFF" "-DCIVETWEB_SSL_OPENSSL_API_1_0=OFF" "-DCIVETWEB_ENABLE_SSL=ON")
  elseif((${ssl_major} EQUAL "1") AND (${ssl_minor} EQUAL "0"))
    MESSAGE(STATUS "Use SSL API VERSION 1.0 for civetweb")
    set(ROOT_SSL_API "-DCIVETWEB_SSL_OPENSSL_API_1_0=ON" "-DCIVETWEB_SSL_OPENSSL_API_3_0=OFF" "-DCIVETWEB_SSL_OPENSSL_API_1_1=OFF" "-DCIVETWEB_ENABLE_SSL=ON")
  else()
    MESSAGE(WARNING "Not able to recognize SSL version ${OPENSSL_VERSION}, disable SSL")
    set(ROOT_SSL_API "-DCIVETWEB_ENABLE_SSL=OFF")
  endif()
else()
  set(ROOT_SSL_API "-DCIVETWEB_ENABLE_SSL=OFF")
endif()
if(NOT MSVC)
  set(ROOT_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_X_DOM_SOCKET")
else()
  set(ROOT_C_FLAGS ${CMAKE_C_FLAGS})
endif()
if (CMAKE_OSX_SYSROOT)
  set(ROOT_C_FLAGS "${ROOT_C_FLAGS} -isysroot ${CMAKE_OSX_SYSROOT}")
endif()

ExternalProject_Add(BUILTIN_CIVETWEB
  PREFIX ${ROOT_CIVETWEB_PREFIX}
  # URL ${lcgpackages}/civetweb-${ROOT_CIVETWEB_VERSION}.tar.gz
  URL https://github.com/civetweb/civetweb/archive/refs/tags/v${ROOT_CIVETWEB_VERSION}.tar.gz # TODO move to LCG
  URL_HASH SHA256=${ROOT_CIVETWEB_HASH}

  LOG_DOWNLOAD  TRUE
  LOG_CONFIGURE FALSE
  LOG_BUILD     FALSE
  LOG_INSTALL   FALSE
  LOG_OUTPUT_ON_FAILURE FALSE
  USES_TERMINAL_DOWNLOAD true
  USES_TERMINAL_UPDATE true
  USES_TERMINAL_BUILD true
  USES_TERMINAL_INSTALL true
  USES_TERMINAL_TEST true
  INSTALL_DIR ${ROOT_CIVETWEB_PREFIX}
  PATCH_COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/0001-message-libs.patch
  CMAKE_ARGS -G ${CMAKE_GENERATOR}
    -DCMAKE_VERBOSE_MAKEFILE=ON
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
    -DCMAKE_INSTALL_LIBDIR:PATH=<INSTALL_DIR>/lib
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_FLAGS=${ROOT_C_FLAGS}
    -DCMAKE_CXX_FLAGS=${ROOT_EXTERNAL_CXX_FLAGS}
    -DBUILD_SHARED_LIBS:BOOL=FALSE
    -DBUILD_TESTING=FALSE
    -DCIVETWEB_ENABLE_ASAN=OFF # If set to ON, you need to set below link_library interface to UBSAN libs
    -DCIVETWEB_ENABLE_SSL_DYNAMIC_LOADING=OFF # If set to ON, you need to set below link_library interface to CMAKE_DL_LIBS
    -DCIVETWEB_BUILD_TESTING=FALSE
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=TRUE
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    -DCIVETWEB_ENABLE_WEBSOCKETS=ON
    -DOPENSSL_INCLUDE_DIR=${OPENSSL_INCLUDE_DIR}
    #-DOPENSSL_CRYPTO_LIBRARY=${OPENSSL_CRYPTO_LIBRARY}
    -DOPENSSL_SSL_LIBRARY=${OPENSSL_SSL_LIBRARY}
    #-DOPENSSL_LIBRARIES=${OPENSSL_LIBRARIES}
    -DCMAKE_PREFIX_PATH:STRING=${OPENSSL_PREFIX}
    -DOPENSSL_ROOT_DIR=${OPENSSL_ROOT_DIR}
    -DOPENSSL_FOUND=TRUE
    -DOpenSSL_FOUND=TRUE
    -DOpenSSL_ROOT=${OPENSSL_ROOT} #For CMake <3.27
    -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}
    -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
    # -DCMAKE_INSTALL_RPATH:STRING=${rpath_origin}
    ${ROOT_SSL_API}
  BUILD_COMMAND ${CMAKE_COMMAND} --build . ${ROOT_CIVETWEB_BUILD_COMMAND_FLAGS} -- -j 1 -v
  INSTALL_COMMAND ${CMAKE_COMMAND} --build . ${ROOT_CIVETWEB_BUILD_COMMAND_FLAGS} --target install -v
  BUILD_BYPRODUCTS
    ${ROOT_CIVETWEB_LIBRARY}
  TIMEOUT 600
)

file(MAKE_DIRECTORY ${ROOT_CIVETWEB_PREFIX}/include)
add_library(civetweb::civetweb IMPORTED STATIC GLOBAL)
add_dependencies(civetweb::civetweb BUILTIN_CIVETWEB)
if(builtin_openssl)
  add_dependencies(BUILTIN_CIVETWEB BUILTIN_OPENSSL)
endif()
set_target_properties(civetweb::civetweb PROPERTIES
    IMPORTED_LOCATION ${ROOT_CIVETWEB_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${ROOT_CIVETWEB_PREFIX}/include)
target_compile_definitions(civetweb::civetweb INTERFACE CIVETWEB_LIBRARY_STATIC) # needed for Win32 since public flag is not correctly propagated to parent scope (BUILD_SHARED_LIBS works fine for building but when installing, flag info is lost)
if (ssl)
  target_link_libraries(civetweb::civetweb INTERFACE OpenSSL::SSL)
endif()

# Set the canonical output of find_package according to
# https://cmake.org/cmake/help/latest/manual/cmake-developer.7.html#standard-variable-names
set(civetweb_INCLUDE_DIRS ${ROOT_CIVETWEB_PREFIX}/include PARENT_SCOPE)
set(civetweb_LIBRARIES ${ROOT_CIVETWEB_LIBRARY} PARENT_SCOPE)
set(civetweb_FOUND TRUE PARENT_SCOPE)
set(civetweb_VERSION ${ROOT_CIVETWEB_VERSION} PARENT_SCOPE)
