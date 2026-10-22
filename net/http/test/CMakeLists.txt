# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT net/http/test package
# @author Sergey Linev, GSI
############################################################################

# http server does not work on Windows without extra firewall configuration
if(WIN32)
  return()
endif()

# curl and gunzip are required for tests
find_program(CURL_EXECUTABLE curl)
find_program(GUNZIP_EXECUTABLE gunzip)

if(NOT CURL_EXECUTABLE OR NOT GUNZIP_EXECUTABLE)
  return()
endif()

ROOT_ADD_GTEST(testHttpServer test_server.cxx LIBRARIES RHTTP RHTTPSniff Hist)

execute_process(
    COMMAND "${CURL_EXECUTABLE}" --version
    OUTPUT_VARIABLE CURL_CLI_OUTPUT
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(ssl)
  find_program(OPENSSL_EXECUTABLE openssl)
  if(OPENSSL_EXECUTABLE)
    execute_process(
        COMMAND ${OPENSSL_EXECUTABLE} version
        OUTPUT_VARIABLE OPENSSL_OUTPUT
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(OPENSSL_OUTPUT MATCHES "OpenSSL ([0-9]+\\.[0-9]+\\.[0-9]+[a-z]*)")
      set(OPENSSL_VERSION "${CMAKE_MATCH_1}")
    endif()
    # openssl 4.0 fails to accept self-signed certificate, to be investigated later
    if(OPENSSL_VERSION VERSION_GREATER_EQUAL "1.1.0" AND OPENSSL_VERSION VERSION_LESS "4.0.0")
      message(STATUS "Detect OpenSSL ${OPENSSL_VERSION} for http server tests")
      ROOT_ADD_GTEST(testHttpsServer test_ssl_server.cxx LIBRARIES RHTTP RHTTPSniff Hist)
    else()
      message(STATUS "Detected OpenSSL ${OPENSSL_VERSION} cannot be used for http server tests")
    endif()
  endif()
endif()

# only newer curl support websockets - so test version
# curl introduce support from 8.11 but it works reliably only with 8.18
if(CURL_CLI_OUTPUT MATCHES "curl ([0-9]+\\.[0-9]+\\.[0-9]+)")
  set(CURL_CLI_VERSION "${CMAKE_MATCH_1}")
  message(STATUS "Found curl ${CURL_CLI_VERSION} for http server tests")
  if(CURL_CLI_VERSION VERSION_GREATER_EQUAL "7.50.0")
    ROOT_ADD_GTEST(testUnixSocket test_xsocket.cxx LIBRARIES RHTTP RHTTPSniff Hist)
  endif()
  if(CURL_CLI_VERSION VERSION_GREATER_EQUAL "8.18.0")
    ROOT_ADD_GTEST(testWebSocket test_websocket.cxx LIBRARIES RHTTP Hist)
  endif()
endif()
