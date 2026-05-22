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

# curl is required for tests
find_program(CURL_EXECUTABLE curl)

if(NOT CURL_EXECUTABLE)
  return()
endif()

ROOT_ADD_GTEST(testHttpServer test_server.cxx LIBRARIES RHTTP RHTTPSniff Hist)

execute_process(
    COMMAND "${CURL_EXECUTABLE}" --version
    OUTPUT_VARIABLE CURL_CLI_OUTPUT
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

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
