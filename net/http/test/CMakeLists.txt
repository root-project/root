# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT net/http package
# @author Sergey Linev, GSI
############################################################################

find_program(CURL_EXECUTABLE curl)

# http server does not work on default Windows configuration
# so skip testing there
if(CURL_EXECUTABLE AND NOT WIN32)
  ROOT_ADD_GTEST(testHttpServer test_server.cxx LIBRARIES RHTTP RHTTPSniff Hist)
endif()
