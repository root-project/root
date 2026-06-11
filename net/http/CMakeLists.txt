# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT net/http package
# @author Pere Mato, CERN
# @author Sergey Linev, GSI
############################################################################

if(NOT FASTCGI_FOUND)
  set(FASTCGI_LIBRARY "")
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(RHTTP
  HEADERS
    THttpCallArg.h
    THttpEngine.h
    THttpServer.h
    THttpWSHandler.h
    TRootSniffer.h
    TRootSnifferStore.h
  SOURCES
    src/TCivetweb.cxx
    src/TCivetweb.h
    src/TFastCgi.cxx
    src/TFastCgi.h
    src/THttpCallArg.cxx
    src/THttpEngine.cxx
    src/THttpLongPollEngine.cxx
    src/THttpLongPollEngine.h
    src/THttpServer.cxx
    src/THttpWSEngine.cxx
    src/THttpWSEngine.h
    src/THttpWSHandler.cxx
    src/TRootSniffer.cxx
    src/TRootSnifferStore.cxx
  LIBRARIES
    ZLIB::ZLIB
    civetweb::civetweb
  DEPENDENCIES
    RIO
    Thread
)

if(TARGET FastCGI::FastCGI)
  target_link_libraries(RHTTP PRIVATE FastCGI::FastCGI)
else()
  target_compile_definitions(RHTTP PUBLIC -DHTTP_WITHOUT_FASTCGI)
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
