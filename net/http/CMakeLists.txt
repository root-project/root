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

set(RHTTP_HEADERS
  THttpCallArg.h
  THttpEngine.h
  THttpServer.h
  THttpWSHandler.h
  TRootSniffer.h
  TRootSnifferStore.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(RHTTP
  HEADERS
    ${RHTTP_HEADERS}
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
    civetweb::civetweb
    Thread
  DEPENDENCIES
    RIO
)

set(RELATIVE_INC_HEADERS ${RHTTP_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(RHTTP
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/ src/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
    src/TCivetweb.h
    src/TFastCgi.h
    src/THttpLongPollEngine.h
    src/THttpWSEngine.h
)

if(TARGET FastCGI::FastCGI)
  target_link_libraries(RHTTP PRIVATE FastCGI::FastCGI)
else()
  target_compile_definitions(RHTTP PUBLIC -DHTTP_WITHOUT_FASTCGI)
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
